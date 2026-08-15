// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;

use crate::st::{Aligned, Dtype};

pub const TRUNK_ALIGN: i64 = 4096;
const HUGE: usize = 2 << 20;
const RING_WANT: usize = 2;

pub struct TrunkTensor {
    pub off: i64,
    pub nbytes: i64,
    pub dtype: Dtype,
}

pub struct TrunkLayer {
    pub file_off: i64,
    pub nbytes: i64,
    pub tensors: HashMap<String, TrunkTensor>,
}

impl TrunkLayer {
    pub fn find(&self, name: &str) -> Option<&TrunkTensor> {
        self.tensors.get(name)
    }
}

fn align_up(v: i64, a: i64) -> i64 {
    (v + a - 1) & !(a - 1)
}

fn slot(len: usize) -> Aligned {
    let a = if std::env::var_os("K3_NOHUGE").is_some() { 4096 } else { HUGE };
    let b = Aligned::with_align(len, a);
    #[cfg(target_os = "linux")]
    if a == HUGE {
        let _ = unsafe {
            rustix::mm::madvise(b.as_ptr().cast(), b.len(), rustix::mm::Advice::LinuxHugepage)
        };
    }
    b
}

struct Spans {
    file: File,
    at: Vec<(i64, i64)>,
    bytes: Arc<AtomicU64>,
    micros: Arc<AtomicU64>,
}

impl Spans {
    fn read(&self, l: usize, dst: &mut [u8]) -> io::Result<()> {
        let (off, n) = self.at[l];
        let n = n as usize;
        if dst.len() < n {
            return Err(io::Error::other(format!("k3_trunk: slot holds {} of {n}", dst.len())));
        }
        let t0 = Instant::now();
        let mut got = 0usize;
        while got < n {
            match self.file.read_at(&mut dst[got..n], (off + got as i64) as u64) {
                Ok(0) => return Err(io::Error::other(format!("k3_trunk: short read on layer {l}"))),
                Ok(k) => got += k,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        self.micros.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        self.bytes.fetch_add(got as u64, Ordering::Relaxed);
        Ok(())
    }
}

type Job = (usize, Aligned);
type Done = (usize, Aligned, io::Result<()>);

pub struct Trunk {
    spans: Arc<Spans>,
    pub n_layers: usize,
    pub lay: Vec<TrunkLayer>,
    pub direct: bool,
    pin: Vec<Aligned>,
    pin_loaded: Vec<bool>,
    npin: usize,
    slot_bytes: usize,
    widen_bytes: usize,
    held: Option<(usize, Aligned)>,
    spare: Option<Aligned>,
    inflight: Option<usize>,
    req: Option<Sender<Job>>,
    done: Option<Receiver<Done>>,
    worker: Option<std::thread::JoinHandle<()>>,
    pub hits: u64,
    pub misses: u64,
    bytes: Arc<AtomicU64>,
    micros: Arc<AtomicU64>,
}

fn dt_of(s: &str) -> Dtype {
    match s {
        "BF16" => Dtype::Bf16,
        "F32" => Dtype::F32,
        "U8" => Dtype::U8,
        "F16" => Dtype::F16,
        "I8R" => Dtype::I8R,
        _ => Dtype::U8,
    }
}

impl Trunk {
    pub fn open(dir: &Path, widen: usize, budget: i64) -> io::Result<Trunk> {
        let txt = std::fs::read_to_string(dir.join("trunk.json"))?;
        let root: Value = serde_json::from_str(&txt).map_err(io::Error::other)?;
        let arr = root["layers"].as_array().ok_or_else(|| io::Error::other("k3_trunk: no layers array"))?;

        let mut lay = Vec::with_capacity(arr.len());
        for (i, e) in arr.iter().enumerate() {
            let ts = e["tensors"].as_object().ok_or_else(|| {
                io::Error::other(format!("k3_trunk: layer {i} has no tensors"))
            })?;
            lay.push(TrunkLayer {
                file_off: e["file_off"].as_i64().unwrap_or(0),
                nbytes: e["nbytes"].as_i64().unwrap_or(0),
                tensors: ts
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            TrunkTensor {
                                off: v["off"].as_i64().unwrap_or(0),
                                nbytes: v["nbytes"].as_i64().unwrap_or(0),
                                dtype: v["dtype"].as_str().map_or(Dtype::U8, dt_of),
                            },
                        )
                    })
                    .collect(),
            });
        }
        let n_layers = lay.len();
        if n_layers == 0 {
            return Err(io::Error::other("k3_trunk: empty layers array"));
        }

        let bin = dir.join("trunk.bin");
        let want_align = root["align"].as_i64().unwrap_or(0);
        let mut direct = want_align == TRUNK_ALIGN;
        let file = match direct.then(|| open_direct(&bin)).flatten() {
            Some(f) => f,
            None => {
                direct = false;
                File::open(&bin)?
            }
        };

        let (mut ring, mut slot_bytes, mut npin) = (RING_WANT, 0i64, 0usize);
        for _ in 0..4 {
            let big = lay[npin..].iter().map(|l| l.nbytes).max().unwrap_or(lay[n_layers - 1].nbytes);
            let rs = align_up(align_up(big, TRUNK_ALIGN) + widen as i64, 4096);
            ring = RING_WANT;
            while ring > 1 && ring as i64 * rs > budget {
                ring -= 1;
            }
            let mut sp = ring as i64 * rs;
            let mut np = 0usize;
            while np < n_layers && sp + lay[np].nbytes + widen as i64 <= budget {
                sp += lay[np].nbytes + widen as i64;
                np += 1;
            }
            if rs == slot_bytes && np == npin {
                break;
            }
            slot_bytes = rs;
            npin = np;
        }

        let bytes = Arc::new(AtomicU64::new(0));
        let micros = Arc::new(AtomicU64::new(0));
        let spans = Arc::new(Spans {
            file,
            at: lay.iter().map(|l| (l.file_off, l.nbytes)).collect(),
            bytes: bytes.clone(),
            micros: micros.clone(),
        });

        let pin: Vec<Aligned> = (0..npin)
            .map(|i| slot(align_up(lay[i].nbytes, TRUNK_ALIGN) as usize + widen))
            .collect();

        let mut tr = Trunk {
            spans,
            n_layers,
            lay,
            direct,
            pin,
            pin_loaded: vec![false; npin],
            npin,
            slot_bytes: slot_bytes as usize,
            widen_bytes: widen,
            held: None,
            spare: Some(slot(slot_bytes as usize)),
            inflight: None,
            req: None,
            done: None,
            worker: None,
            hits: 0,
            misses: 0,
            bytes,
            micros,
        };

        if ring >= 2 {
            tr.held = Some((usize::MAX, slot(slot_bytes as usize)));
            let (rq, rq_rx) = std::sync::mpsc::channel::<Job>();
            let (dn_tx, dn) = std::sync::mpsc::channel::<Done>();
            let sp = tr.spans.clone();
            tr.worker = Some(std::thread::spawn(move || {
                while let Ok((l, mut buf)) = rq_rx.recv() {
                    let r = sp.read(l, &mut buf);
                    if dn_tx.send((l, buf, r)).is_err() {
                        break;
                    }
                }
            }));
            tr.req = Some(rq);
            tr.done = Some(dn);
        }
        Ok(tr)
    }

    pub fn nslot(&self) -> usize {
        1 + usize::from(self.held.is_some()) + usize::from(self.spare.is_some())
            - usize::from(self.held.is_some() && self.spare.is_some())
    }

    pub fn npin(&self) -> usize {
        self.npin
    }

    pub fn run_len(&self, l: usize) -> usize {
        align_up(self.lay[l].nbytes, TRUNK_ALIGN) as usize
    }

    pub fn widen_bytes(&self) -> usize {
        self.widen_bytes
    }

    fn drain(&mut self) {
        if let (Some(l), Some(dn)) = (self.inflight.take(), self.done.as_ref()) {
            if let Ok((_, buf, _)) = dn.recv() {
                let _ = l;
                self.spare = Some(buf);
            }
        }
    }

    pub fn prefetch(&mut self, l: usize) {
        if l >= self.n_layers || l < self.npin || self.inflight.is_some() {
            return;
        }
        if self.held.as_ref().is_some_and(|(h, _)| *h == l) {
            return;
        }
        let (Some(buf), Some(rq)) = (self.spare.take(), self.req.as_ref()) else {
            return;
        };
        if rq.send((l, buf)).is_ok() {
            self.inflight = Some(l);
        }
    }

    pub fn layer(&mut self, l: usize) -> io::Result<&mut [u8]> {
        if l >= self.n_layers {
            return Err(io::Error::other(format!("k3_trunk: layer {l} out of range")));
        }
        if l < self.npin {
            if !self.pin_loaded[l] {
                let mut b = std::mem::replace(&mut self.pin[l], Aligned::with_align(1, 4096));
                let r = self.spans.read(l, &mut b);
                self.pin[l] = b;
                r?;
                self.pin_loaded[l] = true;
                self.misses += 1;
            } else {
                self.hits += 1;
            }
            return Ok(&mut self.pin[l][..]);
        }

        if self.inflight == Some(l) {
            self.inflight = None;
            let (_, buf, r) = self
                .done
                .as_ref()
                .unwrap()
                .recv()
                .map_err(|_| io::Error::other("k3_trunk: reader died"))?;
            let old = self.held.replace((l, buf));
            self.spare = old.map(|(_, b)| b);
            r?;
            self.misses += 1;
            return Ok(&mut self.held.as_mut().unwrap().1[..]);
        }

        if self.held.as_ref().is_some_and(|(h, _)| *h == l) {
            self.hits += 1;
            return Ok(&mut self.held.as_mut().unwrap().1[..]);
        }

        self.drain();
        let mut buf = match self.spare.take().or_else(|| self.held.take().map(|(_, b)| b)) {
            Some(b) => b,
            None => slot(self.slot_bytes),
        };
        let r = self.spans.read(l, &mut buf);
        let old = self.held.replace((l, buf));
        if self.spare.is_none() {
            self.spare = old.map(|(_, b)| b);
        }
        r?;
        self.misses += 1;
        Ok(&mut self.held.as_mut().unwrap().1[..])
    }

    pub fn report(&self, label: &str) {
        let n = self.hits + self.misses;
        let b = self.bytes.load(Ordering::Relaxed);
        let s = self.micros.load(Ordering::Relaxed) as f64 / 1e6;
        println!("trunk [{label}]");
        println!("  pinned {}/{} layers", self.npin, self.n_layers);
        println!(
            "  binds {n}, hits {} ({:.1}%), reads {}",
            self.hits,
            if n > 0 { 100.0 * self.hits as f64 / n as f64 } else { 0.0 },
            self.misses
        );
        println!(
            "  read {:.2} GB in {s:.2} s ({:.0} MB/s), {}",
            b as f64 / 1e9,
            if s > 0.0 { b as f64 / 1e6 / s } else { 0.0 },
            if self.direct { "O_DIRECT" } else { "buffered" }
        );
    }
}

impl Drop for Trunk {
    fn drop(&mut self) {
        self.req.take();
        self.done.take();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

#[cfg(target_os = "linux")]
fn open_direct(path: &Path) -> Option<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new().read(true).custom_flags(0o40000).open(path).ok()
}

#[cfg(not(target_os = "linux"))]
fn open_direct(_path: &Path) -> Option<File> {
    None
}

/// Bytes available on the filesystem holding `path`, walking up to the nearest existing
/// ancestor so this works for a destination directory that has not been created yet.
///
/// `None` when it cannot be determined, which callers must read as "do not block on this":
/// refusing a download because a stat failed would be worse than the problem it prevents.
pub fn free_bytes(path: &Path) -> Option<u64> {
    let mut p = path;
    loop {
        if p.exists() {
            let s = rustix::fs::statvfs(p).ok()?;
            // `bavail` is what a non-root process may actually use, which is the number
            // that decides whether this transfer finishes; `bfree` includes the reserve.
            return Some(s.f_bavail.saturating_mul(s.f_frsize));
        }
        p = p.parent()?;
    }
}
