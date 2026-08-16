//! RustLMHub training / eval workbench — a terminal UI over the fine-tuning and eval tools.
//!
//! Navigate with the arrow keys, Enter to select, Esc to go back, q to quit. Pick a model,
//! pick a dataset (its format is auto-detected and shown), choose Train or Eval, adjust the
//! options, and Launch — the run streams its output live in the same window. No flags.
//!
//! It shells out to the `train_run` / `eval_run` binaries that sit next to it in target/, so
//! the UI is a thin, honest driver over exactly the CLI paths, not a reimplementation.

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;
use std::io::BufRead;
use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;

#[derive(Clone, Copy, PartialEq)]
enum Screen {
    Menu,
    Model,
    Dataset,
    Action,
    TrainCfg,
    EvalCfg,
    Run,
}

struct App {
    screen: Screen,
    quit: bool,
    menu_i: usize,
    // discovery
    models: Vec<PathBuf>,
    model_i: usize,
    datasets: Vec<PathBuf>,
    ds_i: usize,
    ds_fmt: String,
    action_i: usize, // 0 train, 1 eval
    // train config: rank, lr(x1e-4), epochs, accum, max_seq, fcache
    rank: usize,
    lr4: usize, // lr in units of 1e-4
    epochs: usize,
    accum: usize,
    max_seq: usize,
    fcache: bool,
    cfg_i: usize,
    // eval config
    adapter: String,
    score_i: usize, // 0 contains 1 exact 2 refusal
    gen: usize,
    ecfg_i: usize,
    // run
    run_rx: Option<Receiver<String>>,
    run_out: Vec<String>,
    run_done: bool,
}

impl App {
    fn new() -> App {
        App {
            screen: Screen::Menu,
            quit: false,
            menu_i: 0,
            models: scan_models(),
            model_i: 0,
            datasets: scan_datasets(),
            ds_i: 0,
            ds_fmt: String::new(),
            action_i: 0,
            rank: 16,
            lr4: 1,
            epochs: 3,
            accum: 8,
            max_seq: 512,
            fcache: true,
            cfg_i: 0,
            adapter: String::new(),
            score_i: 0,
            gen: 64,
            ecfg_i: 0,
            run_rx: None,
            run_out: Vec::new(),
            run_done: false,
        }
    }

    fn detect_fmt(&mut self) {
        self.ds_fmt = self
            .datasets
            .get(self.ds_i)
            .and_then(|p| std::fs::File::open(p).ok())
            .and_then(|f| std::io::BufReader::new(f).lines().find_map(|l| l.ok().filter(|s| !s.trim().is_empty())))
            .and_then(|line| serde_json::from_str::<serde_json::Value>(&line).ok())
            .and_then(|v| k3::dataset::detect(&v))
            .map(|f| f.label().to_string())
            .unwrap_or_else(|| "unrecognised".into());
    }
}

// --- discovery: models are dirs containing a .gguf; datasets are .jsonl files ---
fn roots() -> Vec<PathBuf> {
    let mut r = vec![PathBuf::from("/mnt/rustlm"), PathBuf::from(".")];
    if let Ok(h) = std::env::var("HOME") {
        r.push(PathBuf::from(h));
    }
    r
}

fn scan_models() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for root in roots() {
        if let Ok(rd) = std::fs::read_dir(&root) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    if let Ok(inner) = std::fs::read_dir(&p) {
                        if inner.flatten().any(|f| f.path().extension().is_some_and(|x| x == "gguf")) {
                            out.push(p);
                        }
                    }
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn scan_datasets() -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn walk(dir: &PathBuf, depth: usize, out: &mut Vec<PathBuf>) {
        if depth > 2 {
            return;
        }
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_file() && p.extension().is_some_and(|x| x == "jsonl") {
                    out.push(p);
                } else if p.is_dir() && !p.ends_with(".cache") {
                    walk(&p, depth + 1, out);
                }
            }
        }
    }
    for root in roots() {
        walk(&root, 0, &mut out);
    }
    out.sort();
    out.dedup();
    out
}

fn sibling_bin(name: &str) -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(name)))
        .unwrap_or_else(|| PathBuf::from(name))
}

fn spawn_run(app: &mut App) {
    let model = match app.models.get(app.model_i) {
        Some(m) => m.clone(),
        None => return,
    };
    let data = match app.datasets.get(app.ds_i) {
        Some(d) => d.clone(),
        None => return,
    };
    let mut cmd = if app.action_i == 0 {
        let mut c = Command::new(sibling_bin("train_run"));
        c.arg(&model).arg(&data)
            .args(["--rank", &app.rank.to_string()])
            .args(["--lr", &format!("{}e-4", app.lr4)])
            .args(["--epochs", &app.epochs.to_string()])
            .args(["--accum", &app.accum.to_string()])
            .args(["--max-seq", &app.max_seq.to_string()])
            .args(["--out", "/mnt/rustlm/adapter.loaa"]);
        if !app.fcache {
            c.args(["--fcache-gb", "0"]);
        }
        c
    } else {
        let score = ["contains", "exact", "refusal"][app.score_i];
        let mut c = Command::new(sibling_bin("eval_run"));
        c.arg(&model).arg(&data).args(["--gen", &app.gen.to_string()]).args(["--score", score]);
        if !app.adapter.trim().is_empty() {
            c.args(["--adapter", app.adapter.trim()]);
        }
        c
    };
    cmd.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());
    match cmd.spawn() {
        Ok(mut child) => {
            let (tx, rx) = channel();
            let out = child.stdout.take();
            let err = child.stderr.take();
            std::thread::spawn(move || {
                if let Some(o) = out {
                    for l in std::io::BufReader::new(o).lines().map_while(Result::ok) {
                        let _ = tx.send(l);
                    }
                }
                if let Some(e) = err {
                    for l in std::io::BufReader::new(e).lines().map_while(Result::ok) {
                        let _ = tx.send(format!("! {l}"));
                    }
                }
                let _ = tx.send("\u{2713} process exited".into());
            });
            app.run_rx = Some(rx);
            app.run_out.clear();
            app.run_done = false;
            app.screen = Screen::Run;
        }
        Err(e) => {
            app.run_out = vec![format!("failed to launch: {e}")];
            app.screen = Screen::Run;
            app.run_done = true;
        }
    }
}

fn main() -> std::io::Result<()> {
    use std::io::IsTerminal;
    if !std::io::stdout().is_terminal() {
        eprintln!("rustlm train: this is an interactive workbench — run it in a terminal, not a pipe.");
        return Ok(());
    }
    let mut term = ratatui::init();
    let mut app = App::new();
    while !app.quit {
        term.draw(|f| ui(f, &app))?;
        // pump run output
        if let Some(rx) = &app.run_rx {
            while let Ok(line) = rx.try_recv() {
                if line.contains("process exited") {
                    app.run_done = true;
                }
                app.run_out.push(line);
                if app.run_out.len() > 500 {
                    app.run_out.drain(0..100);
                }
            }
        }
        if event::poll(Duration::from_millis(120))? {
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press {
                    handle_key(&mut app, k.code);
                }
            }
        }
    }
    ratatui::restore();
    Ok(())
}

fn handle_key(app: &mut App, code: KeyCode) {
    let up = |i: &mut usize| { if *i > 0 { *i -= 1; } };
    let dn = |i: &mut usize, n: usize| { if *i + 1 < n { *i += 1; } };
    match app.screen {
        Screen::Menu => match code {
            KeyCode::Up => up(&mut app.menu_i),
            KeyCode::Down => dn(&mut app.menu_i, 2),
            KeyCode::Char('q') => app.quit = true,
            KeyCode::Enter => {
                if app.menu_i == 1 { app.quit = true; } else { app.screen = Screen::Model; }
            }
            _ => {}
        },
        Screen::Model => match code {
            KeyCode::Up => up(&mut app.model_i),
            KeyCode::Down => dn(&mut app.model_i, app.models.len()),
            KeyCode::Esc => app.screen = Screen::Menu,
            KeyCode::Char('q') => app.quit = true,
            KeyCode::Enter if !app.models.is_empty() => { app.screen = Screen::Dataset; app.detect_fmt(); }
            _ => {}
        },
        Screen::Dataset => match code {
            KeyCode::Up => { up(&mut app.ds_i); app.detect_fmt(); }
            KeyCode::Down => { dn(&mut app.ds_i, app.datasets.len()); app.detect_fmt(); }
            KeyCode::Esc => app.screen = Screen::Model,
            KeyCode::Char('q') => app.quit = true,
            KeyCode::Enter if !app.datasets.is_empty() => app.screen = Screen::Action,
            _ => {}
        },
        Screen::Action => match code {
            KeyCode::Up => up(&mut app.action_i),
            KeyCode::Down => dn(&mut app.action_i, 2),
            KeyCode::Esc => app.screen = Screen::Dataset,
            KeyCode::Char('q') => app.quit = true,
            KeyCode::Enter => app.screen = if app.action_i == 0 { Screen::TrainCfg } else { Screen::EvalCfg },
            _ => {}
        },
        Screen::TrainCfg => {
            let n = 7; // 6 fields + Launch
            match code {
                KeyCode::Up => up(&mut app.cfg_i),
                KeyCode::Down => dn(&mut app.cfg_i, n),
                KeyCode::Esc => app.screen = Screen::Action,
                KeyCode::Char('q') => app.quit = true,
                KeyCode::Left | KeyCode::Right => {
                    let d: i64 = if code == KeyCode::Right { 1 } else { -1 };
                    match app.cfg_i {
                        0 => app.rank = (app.rank as i64 + d * 8).clamp(4, 128) as usize,
                        1 => app.lr4 = (app.lr4 as i64 + d).clamp(1, 50) as usize,
                        2 => app.epochs = (app.epochs as i64 + d).clamp(1, 50) as usize,
                        3 => app.accum = (app.accum as i64 + d).clamp(1, 64) as usize,
                        4 => app.max_seq = (app.max_seq as i64 + d * 64).clamp(64, 2048) as usize,
                        5 => app.fcache = !app.fcache,
                        _ => {}
                    }
                }
                KeyCode::Enter if app.cfg_i == n - 1 => spawn_run(app),
                _ => {}
            }
        }
        Screen::EvalCfg => {
            let n = 4; // adapter, score, gen, Launch
            match code {
                KeyCode::Up => up(&mut app.ecfg_i),
                KeyCode::Down => dn(&mut app.ecfg_i, n),
                KeyCode::Esc => app.screen = Screen::Action,
                KeyCode::Left | KeyCode::Right => {
                    let d: i64 = if code == KeyCode::Right { 1 } else { -1 };
                    match app.ecfg_i {
                        1 => app.score_i = ((app.score_i as i64 + d).rem_euclid(3)) as usize,
                        2 => app.gen = (app.gen as i64 + d * 16).clamp(16, 512) as usize,
                        _ => {}
                    }
                }
                KeyCode::Char(c) if app.ecfg_i == 0 => app.adapter.push(c),
                KeyCode::Backspace if app.ecfg_i == 0 => { app.adapter.pop(); }
                KeyCode::Enter if app.ecfg_i == n - 1 => spawn_run(app),
                _ => {}
            }
        }
        Screen::Run => match code {
            KeyCode::Esc | KeyCode::Char('q') => { app.run_rx = None; app.screen = Screen::Menu; }
            _ => {}
        },
    }
}

fn list<'a>(items: Vec<ListItem<'a>>, sel: usize, title: &'a str) -> (List<'a>, ListState) {
    let mut st = ListState::default();
    st.select(Some(sel));
    let l = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_style(Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD))
        .highlight_symbol("> ");
    (l, st)
}

fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(5), Constraint::Length(2)])
        .split(f.area());

    let title = Paragraph::new(Line::from(vec![
        Span::styled(" RustLMHub ", Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw("  fine-tune & eval workbench"),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    match app.screen {
        Screen::Menu => {
            let items = vec![ListItem::new("Train / Eval a model"), ListItem::new("Quit")];
            let (l, mut s) = list(items, app.menu_i, "Main menu");
            f.render_stateful_widget(l, chunks[1], &mut s);
        }
        Screen::Model => {
            let items: Vec<_> = if app.models.is_empty() {
                vec![ListItem::new("(no models found under /mnt/rustlm, ., ~ — need a dir with a .gguf)")]
            } else {
                app.models.iter().map(|m| ListItem::new(m.display().to_string())).collect()
            };
            let (l, mut s) = list(items, app.model_i, "Select model (dir with a .gguf)");
            f.render_stateful_widget(l, chunks[1], &mut s);
        }
        Screen::Dataset => {
            let items: Vec<_> = if app.datasets.is_empty() {
                vec![ListItem::new("(no .jsonl datasets found)")]
            } else {
                app.datasets.iter().enumerate().map(|(i, d)| {
                    let tag = if i == app.ds_i { format!("   [{}]", app.ds_fmt) } else { String::new() };
                    ListItem::new(format!("{}{}", d.display(), tag))
                }).collect()
            };
            let (l, mut s) = list(items, app.ds_i, "Select dataset (format auto-detected)");
            f.render_stateful_widget(l, chunks[1], &mut s);
        }
        Screen::Action => {
            let items = vec![ListItem::new("Train  (LoRA fine-tune, feature-cached)"), ListItem::new("Eval   (generate & score)")];
            let (l, mut s) = list(items, app.action_i, "Action");
            f.render_stateful_widget(l, chunks[1], &mut s);
        }
        Screen::TrainCfg => {
            let rows = [
                format!("rank            {}", app.rank),
                format!("learning rate   {}e-4", app.lr4),
                format!("epochs          {}", app.epochs),
                format!("grad accum      {}", app.accum),
                format!("max seq tokens  {}", app.max_seq),
                format!("feature cache   {}", if app.fcache { "on (~15x multi-epoch)" } else { "off" }),
                "  >> LAUNCH TRAINING <<".to_string(),
            ];
            let items: Vec<_> = rows.iter().map(|r| ListItem::new(r.clone())).collect();
            let (l, mut s) = list(items, app.cfg_i, "Train config  (<-/-> change, Enter on LAUNCH)");
            f.render_stateful_widget(l, chunks[1], &mut s);
        }
        Screen::EvalCfg => {
            let score = ["contains", "exact", "refusal"][app.score_i];
            let rows = [
                format!("adapter path    {}", if app.adapter.is_empty() { "(base model — type a path to eval a fine-tune)".into() } else { app.adapter.clone() }),
                format!("scoring         {score}"),
                format!("gen tokens      {}", app.gen),
                "  >> LAUNCH EVAL <<".to_string(),
            ];
            let items: Vec<_> = rows.iter().map(|r| ListItem::new(r.clone())).collect();
            let (l, mut s) = list(items, app.ecfg_i, "Eval config  (type path on line 1; <-/-> change)");
            f.render_stateful_widget(l, chunks[1], &mut s);
        }
        Screen::Run => {
            let tail: Vec<Line> = app.run_out.iter().rev().take(chunks[1].height as usize).rev()
                .map(|l| {
                    let style = if l.starts_with('!') { Style::default().fg(Color::Red) } else { Style::default() };
                    Line::styled(l.clone(), style)
                }).collect();
            let title = if app.run_done { "Run — finished (Esc to menu)" } else { "Run — live (Esc to stop watching)" };
            let p = Paragraph::new(tail).block(Block::default().borders(Borders::ALL).title(title)).wrap(Wrap { trim: false });
            f.render_widget(p, chunks[1]);
        }
    }

    let help = match app.screen {
        Screen::Run => " Esc back to menu   q quit ",
        Screen::TrainCfg | Screen::EvalCfg => " Up/Down field   Left/Right change   Enter=LAUNCH   Esc back ",
        _ => " Up/Down move   Enter select   Esc back   q quit ",
    };
    let sel = format!(" model: {}   dataset: {}   ",
        app.models.get(app.model_i).map(|m| m.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()).unwrap_or_else(|| "-".into()),
        app.datasets.get(app.ds_i).map(|d| d.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()).unwrap_or_else(|| "-".into()));
    let footer = Paragraph::new(Line::from(vec![
        Span::styled(help, Style::default().fg(Color::Yellow)),
        Span::styled(sel, Style::default().fg(Color::DarkGray)),
    ]));
    f.render_widget(footer, chunks[2]);
}
