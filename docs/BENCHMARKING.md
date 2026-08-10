# Benchmarking

## The one rule

**The measured run-to-run spread on an identical configuration is 33%.** Three runs of
the same binary, same prompt, same flags, in the same minute:

| run | s/token |
|---:|---:|
| 1 | 14.78 |
| 2 | 14.67 |
| 3 | 20.14 |

mean 16.53, sd 3.13, spread 33.1%.

If your A/B differs by less than that, you have measured nothing. This is not a caution
in the abstract: on this project, replication was run *last*, and it retracted results
that had already been written down and reasoned about.

## Procedure

```bash
# 3 runs per arm, minimum.
for r in 1 2 3; do
  ./bin/k3 $MODEL --trunk $TRUNK --preset server --ids 1008,10484,318,15383,387 \
           --gen 8 --incremental --out /tmp/arm_a_$r.json | grep 's/token average'
done
```

Then report every run, not the best one:

| arm | run 1 | run 2 | run 3 | mean |
|---|---|---|---|---|

## Things that will contaminate a measurement

Each of these was observed on this project.

- **Background updates.** `unattended-upgrades` was consuming ~63% of a core during one
  run. Stop it: `sudo systemctl stop unattended-upgrades apt-daily.timer`.
- **Two engines at once.** A kill that takes the wrapper but not the child leaves both
  competing for disk and CPU. Check `pgrep -x k3` before starting.
- **Cold vs warm page cache.** A repeat run reads a warm cache. Either drop caches
  between runs (`echo 3 | sudo tee /proc/sys/vm/drop_caches`) or state that you did not.
- **Cold start.** Token 0 pays the full trunk-pinning cost. An 8-token run is dominated
  by it; a 64-token run is not. Never compare runs of different lengths.
- **Different storage.** This workload moves ~135 GB per token, so device bandwidth
  dominates. Comparing two runs from different disks compares the disks.

## What to measure instead of wall clock

Timing is the noisiest thing the engine reports. These are counts, and they do not move:

- `GB read/token`, bytes actually pulled from disk
- `requests` / `evictions`, cache retention, derived as `1 - evictions/requests`
- `pinned N/93 layers`, how much trunk is resident
- generated token ids, for correctness, exact

If a change is supposed to reduce I/O, **measure the bytes**. It is a far stronger claim
than a second-count and it needs no replication.

## Reporting

`docs/PERFORMANCE.md` is the template: state the machine, state the noise floor, mark
single-sample rows as single-sample, and separate what survives the floor from what does
not.
