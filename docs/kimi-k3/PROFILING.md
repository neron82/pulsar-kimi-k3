# Kimi K3 Profiling Baseline

This is the canonical, measurement-only K3 baseline. It does not change
model math, cache policy, placement, queueing, or synchronization in the
normal profile mode.

## Canonical Command

CUDA's process-local ordering currently reports the RTX 3090 as CUDA 0. Select
it explicitly with `PULSAR_GPU=0` and confirm the header line contains both
`RTX 3090` and its `GPU-...` UUID. Do not infer the index from `nvidia-smi`
ordering; the startup identity is authoritative.

```bash
PULSAR_GPU=0 PULSAR_PROFILE=1 \
  ./target/release/pulsar-cli \
  -m /home/neron/models/kimi-k3/Q2_K/Kimi-K3-Q2_K-00001-of-00024.gguf \
  --tokens 1 -n 2 --ctx 16 --temp 0 --top-p 1 --min-p 0 --seed 1 \
  2>&1 | tee /tmp/pulsar-k3-profile.log
```

`--tokens 1` is the fixed one-token prompt. `-n 2` produces two greedy
decode tokens when the model does not emit EOG. The first forward is labelled
`prefill` and later forwards are labelled `decode`; K3 currently executes
both through its sequential single-token path.

For per-layer output and grouped CUDA-event measurements, add
`PULSAR_PROFILE_DETAIL=layers`:

```bash
PULSAR_GPU=0 PULSAR_PROFILE=1 PULSAR_PROFILE_DETAIL=layers \
  ./target/release/pulsar-cli -m /home/neron/models/kimi-k3/Q2_K/Kimi-K3-Q2_K-00001-of-00024.gguf \
  --tokens 1 -n 2 --ctx 16 --temp 0 --top-p 1 --min-p 0 --seed 1 \
  2>&1 | tee /tmp/pulsar-k3-profile-detail.log
```

## Cold And Warm

* **Fresh-process run:** stop the previous CLI, remove only the Pulsar warm
  census/cache artifacts if testing an empty Pulsar runtime cache, then run the
  command above. This is a cold *Pulsar process/cache* run, not an OS page-cache
  cold run.
* **Warm run 1 and 2:** run the exact command twice more without deleting the
  warm artifacts and without changing environment variables. These measure the
  intended retained host/device cache state across fresh processes where the
  existing warm mechanism restores it. A process restart alone does not prove
  that Linux page cache or CUDA memory is cold.
* A truly cold NVMe/page-cache run requires privileged cache dropping and is
  intentionally not part of the normal baseline. If used, record the exact
  `sync` and `/proc/sys/vm/drop_caches` procedure separately.

The model path, cache paths, and the resolved device UUID must be recorded with
each run. Do not launch unrelated model servers during a baseline.

## Output And Definitions

Default `PULSAR_PROFILE=1` prints a compact aggregate and one K3 token line.
Detailed mode prints every layer. CPU `Instant` timings measure submission to
completion at the existing blocking boundaries. The KDA/MLA CUDA-event value
is device execution time for the grouped attention section; stopping that
event synchronizes and therefore detailed mode has measurement perturbation.

The following are exact counters at the instrumented boundaries: token/layer
wall time, selected token IDs, router D2H bytes, requested expert IDs, cache
hit/miss deltas, storage read request count/bytes/max size, and H2D/D2H byte
counts. Storage wait is time spent waiting for io_uring completions during the
blocking `fetch_each` call; completion-callback work is not included. It is not
an NVMe hardware latency sample. H2D time is exact for the measured K3 staging
writes. Resident-tier accesses are counted as device-resident but are not
storage reads.

`PULSAR_STREAM_QD` overrides the default 32-read io_uring depth for controlled
queue-depth experiments. The queue depth changes submission behavior only; it
does not change the model or cache policy.

Wall accounting is deliberately separate from activity accounting. Per-token
`layers total + output/lm-head + unclassified` is a non-overlapping wall view;
`unclassified` is clamped at zero to avoid timer noise. Per-layer GPU-event,
CPU, storage, and transfer values can overlap, so their sum is not expected to
equal layer wall time. The report does not pretend overlapped activity is
additive.

The existing generic profiler fields remain printed for non-K3 paths. K3
profiling-disabled execution does not allocate CUDA timers, read profile
counters, or call additional synchronizations; it retains only small monotonic
stage timestamps around the existing path. Detailed mode is intentionally not suitable for
an overhead comparison against the default mode.

## Overhead Measurement

Run the same command three times, changing only the profile variables:

```bash
# disabled
PULSAR_GPU=0 ./target/release/pulsar-cli -m MODEL.gguf --tokens 1 -n 2 --ctx 16 --temp 0 --top-p 1 --min-p 0 --seed 1 2>&1 | tee /tmp/k3-off.log
# summary
PULSAR_GPU=0 PULSAR_PROFILE=1 ./target/release/pulsar-cli -m MODEL.gguf --tokens 1 -n 2 --ctx 16 --temp 0 --top-p 1 --min-p 0 --seed 1 2>&1 | tee /tmp/k3-summary.log
# detailed, event-timed
PULSAR_GPU=0 PULSAR_PROFILE=1 PULSAR_PROFILE_DETAIL=layers ./target/release/pulsar-cli -m MODEL.gguf --tokens 1 -n 2 --ctx 16 --temp 0 --top-p 1 --min-p 0 --seed 1 2>&1 | tee /tmp/k3-detail.log
```

Report median token wall time from repeated runs. No real-model result is
claimed by this document until the logs contain `loaded`, the RTX 3090 UUID
header, token IDs, and the profile summary. The example output in this file is
intentionally command-only rather than fabricated measurements.
