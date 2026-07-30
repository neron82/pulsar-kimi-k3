//! Expert streaming core: bounded io_uring/O_DIRECT reads over single or split GGUF files.
//!
//! The stream crate stays CUDA-free. Callers may inject CUDA-pinned staging
//! buffers through `uring::BufAlloc`; `fetch::Fetcher::fetch_each` hands each
//! completed slab to the caller while later reads remain in flight.
//!
//! ## Bounded pipeline
//!
//! The [`pipeline`] module provides a production-usable bounded streaming
//! primitive for model tensor loads: multiple in-flight read requests,
//! bounded memory via a pre-allocated buffer pool, optional pinned staging,
//! completion ordering, cancellation/cleanup, and per-stage timing counters.

use gguf::Gguf;

/// One disk read at an absolute virtual-file offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Read {
    pub offset: u64,
    pub len: u64,
}

/// Build the per-expert slab read universe for a streamed MoE GGUF.
pub fn expert_reads(g: &Gguf, model_len: u64) -> Result<Vec<Read>, String> {
    let n_expert = g
        .arch_meta("expert_count")
        .and_then(gguf::Value::as_u64)
        .ok_or("missing expert_count")?;
    let mut out = Vec::new();
    for t in &g.tensors {
        if !t.name.ends_with("_exps.weight") {
            continue;
        }
        if t.dims.len() != 3 || t.dims[2] != n_expert {
            return Err(format!("{}: unexpected exps dims {:?}", t.name, t.dims));
        }
        let row = t
            .ty
            .row_bytes(t.dims[0])
            .ok_or_else(|| format!("{}: unmodeled type {:?}", t.name, t.ty))?;
        let expert_bytes = row * t.dims[1];
        let base = g.data_offset + t.offset;
        for e in 0..n_expert {
            let offset = base + e * expert_bytes;
            if offset + expert_bytes > model_len {
                return Err(format!("{}: expert {} beyond eof", t.name, e));
            }
            out.push(Read { offset, len: expert_bytes });
        }
    }
    if out.is_empty() {
        return Err("no *_exps.weight tensors found".into());
    }
    Ok(out)
}

/// Plan file format shared with the reference I/O benchmark.
pub fn plan_to_string(reads: &[Read]) -> String {
    let mut s = String::with_capacity(reads.len() * 24);
    for r in reads {
        s.push_str(&format!("{} {}\n", r.offset, r.len));
    }
    s
}

pub fn plan_from_str(s: &str) -> Result<Vec<Read>, String> {
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace();
            let offset = it
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| format!("bad plan line: {l}"))?;
            let len = it
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| format!("bad plan line: {l}"))?;
            Ok(Read { offset, len })
        })
        .collect()
}

#[cfg(target_os = "linux")]
pub mod uring {
    use super::Read;
    use io_uring::{opcode, types, IoUring};
    use std::os::fd::AsRawFd;

    /// CUDA-free injection point for page-aligned/pinned staging memory.
    #[derive(Clone, Copy)]
    pub struct BufAlloc {
        pub alloc: fn(usize) -> *mut u8,
        pub free: fn(*mut u8, usize),
    }

    pub struct Aligned {
        ptr: *mut u8,
        cap: usize,
        custom_free: Option<fn(*mut u8, usize)>,
    }

    unsafe impl Send for Aligned {}

    impl Aligned {
        pub fn ptr(&self) -> *mut u8 {
            self.ptr
        }

        pub fn cap(&self) -> usize {
            self.cap
        }

        pub fn new(cap: usize, align: usize) -> Option<Self> {
            let layout = std::alloc::Layout::from_size_align(cap, align).ok()?;
            let ptr = unsafe { std::alloc::alloc(layout) };
            (!ptr.is_null()).then_some(Self {
                ptr,
                cap,
                custom_free: None,
            })
        }

        pub fn new_with(cap: usize, align: usize, a: Option<BufAlloc>) -> Option<Self> {
            if let Some(a) = a {
                let ptr = (a.alloc)(cap);
                if !ptr.is_null() {
                    return Some(Self {
                        ptr,
                        cap,
                        custom_free: Some(a.free),
                    });
                }
            }
            Self::new(cap, align)
        }
    }

    impl Drop for Aligned {
        fn drop(&mut self) {
            match self.custom_free {
                Some(free) => free(self.ptr, self.cap),
                None => {
                    let layout = std::alloc::Layout::from_size_align(self.cap, 4096).unwrap();
                    unsafe { std::alloc::dealloc(self.ptr, layout) };
                }
            }
        }
    }

    #[derive(Debug, Clone, Copy, Default)]
    pub struct Stats {
        pub bytes_payload: u64,
        pub bytes_disk: u64,
        pub reads: u64,
        pub secs: f64,
        pub checksum: u8,
    }

    struct InFlight {
        buf: Aligned,
        payload_off: usize,
        payload_len: usize,
    }

    /// Run a read plan with `qd` aligned reads in flight.
    pub fn run_plan(
        file: &std::fs::File,
        reads: &[Read],
        qd: usize,
        align: u64,
    ) -> std::io::Result<Stats> {
        if qd == 0 || !align.is_power_of_two() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "qd must be nonzero and align must be a power of two",
            ));
        }
        let mut ring = IoUring::new(qd as u32 * 2)?;
        let fd = types::Fd(file.as_raw_fd());
        let t0 = std::time::Instant::now();
        let mut stats = Stats::default();
        let mut slots: Vec<Option<InFlight>> = (0..qd).map(|_| None).collect();
        let mut next = 0usize;
        let mut inflight = 0usize;

        loop {
            while inflight < qd && next < reads.len() {
                let r = reads[next];
                let aligned_off = r.offset & !(align - 1);
                let payload_off = (r.offset - aligned_off) as usize;
                let disk_len = (payload_off as u64 + r.len).next_multiple_of(align);
                let slot = slots
                    .iter()
                    .position(Option::is_none)
                    .expect("free slot when inflight < qd");
                let buf = Aligned::new(disk_len as usize + 256, align as usize)
                    .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::OutOfMemory))?;
                let sqe = opcode::Read::new(fd, buf.ptr(), disk_len as u32)
                    .offset(aligned_off)
                    .build()
                    .user_data(slot as u64);
                slots[slot] = Some(InFlight {
                    buf,
                    payload_off,
                    payload_len: r.len as usize,
                });
                unsafe { ring.submission().push(&sqe).expect("submission queue full") };
                stats.bytes_disk += disk_len;
                inflight += 1;
                next += 1;
            }
            if inflight == 0 {
                break;
            }
            ring.submit_and_wait(1)?;
            let completions: Vec<(u64, i32)> =
                ring.completion().map(|c| (c.user_data(), c.result())).collect();
            for (ud, res) in completions {
                if res < 0 {
                    return Err(std::io::Error::from_raw_os_error(-res));
                }
                let slot = ud as usize;
                let inf = slots[slot].take().expect("slot occupied");
                inflight -= 1;
                stats.reads += 1;
                stats.bytes_payload += inf.payload_len as u64;
                stats.checksum ^= unsafe {
                    *inf.buf.ptr.add(inf.payload_off + inf.payload_len / 2)
                };
            }
        }
        stats.secs = t0.elapsed().as_secs_f64();
        Ok(stats)
    }
}

#[cfg(target_os = "linux")]
pub mod fetch {
    use super::uring::Aligned;
    use super::Read;
    use io_uring::{opcode, types, IoUring};
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::OpenOptionsExt;

    const ALIGN: u64 = 4096;

    /// One fetched aligned bracket with a payload window into it.
    pub struct Slab {
        pub(crate) buf: Aligned,
        pub(crate) payload_off: usize,
        pub(crate) payload_len: usize,
    }

    impl Slab {
        pub fn payload(&self) -> &[u8] {
            unsafe {
                std::slice::from_raw_parts(self.buf.ptr().add(self.payload_off), self.payload_len)
            }
        }

        pub fn bytes(&self) -> usize {
            self.buf.cap()
        }
    }

    pub struct Fetcher {
        ring: IoUring,
        files: Vec<(u64, std::fs::File)>,
        qd: usize,
        buf_alloc: Option<super::uring::BufAlloc>,
    }

    impl Fetcher {
        pub fn open(path: &std::path::Path, qd: usize) -> std::io::Result<Fetcher> {
            Self::open_with(path, qd, None)
        }

        pub fn open_with(
            path: &std::path::Path,
            qd: usize,
            buf_alloc: Option<super::uring::BufAlloc>,
        ) -> std::io::Result<Fetcher> {
            Self::open_split(std::slice::from_ref(&(0, path.to_path_buf())), qd, buf_alloc)
        }

        pub fn open_split(
            shards: &[(u64, std::path::PathBuf)],
            qd: usize,
            buf_alloc: Option<super::uring::BufAlloc>,
        ) -> std::io::Result<Fetcher> {
            if qd == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "queue depth must be nonzero",
                ));
            }
            let mut files = Vec::with_capacity(shards.len());
            for (base, path) in shards {
                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(path)?;
                files.push((*base, file));
            }
            Ok(Fetcher {
                ring: IoUring::new(qd as u32 * 2)?,
                files,
                qd,
                buf_alloc,
            })
        }

        fn route(&self, offset: u64) -> (types::Fd, u64) {
            let i = match self.files.binary_search_by(|(b, _)| b.cmp(&offset)) {
                Ok(i) => i,
                Err(i) => i.saturating_sub(1),
            };
            (types::Fd(self.files[i].1.as_raw_fd()), offset - self.files[i].0)
        }

        pub fn fetch(&mut self, reads: &[Read]) -> std::io::Result<Vec<Slab>> {
            let mut out: Vec<Option<Slab>> = (0..reads.len()).map(|_| None).collect();
            self.fetch_each(reads, |i, slab| {
                out[i] = Some(slab);
                Ok(())
            })?;
            Ok(out.into_iter().map(|s| s.expect("all fetched")).collect())
        }

        /// Submit up to `qd` aligned O_DIRECT reads and hand each completion to
        /// the callback. The callback runs while later reads remain in flight,
        /// providing the SSD→processing overlap used by expert streaming.
        pub fn fetch_each(
            &mut self,
            reads: &[Read],
            mut on_slab: impl FnMut(usize, Slab) -> std::io::Result<()>,
        ) -> std::io::Result<()> {
            let mut pending: Vec<Option<Slab>> = (0..reads.len()).map(|_| None).collect();
            let mut next = 0usize;
            let mut inflight = 0usize;

            loop {
                while inflight < self.qd && next < reads.len() {
                    let r = reads[next];
                    let (fd, local) = self.route(r.offset);
                    let aligned_off = local & !(ALIGN - 1);
                    let payload_off = (local - aligned_off) as usize;
                    let disk_len = (payload_off as u64 + r.len).next_multiple_of(ALIGN);
                    let buf = Aligned::new_with(
                        disk_len as usize + 256,
                        ALIGN as usize,
                        self.buf_alloc,
                    )
                    .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::OutOfMemory))?;
                    let sqe = opcode::Read::new(fd, buf.ptr(), disk_len as u32)
                        .offset(aligned_off)
                        .build()
                        .user_data(next as u64);
                    pending[next] = Some(Slab {
                        buf,
                        payload_off,
                        payload_len: r.len as usize,
                    });
                    unsafe { self.ring.submission().push(&sqe).expect("submission queue full") };
                    inflight += 1;
                    next += 1;
                }
                if inflight == 0 {
                    break;
                }
                self.ring.submit_and_wait(1)?;
                let completions: Vec<(u64, i32)> = self
                    .ring
                    .completion()
                    .map(|c| (c.user_data(), c.result()))
                    .collect();
                for (ud, res) in completions {
                    if res < 0 {
                        return Err(std::io::Error::from_raw_os_error(-res));
                    }
                    inflight -= 1;
                    let idx = ud as usize;
                    let slab = pending[idx].take().expect("slot occupied");
                    on_slab(idx, slab)?;
                }
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Bounded pipeline (Linux only)
// ---------------------------------------------------------------------------

/// Bounded SSD-to-staging streaming pipeline with overlap and telemetry.
///
/// Wraps a [`Fetcher`](fetch::Fetcher) with a pre-allocated buffer pool so
/// memory is bounded regardless of how many reads are submitted. Multiple
/// reads can be in-flight simultaneously; each batch is yielded in submission
/// order and must be drained before submitting the next batch.
///
/// # Backpressure
///
/// [`Pipeline::submit`] blocks (via `submit_and_wait`) when the buffer pool
/// is empty, draining completions until a slot frees up.  This keeps total
/// memory at `max_slots × max_bracket_size`.
///
/// # Cancellation
///
/// Dropping a [`PipelineBatch`] discards its remaining completions.  Dropping
/// the [`Pipeline`] itself cancels all in-flight I/O and frees all buffers.
#[cfg(target_os = "linux")]
pub mod pipeline {
    use super::fetch::Slab;
    use super::uring::Aligned;
    use super::Read;
    use std::collections::VecDeque;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::OpenOptionsExt;
    use std::path::Path;
    use std::time::Instant;

    use io_uring::{opcode, types, IoUring};

    const ALIGN: u64 = 4096;

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    /// Configuration for opening a [`Pipeline`].
    pub struct PipelineConfig {
        /// io_uring queue depth (max concurrent in-flight reads).
        pub qd: usize,
        /// Max number of pre-allocated buffer slots.  Must be ≥ `qd`.
        /// Total memory = `max_slots × max_bracket_size`.
        pub max_slots: usize,
        /// Optional custom allocator (e.g. CUDA pinned memory).
        pub buf_alloc: Option<super::uring::BufAlloc>,
    }

    // -----------------------------------------------------------------------
    // Per-stage timing counters
    // -----------------------------------------------------------------------

    /// Cumulative timing counters for the pipeline lifecycle.
    ///
    /// Each counter is the sum of wall-clock durations for that stage across
    /// all completed reads.  Divide by `reads` to get the average.
    #[derive(Debug, Clone, Default)]
    pub struct PipelineStats {
        /// Total reads completed.
        pub reads: u64,
        /// Total bytes of payload data read.
        pub bytes_payload: u64,
        /// Total bytes transferred from disk (aligned brackets).
        pub bytes_disk: u64,
        /// Wall time spent submitting SQEs (submit stage).
        pub t_submit_ns: u64,
        /// Wall time spent waiting for and collecting completions.
        pub t_complete_ns: u64,
        /// Wall time the caller spent processing yielded slabs (yield stage).
        pub t_yield_ns: u64,
    }

    // -----------------------------------------------------------------------
    // Internal: one slot in the pre-allocated pool
    // -----------------------------------------------------------------------

    struct Slot {
        buf: Aligned,
        /// Offset within the aligned bracket where the payload starts.
        payload_off: usize,
        /// Length of the payload.
        payload_len: usize,
        /// Index of this read within its batch (for ordering).
        batch_pos: usize,
        /// When the SQE for this slot was submitted (for timing).
        t_submit: Instant,
    }

    /// A completed slab whose buffer has been moved out of the pool slot.
    /// The pool slot is freed immediately; the slab waits here for ordered yield.
    struct PendingSlab {
        buf: Aligned,
        payload_off: usize,
        payload_len: usize,
    }

    // -----------------------------------------------------------------------
    // Pipeline
    // -----------------------------------------------------------------------

    /// Bounded streaming pipeline.
    ///
    /// Create one via [`Pipeline::open`] or [`Pipeline::open_split`], then
    /// call [`submit`](Pipeline::submit) to enqueue a batch of reads.
    pub struct Pipeline {
        ring: IoUring,
        /// (virtual base, file), same layout as Fetcher.
        files: Vec<(u64, std::fs::File)>,
        qd: usize,
        buf_alloc: Option<super::uring::BufAlloc>,

        /// Pre-allocated buffer pool.  `pool[i]` is `Some` when the slot is
        /// in use (submitted to the ring), `None` when free.
        pool: Vec<Option<Slot>>,
        /// Free slot indices (LIFO for cache warmth).
        free_slots: Vec<usize>,

        /// Per-batch completion queue.  Each entry is (batch_id, index_in_batch, PendingSlab).
        /// Completions are moved here from the ring immediately; the pool slot
        /// is freed so the buffer can be reused.  The iterator yields them in order.
        pending_completions: VecDeque<(u64, usize, PendingSlab)>,

        /// Monotonically increasing batch id.
        next_batch_id: u64,

        /// Cumulative stats.
        stats: PipelineStats,
    }

    impl Pipeline {
        /// Open a single-file pipeline.
        pub fn open(path: &Path, cfg: PipelineConfig) -> std::io::Result<Self> {
            Self::open_split(std::slice::from_ref(&(0u64, path.to_path_buf())), cfg)
        }

        /// Open a split-shard pipeline.
        ///
        /// `shards[i]` = (virtual base offset, path).  Bases must be ascending.
        pub fn open_split(
            shards: &[(u64, std::path::PathBuf)],
            cfg: PipelineConfig,
        ) -> std::io::Result<Self> {
            assert!(cfg.max_slots >= cfg.qd, "max_slots must be >= qd");

            let mut files = Vec::with_capacity(shards.len());
            for (base, path) in shards {
                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(path)?;
                files.push((*base, file));
            }

            let ring = IoUring::new(cfg.qd as u32 * 2)?;
            let pool: Vec<Option<Slot>> = (0..cfg.max_slots).map(|_| None).collect();
            let free_slots: Vec<usize> = (0..cfg.max_slots).rev().collect();

            Ok(Pipeline {
                ring,
                files,
                qd: cfg.qd,
                buf_alloc: cfg.buf_alloc,
                pool,
                free_slots,
                pending_completions: VecDeque::new(),
                next_batch_id: 0,
                stats: PipelineStats::default(),
            })
        }

        /// Return a snapshot of cumulative timing counters.
        pub fn stats(&self) -> &PipelineStats {
            &self.stats
        }

        /// Reset cumulative stats to zero.
        pub fn reset_stats(&mut self) {
            self.stats = PipelineStats::default();
        }

        /// Number of free buffer slots remaining.
        pub fn free_slots(&self) -> usize {
            self.free_slots.len()
        }

        /// Number of in-flight reads.
        pub fn inflight(&self) -> usize {
            self.pool.len() - self.free_slots.len()
        }

        // -- internal helpers ------------------------------------------------

        fn route(&self, offset: u64) -> (types::Fd, u64) {
            let i = match self.files.binary_search_by(|(b, _)| b.cmp(&offset)) {
                Ok(i) => i,
                Err(i) => i.saturating_sub(1),
            };
            (types::Fd(self.files[i].1.as_raw_fd()), offset - self.files[i].0)
        }

        /// Allocate a buffer from the pool (or return None if empty).
        fn alloc_slot(&mut self, disk_len: usize) -> Option<usize> {
            let idx = self.free_slots.pop()?;
            let buf = Aligned::new_with(disk_len + 256, ALIGN as usize, self.buf_alloc)?;
            self.pool[idx] = Some(Slot {
                buf,
                payload_off: 0,
                payload_len: 0,
                batch_pos: 0,
                t_submit: Instant::now(),
            });
            Some(idx)
        }

        /// Free a slot back to the pool.
        fn free_slot(&mut self, idx: usize) {
            self.pool[idx] = None;
            self.free_slots.push(idx);
        }

        /// Drain completions from the ring, moving buffers out of pool slots
        /// and freeing them immediately.  Returns the number of completions drained.
        fn drain_ring(&mut self) -> std::io::Result<usize> {
            // Collect completions first to avoid borrow conflicts
            let cqes: Vec<(u64, i32)> = self.ring.completion().map(|c| (c.user_data(), c.result())).collect();
            let mut count = 0;
            for (ud, res) in cqes {
                if res < 0 {
                    return Err(std::io::Error::from_raw_os_error(-res));
                }
                // user_data encodes: (batch_id << 32) | slot_idx
                let slot_idx = (ud & 0xFFFF_FFFF) as usize;
                let batch_id = (ud >> 32) as u64;
                let slot = self.pool[slot_idx].take().expect("completion on empty slot");
                let batch_pos = slot.batch_pos;
                self.stats.reads += 1;
                self.stats.bytes_payload += slot.payload_len as u64;
                self.stats.t_complete_ns += slot.t_submit.elapsed().as_nanos() as u64;
                // Move the buffer out of the slot and free the pool slot immediately
                let pending = PendingSlab {
                    buf: slot.buf,
                    payload_off: slot.payload_off,
                    payload_len: slot.payload_len,
                };
                self.free_slot(slot_idx);
                self.pending_completions.push_back((batch_id, batch_pos, pending));
                count += 1;
            }
            Ok(count)
        }

        /// Submit as many reads as possible, up to `qd` in-flight.
        /// Returns the number of reads submitted.
        fn submit_batch_reads(
            &mut self,
            reads: &[Read],
            batch_id: u64,
            start: &mut usize,
            inflight: &mut usize,
        ) -> std::io::Result<usize> {
            let mut submitted = 0;
            while *inflight < self.qd && *start < reads.len() {
                let r = reads[*start];
                let (fd, local) = self.route(r.offset);
                let aligned_off = local & !(ALIGN - 1);
                let payload_off = (local - aligned_off) as usize;
                let disk_len = (payload_off as u64 + r.len).next_multiple_of(ALIGN);

                // Get a buffer from the pool (backpressure if empty)
                let slot_idx = match self.alloc_slot(disk_len as usize) {
                    Some(idx) => idx,
                    None => break, // pool exhausted; will drain completions first
                };

                let slot = self.pool[slot_idx].as_mut().unwrap();
                slot.payload_off = payload_off;
                slot.payload_len = r.len as usize;
                slot.batch_pos = *start;

                // user_data: (batch_id << 32) | slot_idx
                let ud = (batch_id << 32) | slot_idx as u64;
                let sqe = opcode::Read::new(fd, slot.buf.ptr(), disk_len as u32)
                    .offset(aligned_off)
                    .build()
                    .user_data(ud);
                unsafe { self.ring.submission().push(&sqe).expect("sq room") };

                slot.t_submit = Instant::now();
                self.stats.bytes_disk += disk_len;
                *inflight += 1;
                *start += 1;
                submitted += 1;
            }
            Ok(submitted)
        }

        /// Submit a batch of reads and return a handle that yields completions
        /// in submission order.
        ///
        /// This may block (via `submit_and_wait`) if the buffer pool is
        /// exhausted, draining completions until a slot frees up.
        pub fn submit(&mut self, reads: &[Read]) -> std::io::Result<PipelineBatch<'_>> {
            if reads.is_empty() {
                return Ok(PipelineBatch {
                    pipeline: self as *mut Pipeline as *mut u8,
                    batch_id: 0,
                    remaining: 0,
                    pos: 0,
                    _marker: std::marker::PhantomData,
                });
            }

            let batch_id = self.next_batch_id;
            self.next_batch_id += 1;

            let mut start = 0usize;
            let mut inflight = 0usize;

            // Submit first wave
            let t_sub = Instant::now();
            self.submit_batch_reads(reads, batch_id, &mut start, &mut inflight)?;
            self.stats.t_submit_ns += t_sub.elapsed().as_nanos() as u64;

            // If the pool was exhausted, drain completions to free slots
            while start < reads.len() {
                // Submit what we can
                self.submit_batch_reads(reads, batch_id, &mut start, &mut inflight)?;
                if start >= reads.len() {
                    break;
                }
                // Drain completions to free slots
                self.ring.submit_and_wait(1)?;
                let t_sub = Instant::now();
                self.drain_ring()?;
                self.stats.t_submit_ns += t_sub.elapsed().as_nanos() as u64;
                inflight = self.pool.len() - self.free_slots.len();
            }

            Ok(PipelineBatch {
                pipeline: self as *mut Pipeline as *mut u8,
                batch_id,
                remaining: reads.len(),
                pos: 0,
                _marker: std::marker::PhantomData,
            })
        }

        /// Drain completions from the ring, blocking until at least one
        /// completion is available.  Returns the number of completions drained.
        fn drain_one(&mut self) -> std::io::Result<usize> {
            self.ring.submit_and_wait(1)?;
            self.drain_ring()
        }

        /// Try to yield the next completion for `batch_id` at position `pos`.
        /// Returns `None` if the completion is not yet available (caller should
        /// call `drain_one` and retry).
        fn try_yield(&mut self, batch_id: u64, pos: usize) -> Option<Slab> {
            let idx = self.pending_completions.iter().position(|&(bid, p, _)| {
                bid == batch_id && p == pos
            })?;

            let (_, _, pending) = self.pending_completions.remove(idx).unwrap();
            Some(Slab {
                buf: pending.buf,
                payload_off: pending.payload_off,
                payload_len: pending.payload_len,
            })
        }
    }

    // -----------------------------------------------------------------------
    // PipelineBatch
    // -----------------------------------------------------------------------

    /// A handle to a submitted batch of reads.  Implements `Iterator` that
    /// yields [`Slab`]s in submission order, blocking as needed.
    ///
    /// Dropping the batch before exhausting it discards remaining completions.
    pub struct PipelineBatch<'p> {
        // Raw pointer to avoid borrow-checker gymnastics with the iterator impl.
        pipeline: *mut u8,
        batch_id: u64,
        remaining: usize,
        pos: usize,
        _marker: std::marker::PhantomData<&'p mut Pipeline>,
    }

    impl Pipeline {
        fn poll_next(
            &mut self,
            batch_id: u64,
            pos: &mut usize,
            remaining: &mut usize,
        ) -> Option<std::io::Result<Slab>> {
            if *remaining == 0 {
                return None;
            }

            loop {
                if let Some(slab) = self.try_yield(batch_id, *pos) {
                    *pos += 1;
                    *remaining -= 1;
                    return Some(Ok(slab));
                }

                if self.pending_completions.is_empty() && self.inflight() == 0 {
                    return Some(Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "pipeline: missing completion",
                    )));
                }

                if let Err(e) = self.drain_one() {
                    return Some(Err(e));
                }
            }
        }
    }

    impl<'p> Iterator for PipelineBatch<'p> {
        type Item = std::io::Result<Slab>;

        fn next(&mut self) -> Option<Self::Item> {
            let pl: &mut Pipeline = unsafe { &mut *(self.pipeline as *mut Pipeline) };
            let t0 = Instant::now();
            let result = pl.poll_next(self.batch_id, &mut self.pos, &mut self.remaining);
            pl.stats.t_yield_ns += t0.elapsed().as_nanos() as u64;
            result
        }
    }

    impl Drop for PipelineBatch<'_> {
        fn drop(&mut self) {
            self.remaining = 0;
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::io::Write;

        /// Helper: create a temp file with known content, return (path, file).
        fn temp_file(size: usize, label: &str) -> (std::path::PathBuf, std::fs::File) {
            let dir = std::env::temp_dir();
            let path = dir.join(format!("pipeline_test_{}_{}", label, std::process::id()));
            let mut f = std::fs::File::create(&path).unwrap();
            let buf = vec![0xABu8; size.next_multiple_of(4096)];
            f.write_all(&buf).unwrap();
            f.flush().unwrap();
            let path2 = path.clone();
            (path, std::fs::File::open(&path2).unwrap())
        }

        fn cleanup(path: &std::path::Path) {
            let _ = std::fs::remove_file(path);
        }

        #[test]
        fn test_pipeline_single_read() {
            let (path, _f) = temp_file(8192, "single_read");
            let mut pl = Pipeline::open(
                &path,
                PipelineConfig { qd: 4, max_slots: 8, buf_alloc: None },
            )
            .unwrap();

            let reads = vec![Read { offset: 0, len: 4096 }];
            let batch = pl.submit(&reads).unwrap();
            let results: Vec<_> = batch.collect::<std::io::Result<Vec<_>>>().unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].payload().len(), 4096);
            assert_eq!(results[0].payload()[0], 0xAB);
            assert_eq!(pl.stats().reads, 1);
            assert!(pl.stats().bytes_payload >= 4096);
            cleanup(&path);
        }

        #[test]
        fn test_pipeline_multiple_reads_ordered() {
            let (path, _f) = temp_file(65536, "multiple_reads_ordered");
            let mut pl = Pipeline::open(
                &path,
                PipelineConfig { qd: 4, max_slots: 8, buf_alloc: None },
            )
            .unwrap();

            let reads = vec![
                Read { offset: 0, len: 4096 },
                Read { offset: 4096, len: 4096 },
                Read { offset: 8192, len: 4096 },
            ];
            let batch = pl.submit(&reads).unwrap();
            let results: Vec<_> = batch.collect::<std::io::Result<Vec<_>>>().unwrap();
            assert_eq!(results.len(), 3);
            for (i, slab) in results.iter().enumerate() {
                assert_eq!(slab.payload().len(), 4096);
                assert_eq!(slab.payload()[0], 0xAB, "slab {i} content mismatch");
            }
            cleanup(&path);
        }

        #[test]
        fn test_pipeline_backpressure_pool_exhaustion() {
            let (path, _f) = temp_file(1 << 20, "backpressure");
            let mut pl = Pipeline::open(
                &path,
                PipelineConfig { qd: 2, max_slots: 2, buf_alloc: None },
            )
            .unwrap();

            let reads: Vec<Read> = (0..8)
                .map(|i| Read { offset: (i as u64) * 4096, len: 4096 })
                .collect();
            let batch = pl.submit(&reads).unwrap();
            let results: Vec<_> = batch.collect::<std::io::Result<Vec<_>>>().unwrap();
            assert_eq!(results.len(), 8);
            for slab in &results {
                assert_eq!(slab.payload().len(), 4096);
            }
            cleanup(&path);
        }

        #[test]
        fn test_pipeline_empty_batch() {
            let (path, _f) = temp_file(4096, "empty_batch");
            let mut pl = Pipeline::open(
                &path,
                PipelineConfig { qd: 4, max_slots: 8, buf_alloc: None },
            )
            .unwrap();

            let batch = pl.submit(&[]).unwrap();
            let results: Vec<_> = batch.collect::<std::io::Result<Vec<_>>>().unwrap();
            assert!(results.is_empty());
            cleanup(&path);
        }

        #[test]
        fn test_pipeline_multiple_batches() {
            let (path, _f) = temp_file(65536, "multiple_batches");
            let mut pl = Pipeline::open(
                &path,
                PipelineConfig { qd: 4, max_slots: 8, buf_alloc: None },
            )
            .unwrap();

            let reads_a = vec![Read { offset: 0, len: 4096 }];
            let reads_b = vec![Read { offset: 4096, len: 4096 }];

            let batch_a = pl.submit(&reads_a).unwrap();
            let res_a: Vec<_> = batch_a.collect::<std::io::Result<Vec<_>>>().unwrap();
            assert_eq!(res_a.len(), 1);
            assert_eq!(res_a[0].payload()[0], 0xAB);

            let batch_b = pl.submit(&reads_b).unwrap();
            let res_b: Vec<_> = batch_b.collect::<std::io::Result<Vec<_>>>().unwrap();
            assert_eq!(res_b.len(), 1);
            assert_eq!(res_b[0].payload()[0], 0xAB);
            cleanup(&path);
        }

        #[test]
        fn test_pipeline_stats_counters() {
            let (path, _f) = temp_file(65536, "stats_counters");
            let mut pl = Pipeline::open(
                &path,
                PipelineConfig { qd: 4, max_slots: 8, buf_alloc: None },
            )
            .unwrap();

            let reads = vec![
                Read { offset: 0, len: 4096 },
                Read { offset: 8192, len: 8192 },
            ];
            let batch = pl.submit(&reads).unwrap();
            let _: Vec<_> = batch.collect::<std::io::Result<Vec<_>>>().unwrap();

            let s = pl.stats();
            assert_eq!(s.reads, 2);
            assert!(s.bytes_payload >= 4096 + 8192);
            assert!(s.bytes_disk >= s.bytes_payload);
            assert!(s.t_submit_ns > 0);
            assert!(s.t_complete_ns > 0);
            assert!(s.t_yield_ns > 0);
            cleanup(&path);
        }

        #[test]
        fn test_pipeline_reset_stats() {
            let (path, _f) = temp_file(65536, "reset_stats");
            let mut pl = Pipeline::open(
                &path,
                PipelineConfig { qd: 4, max_slots: 8, buf_alloc: None },
            )
            .unwrap();

            let reads = vec![Read { offset: 0, len: 4096 }];
            let batch = pl.submit(&reads).unwrap();
            let _: Vec<_> = batch.collect::<std::io::Result<Vec<_>>>().unwrap();
            assert!(pl.stats().reads > 0);

            pl.reset_stats();
            assert_eq!(pl.stats().reads, 0);
            assert_eq!(pl.stats().bytes_payload, 0);
            cleanup(&path);
        }

        #[test]
        fn test_pipeline_cancel_drop() {
            let (path, _f) = temp_file(1 << 20, "cancel_drop");
            let mut pl = Pipeline::open(
                &path,
                PipelineConfig { qd: 4, max_slots: 8, buf_alloc: None },
            )
            .unwrap();

            let reads: Vec<Read> = (0..16)
                .map(|i| Read { offset: (i as u64) * 4096, len: 4096 })
                .collect();

            let batch = pl.submit(&reads).unwrap();
            drop(batch);

            let reads2 = vec![Read { offset: 0, len: 4096 }];
            let batch2 = pl.submit(&reads2).unwrap();
            let results: Vec<_> = batch2.collect::<std::io::Result<Vec<_>>>().unwrap();
            assert_eq!(results.len(), 1);
            cleanup(&path);
        }

        #[test]
        fn test_pipeline_large_reads() {
            let (path, _f) = temp_file(2 << 20, "large_reads");
            let mut pl = Pipeline::open(
                &path,
                PipelineConfig { qd: 4, max_slots: 8, buf_alloc: None },
            )
            .unwrap();

            let reads = vec![Read { offset: 0, len: 1 << 20 }];
            let batch = pl.submit(&reads).unwrap();
            let results: Vec<_> = batch.collect::<std::io::Result<Vec<_>>>().unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].payload().len(), 1 << 20);
            cleanup(&path);
        }

        #[test]
        fn test_pipeline_invalid_offset() {
            let (path, _f) = temp_file(4096, "invalid_offset");
            let mut pl = Pipeline::open(
                &path,
                PipelineConfig { qd: 4, max_slots: 8, buf_alloc: None },
            )
            .unwrap();

            // Read beyond file — io_uring returns a short read (0 bytes), not an error.
            // The slab will have a zero-length payload.
            let reads = vec![Read { offset: 1 << 30, len: 4096 }];
            let batch = pl.submit(&reads).unwrap();
            let results: Vec<_> = batch.collect::<std::io::Result<Vec<_>>>().unwrap();
            // io_uring O_DIRECT reads past EOF return short reads without error
            assert_eq!(results.len(), 1);
            cleanup(&path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_roundtrip() {
        let reads = vec![
            Read { offset: 4096, len: 1536 },
            Read { offset: 1 << 33, len: 4718592 },
        ];
        assert_eq!(plan_from_str(&plan_to_string(&reads)).unwrap(), reads);
    }
}
