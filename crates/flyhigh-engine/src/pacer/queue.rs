//! Presentations waiting for their time, ordered by deadline.
use super::source::VIDEO_RATE_MATCH_TOLERANCE;
use serde::{Deserialize, Serialize};

/// A frame the host can show: one of its source frames or one of the engine's outputs, by the host's own ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Frame {
    Source(u32),
    Output(u32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub due: f64,
    /// None while the generated frame is still owed (see `request`).
    pub frame: Option<Frame>,
    /// Shown instead when the generated frame is not there in time.
    pub fallback: Frame,
    pub request: Option<u64>,
    pub mid: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    pub index: Option<usize>,
    pub due_count: usize,
    pub drop_count: usize,
}

#[derive(Debug, Clone)]
pub struct PresentationQueue {
    entries: Vec<Entry>,
    capacity: usize,
    last_presented_due: f64,
}

impl PresentationQueue {
    pub fn new(capacity: usize) -> Self {
        PresentationQueue { entries: Vec::new(), capacity: capacity.max(2), last_presented_due: f64::NEG_INFINITY }
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity.max(2);
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Frames the queue can hold per second at the current presentation delay.
    pub fn capacity_hz(&self, delay_ms: f64) -> f64 {
        self.capacity as f64 * 1000.0 / delay_ms.max(16.0)
    }

    /// Inserts by deadline (after equal deadlines). Over capacity the oldest entry gives way and is returned.
    pub fn enqueue(&mut self, e: Entry) -> Option<Entry> {
        let at = self.entries.partition_point(|x| x.due <= e.due);
        self.entries.insert(at, e);
        if self.entries.len() > self.capacity {
            Some(self.entries.remove(0))
        } else {
            None
        }
    }

    /// Removes entries due before the last presented one (they would step time backwards). Returns how many.
    pub fn remove_stale(&mut self) -> usize {
        let n = self.entries.iter().take_while(|e| e.due < self.last_presented_due).count();
        self.entries.drain(..n);
        n
    }

    /// One tick shows one frame. When the display can service the target, the oldest entry still inside the
    /// lateness contract is shown, so a display that merely matches the target absorbs a one-vsync backlog; entries
    /// older than the contract, and any target the display cannot service, show the newest due entry and drop the
    /// rest. `target_hz` 0 disables recovery.
    pub fn select_due(&self, now: f64, target_hz: f64, display_capacity_hz: f64, lateness_intervals: f64) -> Selection {
        let newest_due = self.entries.iter().rposition(|e| e.due <= now);
        let Some(newest) = newest_due else {
            return Selection { index: None, due_count: 0, drop_count: 0 };
        };
        let can_service = target_hz > 0.0 && display_capacity_hz >= target_hz * (1.0 - VIDEO_RATE_MATCH_TOLERANCE);
        let mut first_recoverable = 0;
        if can_service {
            let budget = lateness_intervals * 1000.0 / target_hz;
            while first_recoverable <= newest && now - self.entries[first_recoverable].due > budget {
                first_recoverable += 1;
            }
        }
        let index = if can_service && first_recoverable <= newest { first_recoverable } else { newest };
        Selection { index: Some(index), due_count: newest + 1, drop_count: index }
    }

    /// The deadline of the last entry presented (negative infinity before the first).
    pub fn last_presented_due(&self) -> f64 {
        self.last_presented_due
    }

    /// Takes the entry at `index` as presented and discards everything before it.
    pub fn present(&mut self, index: usize) -> Entry {
        let e = self.entries[index].clone();
        self.entries.drain(..=index);
        self.last_presented_due = e.due;
        e
    }

    /// The generated frame of a request arrived.
    pub fn fill(&mut self, request: u64, frame: Frame) -> bool {
        match self.entries.iter_mut().find(|e| e.request == Some(request)) {
            Some(e) => {
                e.frame = Some(frame);
                e.request = None;
                true
            }
            None => false,
        }
    }

    /// The request resolved to its fallback (the source frame is enough).
    pub fn resolve_fallback(&mut self, request: u64) -> bool {
        match self.entries.iter_mut().find(|e| e.request == Some(request)) {
            Some(e) => {
                e.frame = Some(e.fallback);
                e.request = None;
                e.mid = false;
                true
            }
            None => false,
        }
    }

    /// A re-timed real frame replaces the queued source presentation of `source`, if it is still queued.
    pub fn replace_source(&mut self, source: u32, frame: Frame) -> bool {
        match self.entries.iter_mut().find(|e| !e.mid && e.frame == Some(Frame::Source(source))) {
            Some(e) => {
                e.frame = Some(frame);
                e.mid = true;
                true
            }
            None => false,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn clear(&mut self) -> usize {
        let n = self.entries.len();
        self.entries.clear();
        n
    }

    pub fn reset(&mut self) {
        self.entries.clear();
        self.last_presented_due = f64::NEG_INFINITY;
    }
}
