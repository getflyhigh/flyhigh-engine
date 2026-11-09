//! The pair classifier: is the interval between two source frames motion, a scene cut or a held frame. A tiny
//! compute pass over the two textures (classify.wgsl) and an 8-byte readback; the decision lands asynchronously,
//! before the interval's first mid is due.
use super::IntervalKind;
use crate::gpu::readback::Readback;
use crate::Gpu;
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

/// Cells sampled per frame.
pub const GRID: (u32, u32) = (48, 27);
/// Readbacks that may be in flight at once (two are, at 120 fps sources); beyond that a pair counts as motion.
const IN_FLIGHT: usize = 3;
/// The kernel's unit: the rgb absolute difference of one cell, 0..765 in 1/255 steps.
const CELL_MAX: f32 = 765.0;
/// A pair over the cut threshold is a cut only when the moving pair before it differed by less than this share of
/// the threshold. A camera flick in a game is a run of large differences (on a 60 fps capture the pair before a
/// "cut" differed by 54 to 98 against a threshold of 90), a cut a lone spike after a calm pair (1 to 35 on anime);
/// taking a flick for a cut left it without in-betweens, a 16.7 ms step in the middle of 4 ms ones.
const CUT_AFTER_CALM: f32 = 0.5;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClassifyParams {
    gw: u32,
    gh: u32,
    _a: u32,
    _b: u32,
}

/// Where motion ends and a cut or a hold begins, in mean per-channel difference (0..1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    /// A pair whose mean difference is above this is a scene cut; None = never.
    pub cut_mean: Option<f32>,
    /// A pair whose mean difference is below this and whose largest cell is below `hold_max` is a held frame.
    pub hold_mean: Option<f32>,
    pub hold_max: f32,
    /// The first change after a held frame is not interpolated (see `Classification::after_hold`).
    pub step_after_hold: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Classification {
    pub kind: IntervalKind,
    /// Mean per-channel absolute difference, 0..1.
    pub mean: f32,
    /// The most different cell, 0..1.
    pub max: f32,
    /// Both frames identical (a black or frozen source, or a protected one the capture could not read).
    pub black: bool,
    /// The statistic the Pacer's motion-adaptive ceiling expects: the mean cell difference in 0..765 units.
    pub motion: f64,
}

impl Classification {
    /// From the kernel's sums: `sum` and `max` in cell units (0..765), over `cells` cells.
    pub fn from_stats(sum: u32, max: u32, cells: u32, th: &Thresholds) -> Classification {
        let mean = sum as f32 / cells.max(1) as f32 / CELL_MAX;
        let max = max as f32 / CELL_MAX;
        let kind = match th.cut_mean {
            Some(c) if mean > c => IntervalKind::Cut,
            _ => match th.hold_mean {
                Some(h) if mean < h && max < th.hold_max => IntervalKind::Hold,
                _ => IntervalKind::Motion,
            },
        };
        Classification { kind, mean, max, black: max == 0.0, motion: mean as f64 * CELL_MAX as f64 }
    }

    /// The same pair once it is known whether the pair before it was a held frame. A change that follows a held frame
    /// is animation drawn on twos or threes (a mouth, a blink, a character against a still background), not motion:
    /// its in-betweens morphed one drawing into the next within one frame time and then stood still for two, four
    /// generated frames in 41 ms and none for 83, which read as a smeared mouth and as a rate jumping between 24 and
    /// 120. Such a pair is shown as it is drawn (a cut for the scheduler: no in-betweens, a new frame). Motion that
    /// goes on from frame to frame is interpolated as ever, from its second pair.
    pub fn after_hold(mut self, held_before: bool, th: &Thresholds) -> Classification {
        if th.step_after_hold && held_before && self.kind == IntervalKind::Motion {
            self.kind = IntervalKind::Cut;
        }
        self
    }

    /// The same pair once the mean difference of the moving pair before it is known (`before`; None at the start):
    /// a cut after a moving pair is the middle of a fast camera move (see `CUT_AFTER_CALM`).
    pub fn after(mut self, before: Option<f32>, th: &Thresholds) -> Classification {
        if let (IntervalKind::Cut, Some(cut), Some(before)) = (self.kind, th.cut_mean, before) {
            if before >= cut * CUT_AFTER_CALM {
                self.kind = IntervalKind::Motion;
            }
        }
        self
    }
}

pub struct Classifier {
    sampler: wgpu::Sampler,
    stats: wgpu::Buffer,
    params: wgpu::Buffer,
    stages: Vec<(wgpu::Buffer, Rc<Cell<bool>>)>,
    bind_groups: HashMap<(usize, usize), wgpu::BindGroup>,
    thresholds: Thresholds,
    /// The mean difference of the last pair that was not a hold, as its readback landed.
    last_moving: Rc<Cell<Option<f32>>>,
    last_held: Rc<Cell<bool>>,
}

impl Classifier {
    pub fn new(gpu: &Gpu, thresholds: Thresholds) -> Classifier {
        use wgpu::util::DeviceExt;
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fh_classify"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let stats = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fh_classify_stats"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fh_classify_params"),
            contents: bytemuck::bytes_of(&ClassifyParams { gw: GRID.0, gh: GRID.1, _a: 0, _b: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let stages = (0..IN_FLIGHT)
            .map(|_| {
                let b = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("fh_classify_stage"),
                    size: 8,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                (b, Rc::new(Cell::new(false)))
            })
            .collect();
        Classifier {
            sampler,
            stats,
            params,
            stages,
            bind_groups: HashMap::new(),
            thresholds,
            last_moving: Rc::new(Cell::new(None)),
            last_held: Rc::new(Cell::new(false)),
        }
    }

    pub fn thresholds(&self) -> &Thresholds {
        &self.thresholds
    }

    /// Submits the classification of the pair (`a`, `b`) (slot index and view each). None when every readback is
    /// still in flight: the pair then counts as ordinary motion.
    pub fn begin(
        &mut self,
        gpu: &Gpu,
        a: (usize, &wgpu::TextureView),
        b: (usize, &wgpu::TextureView),
    ) -> Option<Readback<Classification>> {
        let (stage, busy) = self.stages.iter().find(|(_, busy)| !busy.get()).map(|(s, b)| (s.clone(), b.clone()))?;
        busy.set(true);
        let pipe = gpu.pipeline("classify");
        let bg = self.bind_groups.entry((a.0, b.0)).or_insert_with(|| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fh_classify"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(a.1) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(b.1) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: self.stats.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.params.as_entire_binding() },
                ],
            })
        });
        // one stats buffer serves overlapping classifications: the zeroing, the dispatch and the copy out are
        // queue-ordered per submit
        gpu.queue.write_buffer(&self.stats, 0, &[0u8; 8]);
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fh_classify") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &*bg, &[]);
            pass.dispatch_workgroups(GRID.0.div_ceil(8), GRID.1.div_ceil(8), 1);
        }
        enc.copy_buffer_to_buffer(&self.stats, 0, &stage, 0, 8);
        gpu.queue.submit(Some(enc.finish()));
        let th = self.thresholds;
        let cells = GRID.0 * GRID.1;
        let last_moving = self.last_moving.clone();
        let last_held = self.last_held.clone();
        Some(Readback::new(
            stage,
            Box::new(move |d: &[u8]| {
                let s: &[u32] = bytemuck::cast_slice(&d[..8]);
                let c = Classification::from_stats(s[0], s[1], cells, &th).after(last_moving.get(), &th);
                let held = c.kind == IntervalKind::Hold;
                if !held {
                    last_moving.set(Some(c.mean));
                }
                let c = c.after_hold(last_held.get(), &th);
                last_held.set(held);
                c
            }),
            Some(Box::new(move || busy.set(false))),
        ))
    }

    /// The input textures were recreated: the cached bind groups refer to the old ones.
    pub fn forget_textures(&mut self) {
        self.bind_groups.clear();
    }

    /// A seek or a new source: the pair before the next one is no longer the one this saw.
    pub fn forget_history(&self) {
        self.last_held.set(false);
        self.last_moving.set(None);
    }
}
