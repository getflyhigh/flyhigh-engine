//! Preserve fine repeated edges when a single translation aligns the original frames.
use crate::gpu::memory::Buffer;
use crate::ops::step::meta;
use crate::{clone_step, workgroups_2d, Binding, Gpu, SessionExt, Step, Tensor};
use std::collections::HashMap;

type Input<'a> = (usize, &'a wgpu::TextureView);

pub(crate) struct PanRepair {
    size: (u32, u32),
    sampler: wgpu::Sampler,
    phase: Buffer,
    fits: Vec<Buffer>,
    ages: Vec<Option<Tensor>>,
    errors: Buffer,
    levels: [(u32, Buffer); 3],
    searches: HashMap<(usize, usize, usize), Vec<Step>>,
    renders: HashMap<(usize, usize, usize, usize), Step>,
}

impl PanRepair {
    pub(crate) fn new(gpu: &Gpu, w: u32, h: u32, sessions: &[SessionExt]) -> Self {
        let buffer = |label, size, usage| {
            gpu.buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        Self {
            size: (w, h),
            sampler: gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("fh_pan"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            phase: buffer("fh_pan_phase", 16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            fits: (0..sessions.len()).map(|_| buffer("fh_pan_fit", 16, wgpu::BufferUsages::STORAGE)).collect(),
            ages: sessions.iter().map(|s| s.env.get("fast:age").or_else(|| s.env.get("static:age")).cloned()).collect(),
            errors: buffer("fh_pan_errors", 225 * 4, wgpu::BufferUsages::STORAGE),
            levels: [(15u32, 1u32), (9, 4), (9, 16)].map(|(width, divisor)| {
                let params = [w, h, width, divisor, u32::from(divisor == 1), u32::from(divisor == 16), 0, 0];
                (
                    width,
                    gpu.buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("fh_pan_search_params"),
                        contents: bytemuck::cast_slice(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    }),
                )
            }),
            searches: HashMap::new(),
            renders: HashMap::new(),
        }
    }

    pub(crate) fn search(&mut self, gpu: &Gpu, pair: usize, a: Input<'_>, b: Input<'_>, plan: &mut Vec<Step>) {
        let (w, h) = self.size;
        let steps = self.searches.entry((pair, a.0, b.0)).or_insert_with(|| {
            let mut steps = Vec::with_capacity(2 * self.levels.len());
            let fit = &self.fits[pair];
            for (width, params) in &self.levels {
                let samples = [
                    wgpu::BindingResource::TextureView(a.1),
                    wgpu::BindingResource::TextureView(b.1),
                    wgpu::BindingResource::Sampler(&self.sampler),
                    fit.as_entire_binding(),
                    self.errors.as_entire_binding(),
                    params.as_entire_binding(),
                ];
                let select = [self.errors.as_entire_binding(), fit.as_entire_binding(), params.as_entire_binding()];
                steps.push(step(
                    gpu,
                    "pan_search",
                    &samples,
                    (width * width, 1, 1),
                    vec![
                        Binding::Texture(w, h),
                        Binding::Texture(w, h),
                        Binding::Sampler,
                        Binding::Buffer(fit.size()),
                        Binding::Buffer(self.errors.size()),
                        Binding::Params,
                    ],
                    &[fit, &self.errors, params],
                ));
                steps.push(step(
                    gpu,
                    "pan_select",
                    &select,
                    (1, 1, 1),
                    vec![Binding::Buffer(self.errors.size()), Binding::Buffer(fit.size()), Binding::Params],
                    &[&self.errors, fit, params],
                ));
            }
            steps
        });
        plan.extend(steps.iter().map(clone_step));
    }

    pub(crate) fn render(
        &mut self,
        gpu: &Gpu,
        pair: usize,
        a: Input<'_>,
        b: Input<'_>,
        out: Input<'_>,
        t: f32,
    ) -> Step {
        let (w, h) = self.size;
        let age = self.ages[pair].as_ref();
        let age_buffer = age.map_or(gpu.zero.as_ref(), |t| t.buf.as_ref());
        let fit = &self.fits[pair];
        let step = self.renders.entry((pair, a.0, b.0, out.0)).or_insert_with(|| {
            let resources = [
                wgpu::BindingResource::TextureView(a.1),
                wgpu::BindingResource::TextureView(b.1),
                wgpu::BindingResource::Sampler(&self.sampler),
                fit.as_entire_binding(),
                wgpu::BindingResource::TextureView(out.1),
                self.phase.as_entire_binding(),
                age_buffer.as_entire_binding(),
            ];
            step(
                gpu,
                "pan_apply",
                &resources,
                workgroups_2d(w, h, 1),
                vec![
                    Binding::Texture(w, h),
                    Binding::Texture(w, h),
                    Binding::Sampler,
                    Binding::Buffer(16),
                    Binding::Texture(w, h),
                    Binding::Params,
                    Binding::Buffer(age_buffer.size()),
                ],
                &[fit, &self.phase, age_buffer],
            )
        });
        gpu.queue.write_buffer(&self.phase, 0, bytemuck::cast_slice(&[t, age.map_or(0.0, |a| a.scale), 0.0, 0.0]));
        clone_step(step)
    }
}

fn step(
    gpu: &Gpu,
    pipe: &'static str,
    resources: &[wgpu::BindingResource<'_>],
    groups: (u32, u32, u32),
    bindings: Vec<Binding>,
    buffers: &[&Buffer],
) -> Step {
    let entries: Vec<_> = resources
        .iter()
        .enumerate()
        .map(|(binding, resource)| wgpu::BindGroupEntry { binding: binding as u32, resource: resource.clone() })
        .collect();
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(pipe),
        layout: &gpu.pipeline(pipe).get_bind_group_layout(0),
        entries: &entries,
    });
    Step::Compute { pipe, bg, groups, meta: meta(bindings, &[], buffers.iter().copied(), None) }
}
