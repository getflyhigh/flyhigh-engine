// The reference native host: a window that plays any ffmpeg input (video file, image pattern, capture device) through
// the engine's Stream and Pacer, the shape of "plug the engine in and it works". No CPU readback: source frames go into
// the Stream's input ring, synthesised frames land in its output slots, and the window blits whichever slot the Pacer
// names on each display tick. Everything a native host has to do is in this file, and nothing else.
//   flyhigh-player <input> [mode=auto|2..6|hz|target:N] [hz=<display rate>] [realtime|video] [cadence] [repair=1]
//                  [k=1] [IWxIH] [fps=<source rate>] [seconds=N] [trace=<file>]
// Decoding runs on a thread (ffmpeg pipe, paced to the source rate); the main thread writes frames into slots,
// classifies pairs, feeds the Pacer and ticks it at the display rate (a timer at `hz`, default the monitor's refresh
// rate). `seconds=N` exits after N s, `trace=` writes the session as `trace v1` for `flyhigh-dev replay`.
use anyhow::{Context, Result};
use flyhigh_engine::pacer::{Frame, Pacer, PacerOptions, RateMode, SourceFrame, Stage};
use flyhigh_engine::stream::{IntervalKind, Mode, Stream, StreamOptions};
use flyhigh_engine::{Engine, EngineOptions, Internal, Model, SessionOptions};
use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

const BLIT: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var o: VOut; o.pos = vec4<f32>(p[i], 0.0, 1.0); o.uv = vec2<f32>(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5); return o;
}
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@fragment fn fs(i: VOut) -> @location(0) vec4<f32> { return textureSample(t, s, i.uv); }
"#;

const INPUTS: u32 = 12;
const OUTPUTS: u32 = 8;
/// Source slots kept out of the ring's way: frames the Pacer may still present.
const HISTORY: usize = 8;

struct Cfg {
    input: String,
    pacer: PacerOptions,
    hz: f64,
    mode: Mode,
    cadence: bool,
    repair: u32,
    k: u32,
    internal: Option<(u32, u32)>,
    fps: f64,
    seconds: f64,
    trace: Option<String>,
}

struct Player {
    cfg: Cfg,
    engine: Engine,
    stream: Stream,
    pacer: Pacer,
    w: u32,
    h: u32,
    window: Option<std::sync::Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    surf_fmt: wgpu::TextureFormat,
    pipe: Option<wgpu::RenderPipeline>,
    layout: Option<wgpu::BindGroupLayout>,
    sampler: Option<wgpu::Sampler>,
    binds: HashMap<Frame, wgpu::BindGroup>,
    rx: mpsc::Receiver<Vec<u8>>,
    start: Instant,
    next_tick: Instant,
    tick_interval: Duration,
    prev_slot: Option<usize>,
    recent: VecDeque<usize>,
    next_out: u32,
    n_src: u64,
    costs_measured: bool,
    // the window of the last log line
    presents: u64,
    mids: u64,
    drops: u64,
    late_ms: f64,
    done: bool,
    exiting: bool,
    sized: bool,
    last_log: Instant,
}

impl Player {
    fn now_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
    fn ensure_pipeline(&mut self) {
        if self.pipe.is_some() {
            return;
        }
        let device = &self.engine.gpu().device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surf_fmt,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        self.pipe = Some(pipe);
        self.layout = Some(layout);
        self.sampler = Some(sampler);
    }
    fn bind_for(&mut self, f: Frame) -> wgpu::BindGroup {
        if let Some(b) = self.binds.get(&f) {
            return b.clone();
        }
        let view = match f {
            Frame::Source(i) => self.stream.input_view(i as usize),
            Frame::Output(i) => self.stream.output_view(i as usize),
        };
        let b = self.engine.gpu().device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: self.layout.as_ref().expect("pipeline built"),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(self.sampler.as_ref().expect("pipeline built")),
                },
            ],
        });
        self.binds.insert(f, b.clone());
        b
    }
    fn present(&mut self, f: Frame) -> Result<()> {
        let bind = self.bind_for(f);
        let gpu = self.engine.gpu();
        let surface = self.surface.as_ref().context("no surface")?;
        let out = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return Ok(()),
        };
        let view = out.texture.create_view(&Default::default());
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            rp.set_pipeline(self.pipe.as_ref().expect("pipeline built"));
            rp.set_bind_group(0, &bind, &[]);
            rp.draw(0..3, 0..1);
        }
        gpu.queue.submit(Some(enc.finish()));
        gpu.queue.present(out);
        Ok(())
    }
    /// Generates the Pacer's requests through the Stream and reports each.
    fn generate(&mut self, now: f64, requests: Vec<flyhigh_engine::pacer::Request>) {
        for r in requests {
            let out = self.next_out;
            self.next_out = (out + 1) % OUTPUTS;
            match self.stream.generate(self.engine.gpu(), r.t, out as usize) {
                Ok(flyhigh_engine::stream::Generated::Written) => {
                    self.pacer.generated(now, r.id, Frame::Output(out));
                }
                Ok(flyhigh_engine::stream::Generated::SourceIsEnough) => self.pacer.skipped(r.id),
                Err(e) => {
                    eprintln!("generate: {e}");
                    self.pacer.skipped(r.id);
                }
            }
        }
    }
    /// A decoded source frame: into a slot, classified, handed to the Pacer, pushed into the Stream.
    fn source(&mut self, rgba: &[u8]) {
        let now = self.now_ms();
        let recent = self.recent.clone();
        let Some(slot) = self.stream.next_input(|s| recent.contains(&s)) else {
            eprintln!("input ring exhausted");
            return;
        };
        let gpu = self.engine.gpu();
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self.stream.input_texture(slot),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.w * 4), rows_per_image: Some(self.h) },
            wgpu::Extent3d { width: self.w, height: self.h, depth_or_array_layers: 1 },
        );
        let (mut kind, mut motion) = (IntervalKind::Motion, None);
        if let Some(prev) = self.prev_slot {
            if let Some(rb) = self.stream.classify(gpu, prev, slot) {
                match rb.block(gpu) {
                    Ok(c) => {
                        kind = c.kind;
                        motion = Some(c.motion);
                    }
                    Err(e) => eprintln!("classify: {e}"),
                }
            }
        }
        // ffmpeg's raw frames carry no timestamps: the source is regular, its clock is the frame count
        let pts = self.n_src as f64 * 1000.0 / self.cfg.fps;
        self.n_src += 1;
        let frame = SourceFrame { id: slot as u32, pts_ms: pts, kind, motion, skipped: 0 };
        let flush = self.pacer.source_frame(now, frame);
        self.generate(now, flush);
        match self.stream.push(self.engine.gpu(), slot, pts) {
            Ok(p) => {
                if p.stats_pending {
                    if let Err(e) = self.stream.resolve_stats(self.engine.gpu()) {
                        eprintln!("stats: {e}");
                    }
                }
                // once, outside a settle window (samples inside one are not taken)
                if p.interval.is_some() && !self.costs_measured && !self.pacer.report().settle_active {
                    self.measure_costs();
                }
            }
            Err(e) => {
                eprintln!("push: {e}");
                self.done = true;
            }
        }
        self.prev_slot = Some(slot);
        self.recent.push_back(slot);
        while self.recent.len() > HISTORY {
            self.recent.pop_front();
        }
    }
    /// GPU time of the pair and of one generated frame, once, from timestamp queries: the Pacer's budget.
    fn measure_costs(&mut self) {
        let now = self.now_ms();
        let pair = self.stream.pair_cost_ms(self.engine.gpu());
        let mid = self.stream.frame_cost_ms(self.engine.gpu(), 0);
        // measured once, but only once the Pacer took it: a sample it refuses (a settle window) leaves it on the seeds
        self.costs_measured = true;
        for (stage, ms) in [(Stage::Pair, pair), (Stage::Mid, mid)] {
            if let Some(ms) = ms {
                self.costs_measured &= self.pacer.cost_sample(now, stage, ms);
            }
        }
        if !self.costs_measured {
            return;
        }
        let r = self.pacer.report();
        println!("[player] GPU costs: pair {:.2} ms, generated frame {:.2} ms", r.pair_cost_ms, r.mid_cost_ms);
    }
    fn tick(&mut self) {
        let now = self.now_ms();
        let tick = self.pacer.display_tick(now);
        if let Some(p) = tick.present {
            if let Err(e) = self.present(p.frame) {
                eprintln!("present: {e}");
            }
            self.presents += 1;
            self.mids += p.mid as u64;
            self.late_ms = self.late_ms.max(p.late_ms);
        }
        self.drops += tick.dropped as u64;
        self.generate(now, tick.generate);
    }
    fn pump(&mut self, el: &ActiveEventLoop) {
        if self.exiting {
            return;
        }
        // 1. decoded frames (ffmpeg paces them to the source rate)
        while !self.done {
            match self.rx.try_recv() {
                Ok(src) => self.source(&src),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.done = true;
                    break;
                }
            }
        }
        // 2. the display tick
        let now = Instant::now();
        if now >= self.next_tick {
            self.tick();
            self.next_tick += self.tick_interval;
            if self.next_tick < now {
                self.next_tick = now + self.tick_interval;
            }
        }
        if now.duration_since(self.last_log).as_secs_f64() >= 2.0 {
            let dt = now.duration_since(self.last_log).as_secs_f64();
            let r = self.pacer.report();
            println!(
                "[player] {:.1} s: presented {:.1}/s (mids {:.1}) drops {:.1}/s late max {:.1} ms | factor {} penalty {}+{} pressure {:.2} delay {:.0} ms | display {} Hz source {:.2} fps | source frames {}",
                self.start.elapsed().as_secs_f64(),
                self.presents as f64 / dt,
                self.mids as f64 / dt,
                self.drops as f64 / dt,
                self.late_ms,
                r.factor,
                r.penalty,
                r.strain_penalty,
                r.pressure,
                r.delay_ms,
                r.display.display_hz,
                r.source_hz.unwrap_or(0.0),
                self.n_src
            );
            self.presents = 0;
            self.mids = 0;
            self.drops = 0;
            self.late_ms = 0.0;
            self.last_log = now;
        }
        if self.done || (self.cfg.seconds > 0.0 && self.start.elapsed().as_secs_f64() >= self.cfg.seconds) {
            let r = self.pacer.report();
            println!(
                "[player] done: {} presents ({} mids) in {:.1} s, drops {} (stall-exempt {}), penalty {}, stalls {}, cuts {}, holds {}",
                r.counters.presents,
                r.counters.presented_mids,
                self.start.elapsed().as_secs_f64(),
                r.counters.drops,
                r.counters.drops_stall_exempt,
                r.penalty,
                r.counters.stalls,
                r.counters.cuts,
                r.counters.holds
            );
            if let (Some(path), Some(trace)) = (&self.cfg.trace, self.pacer.trace()) {
                // the report next to the trace: `flyhigh-dev replay` must reproduce it exactly
                let written = std::fs::write(path, trace.to_json())
                    .and_then(|()| std::fs::write(format!("{path}.report.json"), r.to_json()));
                match written {
                    Ok(()) => println!("[player] trace: {path} ({} events) and its report", trace.events.len()),
                    Err(e) => eprintln!("trace: {e}"),
                }
            }
            self.exiting = true;
            el.exit();
        } else {
            el.set_control_flow(ControlFlow::WaitUntil(self.next_tick.min(now + Duration::from_millis(2))));
        }
    }
    fn configure(&self, size: winit::dpi::PhysicalSize<u32>) {
        if let Some(s) = &self.surface {
            let caps = s.get_capabilities(&self.engine.gpu().adapter);
            s.configure(
                &self.engine.gpu().device,
                &wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: self.surf_fmt,
                    color_space: wgpu::SurfaceColorSpace::default(),
                    width: size.width.max(1),
                    height: size.height.max(1),
                    present_mode: wgpu::PresentMode::AutoNoVsync,
                    alpha_mode: caps.alpha_modes[0],
                    view_formats: vec![],
                    desired_maximum_frame_latency: 1,
                },
            );
        }
    }
}

impl ApplicationHandler for Player {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("FlyHigh player")
            .with_inner_size(winit::dpi::PhysicalSize::new((self.w / 2).max(640), (self.h / 2).max(360)));
        let window = std::sync::Arc::new(el.create_window(attrs).expect("window"));
        if self.cfg.hz <= 0.0 {
            // the monitor's refresh rate is the display tick rate; 60 when the window system does not say
            let hz = window
                .current_monitor()
                .and_then(|m| m.refresh_rate_millihertz())
                .map_or(60.0, |mhz| mhz as f64 / 1000.0);
            self.cfg.hz = hz;
            self.tick_interval = Duration::from_secs_f64(1.0 / hz);
            println!("[player] display ticks at {hz:.2} Hz (the monitor's refresh rate)");
        }
        let surface = self.engine.gpu().instance.create_surface(window.clone()).expect("surface");
        let caps = surface.get_capabilities(&self.engine.gpu().adapter);
        self.surf_fmt = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        self.window = Some(window);
        self.surface = Some(surface);
        self.configure(size);
        self.ensure_pipeline();
        self.start = Instant::now();
        self.next_tick = self.start;
        self.last_log = self.start;
        el.set_control_flow(ControlFlow::Poll);
    }
    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                self.configure(size);
                // the first Resized is the window appearing, not a mode change
                if self.sized {
                    let now = self.now_ms();
                    self.pacer.mode_change(now);
                }
                self.sized = true;
            }
            _ => {}
        }
    }
    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            self.pump(el);
        }
    }
}

// Model directory: FH_MODEL_DIR, else the repository's `models/` (two levels above this example), else `models/`
// next to the executable (a copied install).
fn model_dir() -> Result<std::path::PathBuf> {
    if let Some(d) = std::env::var_os("FH_MODEL_DIR") {
        return Ok(std::path::PathBuf::from(d));
    }
    let repo = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("models");
    if repo.join("graph.json").exists() {
        return Ok(repo);
    }
    let local = std::env::current_exe()?.parent().map(|p| p.join("models")).unwrap_or_default();
    anyhow::ensure!(
        local.join("graph.json").exists(),
        "model not found: set FH_MODEL_DIR to the directory with graph.json, flyhighfg.json, flyhighfg.bin"
    );
    Ok(local)
}

fn parse_args(args: &[String]) -> Result<Cfg> {
    let mut cfg = Cfg {
        input: args[1].clone(),
        pacer: PacerOptions::default(),
        hz: 0.0,
        mode: Mode::Realtime,
        cadence: false,
        repair: 1,
        k: 1,
        internal: None,
        fps: 0.0,
        seconds: 0.0,
        trace: None,
    };
    for a in &args[2..] {
        if a == "video" {
            cfg.mode = Mode::Video;
        } else if a == "realtime" {
            cfg.mode = Mode::Realtime;
        } else if a == "cadence" {
            cfg.cadence = true;
        } else if let Some(v) = a.strip_prefix("mode=") {
            cfg.pacer.mode = match v {
                "auto" => RateMode::Auto,
                "hz" => RateMode::DisplayHz,
                t if t.starts_with("target:") => RateMode::Target(t[7..].parse()?),
                n => RateMode::Factor(n.parse()?),
            };
        } else if let Some(v) = a.strip_prefix("factor=") {
            cfg.pacer.mode = RateMode::Factor(v.parse()?);
        } else if let Some(v) = a.strip_prefix("hz=") {
            cfg.hz = v.parse()?;
        } else if let Some(v) = a.strip_prefix("repair=") {
            cfg.repair = v.parse()?;
        } else if let Some(v) = a.strip_prefix("k=") {
            cfg.k = v.parse()?;
        } else if let Some(v) = a.strip_prefix("fps=") {
            cfg.fps = v.parse()?;
        } else if let Some(v) = a.strip_prefix("seconds=") {
            cfg.seconds = v.parse()?;
        } else if let Some(v) = a.strip_prefix("trace=") {
            cfg.trace = Some(v.to_string());
        } else if let Some((w, h)) = a.split_once('x') {
            if let (Ok(w), Ok(h)) = (w.parse::<u32>(), h.parse::<u32>()) {
                cfg.internal = Some((w, h));
            }
        } else if let Ok(f) = a.parse::<u32>() {
            cfg.pacer.mode = RateMode::Factor(f);
        }
    }
    cfg.pacer.lookahead = cfg.mode == Mode::Video;
    Ok(cfg)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 2,
        "usage: flyhigh-player <input> [mode=auto|2..6|hz|target:N] [hz=N] [realtime|video] [cadence] [repair=1] [k=1] [IWxIH] [fps=N] [seconds=N] [trace=file]"
    );
    let mut cfg = parse_args(&args)?;
    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "csv=p=0",
            &cfg.input,
        ])
        .output()
        .context("ffprobe")?;
    let line = String::from_utf8_lossy(&probe.stdout).trim().to_string();
    let f: Vec<&str> = line.split(',').collect();
    anyhow::ensure!(f.len() >= 3, "ffprobe gave '{}'", line);
    let (w, h) = (f[0].parse::<u32>()?, f[1].parse::<u32>()?);
    if cfg.fps <= 0.0 {
        let (n, d) = f[2].split_once('/').unwrap_or((f[2], "1"));
        cfg.fps = n.parse::<f64>().unwrap_or(24.0) / d.parse::<f64>().unwrap_or(1.0);
    }
    if cfg.fps <= 0.0 || cfg.fps > 1000.0 {
        cfg.fps = 24.0;
    }
    let engine = Engine::new(&EngineOptions::from_env())?;
    let model = Model::from_dir(&model_dir()?)?;
    let session = SessionOptions {
        internal: cfg.internal.map_or(Internal::Auto, |(w, h)| Internal::Size(w, h)),
        k: cfg.k,
        repair: cfg.repair,
        fast_render: cfg.mode == Mode::Realtime,
        linear_t: cfg.mode == Mode::Realtime,
        ..SessionOptions::new(w, h)
    };
    let (iw, ih) = session.internal_size();
    println!(
        "[player] {}x{} @ {:.3} fps -> {:?} {:?} cadence={} repair={} k={} internal={}x{} | {:?} | {:?}",
        w,
        h,
        cfg.fps,
        cfg.pacer.mode,
        cfg.mode,
        cfg.cadence,
        cfg.repair,
        cfg.k,
        iw,
        ih,
        engine.info(),
        model.info()
    );
    let sopts =
        StreamOptions { mode: cfg.mode, cadence: cfg.cadence, inputs: INPUTS, outputs: OUTPUTS, ..Default::default() };
    let stream = Stream::open(&engine, &model, &session, &sopts)?;
    cfg.pacer.gpu_timestamps = engine.info().timestamps;
    let mut pacer = Pacer::new(&cfg.pacer)?;
    if cfg.trace.is_some() {
        pacer.record(&format!("flyhigh-player {} {:?}", cfg.input, cfg.mode));
    }
    // decoder thread: ffmpeg -> raw RGBA frames, paced to the source rate so a file is not slurped ahead of time
    let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(4);
    let input = cfg.input.clone();
    std::thread::spawn(move || {
        let mut dec = match std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-re", "-i", &input, "-f", "rawvideo", "-pix_fmt", "rgba", "-"])
            .stdout(std::process::Stdio::piped())
            // the pipe closes when the player exits mid-stream; ffmpeg's complaints about that are not news
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            Ok(d) => d,
            Err(e) => {
                eprintln!("ffmpeg: {e}");
                return;
            }
        };
        let mut src = dec.stdout.take().expect("piped stdout");
        let n = (w * h * 4) as usize;
        loop {
            let mut buf = vec![0u8; n];
            if src.read_exact(&mut buf).is_err() {
                break;
            }
            if tx.send(buf).is_err() {
                break;
            }
        }
        let _ = dec.kill();
    });
    let hz = if cfg.hz > 0.0 { cfg.hz } else { 60.0 };
    let mut app = Player {
        cfg,
        engine,
        stream,
        pacer,
        w,
        h,
        window: None,
        surface: None,
        surf_fmt: wgpu::TextureFormat::Bgra8Unorm,
        pipe: None,
        layout: None,
        sampler: None,
        binds: HashMap::new(),
        rx,
        start: Instant::now(),
        next_tick: Instant::now(),
        tick_interval: Duration::from_secs_f64(1.0 / hz),
        prev_slot: None,
        recent: VecDeque::new(),
        next_out: 0,
        n_src: 0,
        costs_measured: false,
        presents: 0,
        mids: 0,
        drops: 0,
        late_ms: 0.0,
        done: false,
        exiting: false,
        sized: false,
        last_log: Instant::now(),
    };
    let el = EventLoop::new()?;
    el.run_app(&mut app)?;
    Ok(())
}
