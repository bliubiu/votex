use eframe::egui;
use rodio::Source;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};
use crate::theme::colors;

/// 章节标记（来自合成时写出的 `{音频}.chapters.json`）
#[derive(Debug, Clone)]
pub struct ChapterMark {
    pub title: String,
    pub start_sec: f32,
    pub end_sec: f32,
}

/// 音频播放器状态
#[derive(Debug, Clone)]
pub struct AudioPlayerState {
    pub is_playing: bool,
    /// 播放位置（秒，由播放线程回写）
    pub position: f32,
    /// 总时长（秒，解码器不可知时用章节表兜底）
    pub duration: f32,
    pub volume: f32,
    pub file_path: String,
    pub file_name: String,
    /// 倍速（0.5~2.0）
    pub speed: f32,
    /// 章节列表（加载 `{file}.chapters.json`）
    pub chapters: Vec<ChapterMark>,
    /// 当前章节索引（无章节 = 0 且列表为空）
    pub current_chapter: usize,
    /// 用户正在拖动进度条（拖动期间不回写后端位置）
    dragging_slider: bool,
}

impl Default for AudioPlayerState {
    fn default() -> Self {
        Self {
            is_playing: false,
            position: 0.0,
            duration: 0.0,
            volume: 0.8,
            file_path: String::new(),
            file_name: String::new(),
            speed: 1.0,
            chapters: Vec::new(),
            current_chapter: 0,
            dragging_slider: false,
        }
    }
}

impl AudioPlayerState {
    /// 切换到新文件：加载章节表 + 恢复上次播放进度（断点续听）
    pub fn open_file(&mut self, path: &str) {
        self.stop();
        self.file_path = path.to_string();
        self.file_name = std::path::Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        self.chapters = load_chapters_json(path);
        self.current_chapter = 0;
        self.position = 0.0;
        self.duration = self
            .chapters
            .last()
            .map(|c| c.end_sec)
            .unwrap_or(0.0);
        // 断点续听：恢复上次进度
        if let Some(repo) = playback_repo() {
            if let Ok(Some(p)) = repo.get(path) {
                self.position = p.position_sec.min(self.duration.max(p.position_sec));
                if self.duration == 0.0 {
                    self.duration = p.duration_sec;
                }
            }
        }
    }

    /// 停止播放（保留进度）
    pub fn stop(&mut self) {
        self.save_progress();
        send_audio_cmd(AudioCmd::Stop);
        self.is_playing = false;
        self.position = 0.0;
    }

    /// 保存播放进度到 SQLite
    pub fn save_progress(&self) {
        if self.file_path.is_empty() || self.position <= 0.0 {
            return;
        }
        if let Some(repo) = playback_repo() {
            let _ = repo.save(&self.file_path, self.position, self.duration);
        }
    }
}

/// 音频播放器组件
pub struct AudioPlayer;

impl AudioPlayer {
    pub fn show(ui: &mut egui::Ui, state: &mut AudioPlayerState) {
        let can_play = !state.file_path.is_empty();
        sync_from_backend(state);

        egui::Frame::new()
            .fill(colors::CARD_BG)
            .corner_radius(egui::CornerRadius::same(8))
            .stroke(egui::Stroke::new(1.0, colors::WAVE_TEAL.linear_multiply(0.15)))
            .inner_margin(egui::Margin::symmetric(14, 10))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if state.file_name.is_empty() && can_play {
                        let path = std::path::Path::new(&state.file_path);
                        state.file_name = path.file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default();
                    }
                    if !state.file_name.is_empty() {
                        ui.label(egui::RichText::new(&state.file_name)
                            .size(12.0)
                            .color(colors::STONE_GRAY));
                        ui.separator();
                    }

                    let play_btn = if state.is_playing {
                        egui::Button::new(egui::RichText::new("⏸").size(16.0))
                            .fill(colors::WAVE_TEAL)
                            .min_size(egui::vec2(32.0, 32.0))
                            .corner_radius(egui::CornerRadius::same(6))
                    } else {
                        egui::Button::new(egui::RichText::new("▶").size(16.0))
                            .fill(colors::WAVE_TEAL.linear_multiply(0.1))
                            .min_size(egui::vec2(32.0, 32.0))
                            .corner_radius(egui::CornerRadius::same(6))
                    };
                    if ui.add_enabled(can_play, play_btn).clicked() {
                        if state.is_playing {
                            send_audio_cmd(AudioCmd::Pause);
                            state.is_playing = false;
                            state.save_progress();
                        } else {
                            send_audio_cmd(AudioCmd::PlayAt(
                                state.file_path.clone(),
                                state.position,
                                state.speed,
                            ));
                            send_audio_cmd(AudioCmd::SetVolume(state.volume));
                            state.is_playing = true;
                        }
                    }

                    if ui.add_enabled(can_play,
                        egui::Button::new(egui::RichText::new("⏹").size(16.0))
                            .min_size(egui::vec2(32.0, 32.0))
                            .corner_radius(egui::CornerRadius::same(6))
                    ).clicked() {
                        state.stop();
                    }

                    ui.add_space(8.0);
                    ui.label(egui::RichText::new(format!(
                        "{} / {}",
                        format_secs(state.position),
                        format_secs(state.duration)
                    ))
                    .size(12.0)
                    .color(colors::INK));

                    // 进度条：拖动结束才 seek（避免拖动过程反复重载解码器）
                    let mut seek_to: Option<f32> = None;
                    let max = state.duration.max(1.0);
                    let slider = egui::Slider::new(&mut state.position, 0.0..=max)
                        .text("");
                    let slider_resp = ui.add_enabled(can_play, slider);
                    state.dragging_slider = slider_resp.is_pointer_button_down_on();
                    if slider_resp.drag_stopped() || slider_resp.lost_focus() {
                        seek_to = Some(state.position);
                        state.dragging_slider = false;
                    }
                    if let Some(t) = seek_to {
                        if can_play {
                            send_audio_cmd(AudioCmd::SeekTo(t));
                            if state.is_playing {
                                send_audio_cmd(AudioCmd::ResumeWithSpeed(state.speed));
                            }
                        }
                    }

                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("音量").size(11.0).color(colors::STONE_GRAY));
                    let vol_changed = ui.add(
                        egui::Slider::new(&mut state.volume, 0.0..=1.0)
                            .text("")
                    ).changed();
                    if vol_changed {
                        send_audio_cmd(AudioCmd::SetVolume(state.volume));
                    }

                    if ui.button("打开").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("音频", &["wav", "mp3", "flac", "ogg", "m4a", "m4b"])
                            .pick_file()
                        {
                            state.open_file(&path.display().to_string());
                        }
                    }
                });

                // 第二行：倍速 + 章节
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("倍速").size(11.0).color(colors::STONE_GRAY));
                    for s in [0.75f32, 1.0, 1.25, 1.5, 2.0] {
                        let label = format!("{:.2}x", s);
                        let selected = (state.speed - s).abs() < 0.01;
                        if ui.add(egui::Button::new(
                            egui::RichText::new(label).size(11.0).color(if selected {
                                colors::WAVE_TEAL
                            } else {
                                colors::STONE_GRAY
                            }),
                        ))
                        .clicked()
                        {
                            state.speed = s;
                            send_audio_cmd(AudioCmd::SetSpeed(s));
                        }
                    }

                    if !state.chapters.is_empty() {
                        ui.separator();
                        // 当前章节
                        state.current_chapter = state
                            .chapters
                            .iter()
                            .position(|c| state.position >= c.start_sec && state.position < c.end_sec)
                            .unwrap_or(state.current_chapter);
                        egui::ComboBox::from_id_salt("章节跳转")
                            .selected_text(
                                state
                                    .chapters
                                    .get(state.current_chapter)
                                    .map(|c| c.title.as_str())
                                    .unwrap_or("章节"),
                            )
                            .width(220.0)
                            .show_ui(ui, |ui| {
                                for (i, ch) in state.chapters.iter().enumerate() {
                                    let is_current = i == state.current_chapter;
                                    if ui
                                        .selectable_label(is_current, &ch.title)
                                        .clicked()
                                    {
                                        state.current_chapter = i;
                                        state.position = ch.start_sec;
                                        send_audio_cmd(AudioCmd::SeekTo(ch.start_sec));
                                        if state.is_playing {
                                            send_audio_cmd(AudioCmd::ResumeWithSpeed(state.speed));
                                        }
                                    }
                                }
                            });
                    }
                });
            });
    }
}

fn format_secs(s: f32) -> String {
    let total = s.max(0.0) as u64;
    format!("{:02}:{:02}:{:02}", total / 3600, (total % 3600) / 60, total % 60)
}

// ===== 章节表加载 =====

/// 加载 `{音频}.chapters.json`
fn load_chapters_json(audio_path: &str) -> Vec<ChapterMark> {
    let json_path = std::path::Path::new(audio_path).with_extension("chapters.json");
    let Ok(content) = std::fs::read_to_string(&json_path) else {
        return Vec::new();
    };
    #[derive(serde::Deserialize)]
    struct Entry {
        title: String,
        start_ms: u64,
        end_ms: u64,
    }
    serde_json::from_str::<Vec<Entry>>(&content)
        .map(|entries| {
            entries
                .into_iter()
                .map(|e| ChapterMark {
                    title: e.title,
                    start_sec: e.start_ms as f32 / 1000.0,
                    end_sec: e.end_ms as f32 / 1000.0,
                })
                .collect()
        })
        .unwrap_or_default()
}

// ===== 播放进度持久化（SQLite） =====

static PLAYBACK_REPO: std::sync::OnceLock<Option<Arc<votex_infra::persistence::playback_repo::SqlitePlaybackRepository>>> =
    std::sync::OnceLock::new();

fn playback_repo(
) -> Option<Arc<votex_infra::persistence::playback_repo::SqlitePlaybackRepository>> {
    PLAYBACK_REPO
        .get_or_init(|| {
            let conn = votex_infra::persistence::db::open_database(
                std::path::Path::new("data/votex.db"),
            )
            .ok()?;
            Some(Arc::new(
                votex_infra::persistence::playback_repo::SqlitePlaybackRepository::new(conn),
            ))
        })
        .clone()
}

// ===== 音频播放后端 =====

enum AudioCmd {
    /// 从指定位置（秒）按倍速播放
    PlayAt(String, f32, f32),
    Pause,
    /// 跳转到指定位置（音频时间，秒）——实现为重载解码器后 seek
    SeekTo(f32),
    /// 跳转后继续播放（保持当前倍速）
    ResumeWithSpeed(f32),
    Stop,
    SetVolume(f32),
    /// 切换倍速（保持当前播放位置）
    SetSpeed(f32),
}

/// 后端共享状态（播放线程 → UI）
struct BackendStatus {
    position: f32,
    duration: f32,
    playing: bool,
}

static AUDIO_TX: std::sync::OnceLock<mpsc::Sender<AudioCmd>> = std::sync::OnceLock::new();
static AUDIO_STATUS: std::sync::OnceLock<std::sync::Mutex<BackendStatus>> =
    std::sync::OnceLock::new();

fn status_cell() -> &'static std::sync::Mutex<BackendStatus> {
    AUDIO_STATUS.get_or_init(|| {
        std::sync::Mutex::new(BackendStatus { position: 0.0, duration: 0.0, playing: false })
    })
}

/// UI 每帧从后端拉取播放位置
fn sync_from_backend(state: &mut AudioPlayerState) {
    if let Ok(st) = status_cell().lock() {
        if state.dragging_slider {
            return; // 拖动进度条期间不回写位置
        }
        if st.playing {
            state.position = st.position;
            state.is_playing = true;
            if st.duration > 0.0 && state.duration == 0.0 {
                state.duration = st.duration;
            }
        } else if state.is_playing {
            // 播放自然结束
            state.is_playing = false;
            state.position = st.position;
            state.save_progress();
        }
    }
}

fn send_audio_cmd(cmd: AudioCmd) {
    let tx = AUDIO_TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || audio_thread(rx));
        tx
    });
    let _ = tx.send(cmd);
}


/// 本地倍速音源（复刻 rodio::source::Speed——其构造函数未公开导出）
///
/// 通过提升有效采样率实现倍速（与 rodio Speed 同款语义：音调随倍速抬升）。
struct SpeedSource<I> {
    input: I,
    factor: f32,
}

impl<I> SpeedSource<I> {
    fn new(input: I, factor: f32) -> Self {
        Self { input, factor: factor.clamp(0.5, 2.0) }
    }
}

impl<I> Iterator for SpeedSource<I>
where
    I: rodio::Source,
    I::Item: rodio::Sample,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.input.next()
    }
}

impl<I> rodio::Source for SpeedSource<I>
where
    I: rodio::Source,
    I::Item: rodio::Sample,
{
    fn current_frame_len(&self) -> Option<usize> {
        self.input.current_frame_len()
    }
    fn channels(&self) -> u16 {
        self.input.channels()
    }
    fn sample_rate(&self) -> u32 {
        (self.input.sample_rate() as f32 * self.factor) as u32
    }
    fn total_duration(&self) -> Option<Duration> {
        self.input.total_duration().map(|d| d.div_f32(self.factor))
    }
    fn try_seek(&mut self, pos: Duration) -> Result<(), rodio::source::SeekError> {
        self.input.try_seek(pos.mul_f32(self.factor))
    }
}
/// 打开文件 + 按倍速构造音源；返回 (sink 总时长, 解码器时长)
fn append_source(
    sink: &rodio::Sink,
    path: &str,
    speed: f32,
) -> Option<f32> {
    let file = std::fs::File::open(path).ok()?;
    let reader = std::io::BufReader::new(file);
    let source = rodio::Decoder::new(reader).ok()?;
    let duration = source
        .total_duration()
        .map(|d| d.as_secs_f32())
        .unwrap_or(0.0);
    let speed = speed.clamp(0.5, 2.0);
    if (speed - 1.0).abs() > 1e-6 {
        sink.append(SpeedSource::new(source, speed));
    } else {
        sink.append(source);
    }
    sink.play();
    Some(duration)
}

/// 跳转到音频时间 `t`（秒）：重载解码器 → seek → 按倍速继续
fn seek_and_play(sink: &rodio::Sink, path: &str, t: f32, speed: f32, play: bool) -> Option<f32> {
    sink.stop();
    let duration = append_source_idle(sink, path, speed)?;
    if t > 0.01 {
        // 对底层可 seek 的解码器（WAV/常见容器）跳转；失败则从头播放
        let _ = sink.try_seek(Duration::from_secs_f64(t.max(0.0) as f64));
    }
    if play {
        sink.play();
    }
    Some(duration)
}

/// 打开文件 + 按倍速构造音源（不自动播放）
fn append_source_idle(sink: &rodio::Sink, path: &str, speed: f32) -> Option<f32> {
    let file = std::fs::File::open(path).ok()?;
    let reader = std::io::BufReader::new(file);
    let source = rodio::Decoder::new(reader).ok()?;
    let duration = source
        .total_duration()
        .map(|d| d.as_secs_f32())
        .unwrap_or(0.0);
    let speed = speed.clamp(0.5, 2.0);
    if (speed - 1.0).abs() > 1e-6 {
        sink.append(SpeedSource::new(source, speed));
    } else {
        sink.append(source);
    }
    Some(duration)
}

fn audio_thread(rx: mpsc::Receiver<AudioCmd>) {
    let (_stream, stream_handle) = match rodio::OutputStream::try_default() {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("音频输出初始化失败: {}", e);
            return;
        }
    };
    let sink = match rodio::Sink::try_new(&stream_handle) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("音频 Sink 创建失败: {}", e);
            return;
        }
    };

    // 当前文件与倍速、音频时间位置（由时间账本推进）
    let mut current_path = String::new();
    let mut current_speed = 1.0f32;
    let mut audio_pos = 0.0f32;
    let mut audio_duration = 0.0f32;
    let mut last_tick = Instant::now();
    let mut paused_by_cmd = false;

    loop {
        // 进度账本：播放中按倍速推进
        let dt = last_tick.elapsed().as_secs_f32();
        last_tick = Instant::now();
        let playing = !sink.empty() && !sink.is_paused() && !paused_by_cmd;
        if playing {
            audio_pos += dt * current_speed;
        }
        {
            let mut st = status_cell().lock().unwrap();
            st.position = audio_pos;
            st.duration = audio_duration;
            st.playing = playing;
        }

        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(AudioCmd::PlayAt(path, from, speed)) => {
                sink.stop();
                current_path = path;
                current_speed = speed.clamp(0.5, 2.0);
                audio_pos = from.max(0.0);
                paused_by_cmd = false;
                audio_duration =
                    append_source(&sink, &current_path, current_speed).unwrap_or(0.0);
                if audio_pos > 0.01 {
                    let _ = sink.try_seek(Duration::from_secs_f64(audio_pos as f64));
                }
                last_tick = Instant::now();
            }
            Ok(AudioCmd::Pause) => {
                sink.pause();
                paused_by_cmd = true;
            }
            Ok(AudioCmd::SeekTo(t)) => {
                audio_pos = t.max(0.0);
                // 停止状态下跳转：仅记录位置，下次 PlayAt/ResumeWithSpeed 生效
            }
            Ok(AudioCmd::ResumeWithSpeed(speed)) => {
                current_speed = speed.clamp(0.5, 2.0);
                paused_by_cmd = false;
                let _ = seek_and_play(
                    &sink,
                    &current_path,
                    audio_pos,
                    current_speed,
                    true,
                )
                .map(|d| audio_duration = d);
                last_tick = Instant::now();
            }
            Ok(AudioCmd::Stop) => {
                sink.stop();
                audio_pos = 0.0;
                paused_by_cmd = false;
                current_path.clear();
            }
            Ok(AudioCmd::SetVolume(v)) => {
                sink.set_volume(v);
            }
            Ok(AudioCmd::SetSpeed(speed)) => {
                // 保持音频位置不变，按新倍速重建音源
                current_speed = speed.clamp(0.5, 2.0);
                if !current_path.is_empty() && !sink.empty() {
                    let _ = seek_and_play(
                        &sink,
                        &current_path,
                        audio_pos,
                        current_speed,
                        true,
                    )
                    .map(|d| audio_duration = d);
                }
                last_tick = Instant::now();
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // 播放自然结束检测
                if !current_path.is_empty() && !paused_by_cmd && sink.empty() {
                    audio_pos = 0.0;
                    current_path.clear();
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}
