use eframe::egui;

use crate::pages::dashboard::DashboardPage;
use crate::pages::tts_page::TtsPage;
use crate::pages::asr_page::AsrPage;
use crate::pages::ocr_page::OcrPage;
use crate::pages::translation_page::TranslationPage;
use crate::pages::batch_page::BatchPage;
use crate::pages::pipeline_page::PipelinePage;
use crate::pages::video_page::VideoPage;
use crate::pages::settings_page::SettingsPage;
use crate::pages::onboarding::OnboardingPage;
use crate::state::AppState;
use crate::task_runner::TaskEvent;
use crate::theme::colors;

/// 导航分组定义：(图标, 页面, 标题)
const NAV_GROUPS: &[(&str, &[(crate::state::Page, &str, &str)])] = &[
    ("常用", &[
        (crate::state::Page::Dashboard, "◇", "首页"),
    ]),
    ("语音处理", &[
        (crate::state::Page::Tts, "▷", "语音合成"),
        (crate::state::Page::Asr, "♪", "语音识别"),
    ]),
    ("文档处理", &[
        (crate::state::Page::Ocr, "◎", "图文识别"),
        (crate::state::Page::Translation, "⇄", "文本翻译"),
    ]),
    ("自动化", &[
        (crate::state::Page::Batch, "▣", "批量任务"),
        (crate::state::Page::Pipeline, "⚡", "流水线"),
        (crate::state::Page::Video, "◆", "视频生成"),
    ]),
    ("系统", &[
        (crate::state::Page::Settings, "⚙", "系统设置"),
    ]),
];

/// 声阅 GUI 应用
pub struct VotexApp {
    pub state: AppState,
}

impl VotexApp {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }

    /// 处理后台任务事件，更新页面状态
    fn process_task_events(&mut self) {
        let Some(ref rx) = self.state.task_rx else { return };

        while let Ok((page, event)) = rx.try_recv() {
            match (page, event) {
                (crate::state::Page::Tts, TaskEvent::Progress { current, total, message }) => {
                    self.state.tts.progress = if total > 0 {
                        current as f32 / total as f32
                    } else {
                        0.0
                    };
                    self.state.tts.progress_text = message;
                }
                (crate::state::Page::Tts, TaskEvent::Success { message, .. }) => {
                    self.state.tts.is_running = false;
                    self.state.tts.progress = 1.0;
                    self.state.tts.progress_text = "完成".to_string();
                    self.state.tts.result_message = message;
                    // 打开新成品文件：加载章节表 + 恢复断点进度
                    let path = self.state.tts.output_path.clone();
                    self.state.audio_player.open_file(&path);
                }
                (crate::state::Page::Tts, TaskEvent::Error { message }) => {
                    self.state.tts.is_running = false;
                    self.state.tts.result_message = message;
                }

                (crate::state::Page::Asr, TaskEvent::Progress { current, total, message }) => {
                    self.state.asr.progress = if total > 0 {
                        current as f32 / total as f32
                    } else {
                        0.0
                    };
                    self.state.asr.progress_text = message;
                }
                (crate::state::Page::Asr, TaskEvent::Success { message, .. }) => {
                    self.state.asr.is_running = false;
                    self.state.asr.progress = 1.0;
                    self.state.asr.progress_text = "完成".to_string();
                    self.state.asr.result_message = message;
                    let out_path = &self.state.asr.output_path;
                    if let Ok(content) = std::fs::read_to_string(out_path) {
                        self.state.asr.result_text = content.clone();
                        if out_path.ends_with(".srt") {
                            self.state.subtitle_entries = parse_srt(&content);
                        }
                    }
                }
                (crate::state::Page::Asr, TaskEvent::Error { message }) => {
                    self.state.asr.is_running = false;
                    self.state.asr.result_message = message;
                }

                (crate::state::Page::Ocr, TaskEvent::Progress { current, total, message }) => {
                    self.state.ocr.progress = if total > 0 {
                        current as f32 / total as f32
                    } else {
                        0.0
                    };
                    self.state.ocr.progress_text = message;
                }
                (crate::state::Page::Ocr, TaskEvent::Success { message, .. }) => {
                    self.state.ocr.is_running = false;
                    self.state.ocr.progress = 1.0;
                    self.state.ocr.result_message = message;
                }
                (crate::state::Page::Ocr, TaskEvent::Error { message }) => {
                    self.state.ocr.is_running = false;
                    self.state.ocr.result_message = message;
                }

                (crate::state::Page::Settings, TaskEvent::Progress { current, total, .. }) => {
                    for model in &mut self.state.settings.model_list {
                        if model.status.contains("下载") || model.status == "未下载" {
                            model.status = format!("下载中... {}%", if total > 0 { current * 100 / total } else { 0 });
                        }
                    }
                }
                (crate::state::Page::Settings, TaskEvent::Success { .. }) => {
                    self.state.settings.model_list.clear();
                }
                (crate::state::Page::Settings, TaskEvent::Error { .. }) => {
                    for model in &mut self.state.settings.model_list {
                        if model.status.contains("下载") {
                            model.status = "下载失败".to_string();
                        }
                    }
                }

                (crate::state::Page::Pipeline, TaskEvent::Progress { current, total, message }) => {
                    self.state.pipeline.progress = if total > 0 {
                        current as f32 / total as f32
                    } else {
                        0.0
                    };
                    self.state.pipeline.progress_text = message;
                }
                (crate::state::Page::Pipeline, TaskEvent::Success { message, .. }) => {
                    self.state.pipeline.is_running = false;
                    self.state.pipeline.progress = 1.0;
                    self.state.pipeline.progress_text = "流水线完成".to_string();
                    self.state.pipeline.result_message = message;
                }
                (crate::state::Page::Pipeline, TaskEvent::Error { message }) => {
                    self.state.pipeline.is_running = false;
                    self.state.pipeline.result_message = message;
                }

                (crate::state::Page::Video, TaskEvent::Progress { current, total, message }) => {
                    self.state.video.progress = if total > 0 {
                        current as f32 / total as f32
                    } else {
                        0.0
                    };
                    self.state.video.progress_text = message;
                }
                (crate::state::Page::Video, TaskEvent::Success { message, .. }) => {
                    self.state.video.is_running = false;
                    self.state.video.progress = 1.0;
                    self.state.video.progress_text = "视频生成完成".to_string();
                    self.state.video.result_message = message;
                }
                (crate::state::Page::Video, TaskEvent::Error { message }) => {
                    self.state.video.is_running = false;
                    self.state.video.result_message = message;
                }

                (crate::state::Page::Translation, TaskEvent::Progress { current, total, message }) => {
                    self.state.translation.progress = if total > 0 {
                        current as f32 / total as f32
                    } else {
                        0.0
                    };
                    self.state.translation.progress_text = message;
                }
                (crate::state::Page::Translation, TaskEvent::Success { message, data }) => {
                    self.state.translation.is_running = false;
                    self.state.translation.progress = 1.0;
                    self.state.translation.progress_text = "翻译完成".to_string();
                    self.state.translation.result_message = message;
                    if let Some(text) = data {
                        self.state.translation.target_text = text;
                    }
                }
                (crate::state::Page::Translation, TaskEvent::Error { message }) => {
                    self.state.translation.is_running = false;
                    self.state.translation.result_message = message;
                }

                (crate::state::Page::Batch, TaskEvent::Progress { current, total, message }) => {
                    self.state.batch.progress = if total > 0 {
                        current as f32 / total as f32
                    } else {
                        0.0
                    };
                    self.state.batch.progress_text = message;
                    if current > 0 && current <= self.state.batch.task_list.len() {
                        self.state.batch.task_list[current - 1].status = "执行中".to_string();
                        self.state.batch.task_list[current - 1].progress = if total > 0 {
                            current as f32 / total as f32
                        } else {
                            0.0
                        };
                    }
                }
                (crate::state::Page::Batch, TaskEvent::Success { message, .. }) => {
                    self.state.batch.is_running = false;
                    self.state.batch.progress = 1.0;
                    self.state.batch.progress_text = "批量任务完成".to_string();
                    self.state.batch.result_message = message;
                    for task in &mut self.state.batch.task_list {
                        if task.status == "执行中" || task.status == "等待中" {
                            task.status = "完成".to_string();
                            task.progress = 1.0;
                        }
                    }
                }
                (crate::state::Page::Batch, TaskEvent::Error { message }) => {
                    self.state.batch.is_running = false;
                    self.state.batch.result_message = message;
                    for task in &mut self.state.batch.task_list {
                        if task.status == "执行中" {
                            task.status = "失败".to_string();
                        }
                    }
                }

                _ => {}
            }
        }
    }
}

impl eframe::App for VotexApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_task_events();

        // 首次启动引导 — 全屏覆盖
        if self.state.show_onboarding {
            OnboardingPage::show(ctx, &mut self.state);
            return;
        }

        // ==================== 顶部状态栏 ====================
        egui::TopBottomPanel::top("top_bar")
            .min_height(44.0)
            .frame(egui::Frame::new()
                .fill(colors::INK)
                .inner_margin(egui::Margin::symmetric(16, 8)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading(egui::RichText::new("声阅")
                        .size(18.0)
                        .color(egui::Color32::WHITE));
                    ui.label(egui::RichText::new("votex")
                        .size(12.0)
                        .color(colors::NAV_TEXT));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // 全局任务指示
                        let has_running = self.state.tts.is_running
                            || self.state.asr.is_running
                            || self.state.ocr.is_running
                            || self.state.batch.is_running
                            || self.state.pipeline.is_running
                            || self.state.video.is_running;
                        if has_running {
                            ui.add(egui::Spinner::new());
                            ui.label(egui::RichText::new("处理中...")
                                .size(12.0)
                                .color(colors::WAVE_TEAL));
                        }
                    });
                });
            });

        // ==================== 底部状态栏 ====================
        egui::TopBottomPanel::bottom("status_bar")
            .min_height(24.0)
            .frame(egui::Frame::new()
                .fill(colors::RICE_PAPER)
                .inner_margin(egui::Margin::symmetric(12, 3)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // 模型就绪计数由 Dashboard 检测，这里简化显示
                    ui.label(egui::RichText::new("● 就绪")
                        .size(11.0)
                        .color(colors::JADE_GREEN));
                    ui.separator();
                    ui.label(egui::RichText::new("离线运行 · 本地处理")
                        .size(11.0)
                        .color(colors::STONE_GRAY));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(egui::RichText::new(concat!("v", env!("CARGO_PKG_VERSION")))
                            .size(11.0)
                            .color(colors::STONE_GRAY));
                    });
                });
            });

        // ==================== 左侧导航栏 ====================
        egui::SidePanel::left("nav_panel")
            .min_width(150.0)
            .default_width(160.0)
            .resizable(false)
            .frame(egui::Frame::new()
                .fill(colors::INK)
                .inner_margin(egui::Margin::symmetric(0, 0)))
            .show(ctx, |ui| {
                let panel_rect = ui.max_rect();

                // 导航内容使用滚动
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_min_width(panel_rect.width());
                        ui.add_space(12.0);

                        for (group_name, pages) in NAV_GROUPS {
                            // 分组标签
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                ui.add_space(16.0);
                                ui.label(egui::RichText::new(*group_name)
                                    .size(10.0)
                                    .color(colors::NAV_GROUP_LABEL));
                            });
                            ui.add_space(2.0);

                            for (page, icon, title) in *pages {
                                let is_active = self.state.current_page == *page;
                                let response = ui.horizontal(|ui| {
                                    ui.add_space(16.0);
                                    // 选中态声波条
                                    if is_active {
                                        let resp_rect = ui.max_rect();
                                        crate::theme::draw_nav_active_bar(ui, &resp_rect);
                                    }
                                    let text_color = if is_active {
                                        colors::NAV_ACTIVE_TEXT
                                    } else {
                                        colors::NAV_TEXT
                                    };
                                    let label = format!("{}  {}", icon, title);
                                    let resp = ui.selectable_label(is_active,
                                        egui::RichText::new(&label)
                                            .size(13.0)
                                            .color(text_color)
                                    );
                                    if resp.clicked() {
                                        self.state.current_page = *page;
                                    }
                                    resp
                                });

                                // 悬浮高亮背景
                                let resp = response.response;
                                if resp.hovered() && !is_active {
                                    ui.painter().rect_filled(
                                        resp.rect,
                                        egui::CornerRadius::same(4),
                                        colors::CHARCOAL.linear_multiply(0.3),
                                    );
                                }
                            }

                            ui.add_space(4.0);
                        }

                        // 底部版本信息
                        ui.add_space(20.0);
                        ui.horizontal(|ui| {
                            ui.add_space(16.0);
                            ui.label(egui::RichText::new("离线语音工具")
                                .size(10.0)
                                .color(colors::NAV_GROUP_LABEL));
                        });
                    });
            });

        // ==================== 中央主内容区 ====================
        egui::CentralPanel::default()
            .frame(egui::Frame::new()
                .fill(colors::RICE_PAPER)
                .inner_margin(egui::Margin::symmetric(20, 16)))
            .show(ctx, |ui| {
                match self.state.current_page {
                    crate::state::Page::Dashboard => {
                        DashboardPage::show(ui, &mut self.state);
                    }
                    crate::state::Page::Tts => {
                        TtsPage::show(ui, &mut self.state);
                    }
                    crate::state::Page::Asr => {
                        AsrPage::show(ui, &mut self.state);
                    }
                    crate::state::Page::Ocr => {
                        OcrPage::show(ui, &mut self.state);
                    }
                    crate::state::Page::Translation => {
                        TranslationPage::show(ui, &mut self.state);
                    }
                    crate::state::Page::Batch => {
                        BatchPage::show(ui, &mut self.state);
                    }
                    crate::state::Page::Pipeline => {
                        PipelinePage::show(ui, &mut self.state);
                    }
                    crate::state::Page::Video => {
                        VideoPage::show(ui, &mut self.state);
                    }
                    crate::state::Page::Settings => {
                        SettingsPage::show(ui, &mut self.state);
                    }
                }
            });

        ctx.request_repaint_after(std::time::Duration::from_millis(200));
    }
}

/// 简单解析 SRT 字幕文本为 SubtitleEntry 列表
fn parse_srt(content: &str) -> Vec<crate::widgets::subtitle_editor::SubtitleEntry> {
    let mut entries = Vec::new();
    let mut lines = content.lines().peekable();

    while let Some(line) = lines.next() {
        if line.trim().is_empty() {
            continue;
        }
        let _index: usize = match line.trim().parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let time_line = match lines.next() {
            Some(l) => l,
            None => break,
        };
        let parts: Vec<&str> = time_line.split("-->").map(|s| s.trim()).collect();
        if parts.len() != 2 {
            continue;
        }
        let mut text = String::new();
        while let Some(next) = lines.next() {
            if next.trim().is_empty() {
                break;
            }
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(next.trim());
        }

        entries.push(crate::widgets::subtitle_editor::SubtitleEntry {
            index: _index,
            start_time: parts[0].to_string(),
            end_time: parts[1].to_string(),
            text,
        });
    }

    entries
}
