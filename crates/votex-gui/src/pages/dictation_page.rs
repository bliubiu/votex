use std::sync::{Arc, Mutex};
use eframe::egui;

use crate::state::AppState;
use crate::pages::page_template::PageLayout;
use crate::theme::colors;
use votex_app::use_case::dictation_use_case::{DictationEvent, DictationUseCase};

/// 实时听写页面
///
/// 麦克风 → 流式识别：中间结果灰字实时刷新，端点定稿后追加到转写区。
/// 事件经页面专属 channel 拉取（不走全局 task_tx——TaskEvent 无流式语义）；
/// `DictationUseCase::start` 本身只派发线程（轻量），重活在 worker 中执行。
pub struct DictationPage;

impl DictationPage {
    pub fn show(ui: &mut egui::Ui, state: &mut AppState) {
        // 拉取本轮听写事件（非阻塞）
        Self::poll_events(state);

        PageLayout::header(ui, "实时听写", "对着麦克风说话，实时转为文字；端点检测自动分句");
        ui.add_space(4.0);

        // ======== 参数区 ========
        PageLayout::card(ui, Some("听写设置"), |ui| {
            PageLayout::param_grid(ui, |ui| {
                ui.label("听写模型:");
                egui::ComboBox::from_id_salt("dictation_model")
                    .selected_text(if state.dictation.model == "streaming-zipformer" {
                        "流式 Zipformer（中英混合）"
                    } else {
                        state.dictation.model.as_str()
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.dictation.model,
                            "streaming-zipformer".to_string(),
                            "流式 Zipformer（中英混合）",
                        );
                    });
                ui.end_row();

                ui.label("转写保存:");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut state.dictation.output_path)
                            .hint_text("留空则不保存文件"),
                    );
                    if ui.button("浏览").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("文本", &["txt"])
                            .save_file()
                        {
                            state.dictation.output_path = path.display().to_string();
                        }
                    }
                });
                ui.end_row();
            });
            ui.label(
                egui::RichText::new("首次使用请先在「系统设置」下载模型 streaming-zipformer-zh-en")
                    .size(11.0)
                    .color(colors::STONE_GRAY),
            );
        });

        ui.add_space(12.0);

        // ======== 操作区 ========
        PageLayout::card(ui, None, |ui| {
            ui.horizontal(|ui| {
                let can_start = !state.dictation.is_running;
                if PageLayout::primary_button(ui, "◉ 开始听写", can_start).clicked() {
                    Self::start(state);
                }

                let can_stop = state.dictation.is_running;
                if PageLayout::danger_button(ui, "⏹ 停止", can_stop).clicked() {
                    if let Some(handle) = &state.dictation.handle {
                        if let Ok(h) = handle.lock() {
                            h.stop();
                        }
                    }
                    state.dictation.status_message = "正在停止...".to_string();
                }

                if state.dictation.is_running {
                    ui.label(
                        egui::RichText::new(format!(
                            "● 录音中 {:.1}s",
                            state.dictation.elapsed_ms as f64 / 1000.0
                        ))
                        .color(egui::Color32::from_rgb(220, 70, 70)),
                    );
                }
            });

            if !state.dictation.status_message.is_empty() {
                ui.add_space(8.0);
                PageLayout::result_message(ui, &state.dictation.status_message);
            }
        });

        ui.add_space(12.0);

        // ======== 实时中间结果 ========
        if state.dictation.is_running || !state.dictation.partial_text.is_empty() {
            PageLayout::card(ui, Some("正在识别（未定稿）"), |ui| {
                let text = if state.dictation.partial_text.is_empty() {
                    "（等待语音输入...）".to_string()
                } else {
                    state.dictation.partial_text.clone()
                };
                ui.label(
                    egui::RichText::new(&text)
                        .italics()
                        .color(colors::STONE_GRAY),
                );
            });
            ui.add_space(12.0);
        }

        // ======== 转写结果（可编辑） ========
        PageLayout::card(ui, Some("转写结果"), |ui| {
            PageLayout::text_editor(
                ui,
                &mut state.dictation.final_text,
                "定稿文本将逐段追加到这里，可直接编辑...",
                200.0,
            );
        });
    }
}

impl DictationPage {
    /// 非阻塞拉取听写事件，刷新页面状态
    fn poll_events(state: &mut AppState) {
        let Some(rx) = &state.dictation.event_rx else {
            return;
        };
        // 每帧最多处理 32 条事件，避免识别洪峰阻塞 UI
        for _ in 0..32 {
            match rx.try_recv() {
                Ok(DictationEvent::Started) => {
                    state.dictation.status_message = "麦克风就绪，开始聆听".to_string();
                }
                Ok(DictationEvent::Partial { text, elapsed_ms }) => {
                    state.dictation.partial_text = text;
                    state.dictation.elapsed_ms = elapsed_ms;
                }
                Ok(DictationEvent::Final { text, elapsed_ms }) => {
                    if !state.dictation.final_text.is_empty() {
                        state.dictation.final_text.push('\n');
                    }
                    state.dictation.final_text.push_str(&text);
                    state.dictation.partial_text.clear();
                    state.dictation.elapsed_ms = elapsed_ms;
                }
                Ok(DictationEvent::Stopped { full_text, saved_path, .. }) => {
                    // 以工作线程汇总的全文为准（用户在 UI 编辑过的内容被覆盖前已可复制）
                    state.dictation.final_text = full_text;
                    state.dictation.partial_text.clear();
                    state.dictation.is_running = false;
                    state.dictation.status_message = match saved_path {
                        Some(path) => format!("听写结束，已保存: {}", path.display()),
                        None => "听写结束".to_string(),
                    };
                    state.dictation.handle = None;
                    state.dictation.event_rx = None;
                    return; // 会话结束，本轮拉取终止
                }
                Ok(DictationEvent::Error(msg)) => {
                    state.dictation.is_running = false;
                    state.dictation.partial_text.clear();
                    state.dictation.status_message = format!("听写失败: {}", msg);
                    state.dictation.handle = None;
                    state.dictation.event_rx = None;
                    return;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    state.dictation.is_running = false;
                    state.dictation.event_rx = None;
                    break;
                }
            }
        }
    }

    /// 启动听写：建专属事件 channel + 调用用例（内部派发 worker 线程）
    fn start(state: &mut AppState) {
        let (tx, rx) = std::sync::mpsc::channel::<DictationEvent>();

        state.dictation.is_running = true;
        state.dictation.partial_text.clear();
        state.dictation.status_message = "正在加载模型与麦克风...".to_string();
        state.dictation.event_rx = Some(rx);

        let output_path = if state.dictation.output_path.trim().is_empty() {
            None
        } else {
            Some(std::path::PathBuf::from(state.dictation.output_path.trim()))
        };
        let model = state.dictation.model.clone();

        match DictationUseCase::shared().start(
            &model,
            output_path.as_deref(),
            Arc::new(move |event| {
                // 发送失败 = 接收端已丢弃（页面关闭），静默忽略
                let _ = tx.send(event);
            }),
        ) {
            Ok(handle) => {
                state.dictation.handle = Some(Arc::new(Mutex::new(handle)));
            }
            Err(e) => {
                state.dictation.is_running = false;
                state.dictation.event_rx = None;
                state.dictation.status_message = format!("启动听写失败: {}", e);
            }
        }
    }
}
