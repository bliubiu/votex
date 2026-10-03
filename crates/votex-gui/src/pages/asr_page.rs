use std::sync::Arc;
use eframe::egui;

use crate::state::AppState;
use crate::task_runner;
use crate::widgets::subtitle_editor::SubtitleEditor;
use crate::pages::page_template::PageLayout;
use crate::theme::colors;

/// ASR 语音识别页面
pub struct AsrPage;

impl AsrPage {
    pub fn show(ui: &mut egui::Ui, state: &mut AppState) {
        PageLayout::header(ui, "语音识别", "将音频/视频转换为文字，支持中文和中英混合");
        ui.add_space(4.0);

        // ======== 输入文件区 ========
        PageLayout::card(ui, Some("输入文件"), |ui| {
            ui.horizontal(|ui| {
                ui.label("音频/视频路径:");
                ui.text_edit_singleline(&mut state.asr.input_path);
                if ui.button("浏览...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("音频/视频", &["wav", "mp3", "mp4", "mkv", "flac", "ogg"])
                        .pick_file()
                    {
                        state.asr.input_path = path.display().to_string();
                    }
                }
            });
            ui.label(
                egui::RichText::new("支持格式: WAV, MP3, MP4, MKV 等")
                    .size(11.0)
                    .color(colors::STONE_GRAY),
            );
        });

        ui.add_space(12.0);

        // ======== 参数配置区 ========
        PageLayout::card(ui, Some("参数配置"), |ui| {
            PageLayout::param_grid(ui, |ui| {
                ui.label("识别模型:");
                egui::ComboBox::from_id_salt("asr_model")
                    .selected_text(&state.asr.model)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.asr.model, "whisper-base".to_string(), "Whisper Base（快速）");
                        ui.selectable_value(&mut state.asr.model, "whisper-small".to_string(), "Whisper Small（精准）");
                        ui.selectable_value(&mut state.asr.model, "sensevoice".to_string(), "SenseVoice（多语言）");
                        ui.selectable_value(&mut state.asr.model, "paraformer".to_string(), "Paraformer（阿里）");
                        ui.selectable_value(&mut state.asr.model, "qwen3-asr".to_string(), "Qwen3-ASR（通义）");
                        ui.selectable_value(&mut state.asr.model, "firered-asr".to_string(), "FireRedASR");
                        ui.selectable_value(&mut state.asr.model, "wenet".to_string(), "WeNet Conformer");
                    });
                ui.end_row();

                ui.label("识别语言:");
                egui::ComboBox::from_id_salt("asr_language")
                    .selected_text(&state.asr.language)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.asr.language, "zh".to_string(), "中文");
                        ui.selectable_value(&mut state.asr.language, "en".to_string(), "英文");
                        ui.selectable_value(&mut state.asr.language, "zhen".to_string(), "中英混合");
                    });
                ui.end_row();

                ui.label("输出格式:");
                egui::ComboBox::from_id_salt("asr_format")
                    .selected_text(&state.asr.format)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.asr.format, "srt".to_string(), "SRT 字幕");
                        ui.selectable_value(&mut state.asr.format, "lrc".to_string(), "LRC 歌词");
                        ui.selectable_value(&mut state.asr.format, "txt".to_string(), "TXT 纯文本");
                    });
                ui.end_row();

                ui.label("输出路径:");
                ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut state.asr.output_path);
                    if ui.button("浏览").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("字幕", &["srt", "lrc", "txt"])
                            .save_file()
                        {
                            state.asr.output_path = path.display().to_string();
                        }
                    }
                });
                ui.end_row();
            });
        });

        ui.add_space(12.0);

        // ======== 操作区 ========
        PageLayout::card(ui, None, |ui| {
            ui.horizontal(|ui| {
                let can_run = !state.asr.input_path.trim().is_empty() && !state.asr.is_running;
                if PageLayout::primary_button(ui, "▶ 开始识别", can_run).clicked() {
                    let tx = state.task_tx.as_ref().unwrap().clone();
                    let input_path = state.asr.input_path.clone();
                    let output_path = state.asr.output_path.clone();
                    let model = state.asr.model.clone();
                    let language = state.asr.language.clone();
                    let format = state.asr.format.clone();
                    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));

                    state.asr.is_running = true;
                    state.asr.progress = 0.0;
                    state.asr.progress_text = "准备中...".to_string();
                    state.asr.result_message = String::new();
                    state.asr.result_text = String::new();

                    task_runner::spawn_asr(tx, input_path, output_path, model, language, format, cancel);
                }

                if state.asr.is_running {
                    if PageLayout::danger_button(ui, "⏹ 取消", true).clicked() {
                        state.asr.is_running = false;
                        state.asr.progress_text = "已取消".to_string();
                    }
                }
            });

            // 进度条
            if state.asr.is_running || !state.asr.progress_text.is_empty() {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(&state.asr.progress_text).size(12.0).color(colors::STONE_GRAY));
                    ui.add(
                        egui::ProgressBar::new(state.asr.progress)
                            .fill(colors::WAVE_TEAL)
                            .animate(state.asr.is_running)
                            .show_percentage(),
                    );
                });
            }

            // 结果消息
            if !state.asr.result_message.is_empty() {
                ui.add_space(8.0);
                PageLayout::result_message(ui, &state.asr.result_message);
            }
        });

        // 识别结果文本
        if !state.asr.result_text.is_empty() {
            ui.add_space(12.0);
            PageLayout::card(ui, Some("识别结果"), |ui| {
                PageLayout::text_editor(ui, &mut state.asr.result_text, "识别结果可在此编辑...", 160.0);
            });
        }

        // 字幕可视化编辑
        if !state.subtitle_entries.is_empty() {
            ui.add_space(12.0);
            PageLayout::card(ui, Some("字幕编辑"), |ui| {
                SubtitleEditor::show(ui, &mut state.subtitle_entries);
            });
        }
    }
}
