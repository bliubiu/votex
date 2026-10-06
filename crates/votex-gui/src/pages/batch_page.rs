use eframe::egui;
use std::sync::Arc;
use crate::state::AppState;
use crate::task_runner;
use crate::pages::page_template::PageLayout;
use crate::theme::colors;

/// 批量任务页面
pub struct BatchPage;

impl BatchPage {
    pub fn show(ui: &mut egui::Ui, state: &mut AppState) {
        PageLayout::header(ui, "批量任务", "批量 TTS 合成 / ASR 识别任务管理");
        ui.add_space(4.0);

        // ======== 参数配置 ========
        PageLayout::card(ui, Some("任务参数"), |ui| {
            PageLayout::param_grid(ui, |ui| {
                ui.label("输入目录:");
                ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut state.batch.input_dir);
                    if ui.button("浏览").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
                            state.batch.input_dir = path.display().to_string();
                        }
                    }
                });
                ui.end_row();

                ui.label("输出目录:");
                ui.text_edit_singleline(&mut state.batch.output_dir);
                ui.end_row();

                ui.label("任务类型:");
                egui::ComboBox::from_id_salt("batch_type")
                    .selected_text(if state.batch.task_list.first().map(|t| t.kind.as_str()) == Some("ASR") { "ASR 识别" } else { "TTS 合成" })
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(state.batch.task_list.first().map_or(true, |t| t.kind != "ASR"), "TTS 合成").clicked() {
                            state.batch.task_list.clear();
                            state.batch.task_list.push(crate::state::BatchTaskItem {
                                name: "批量 TTS".to_string(),
                                kind: "TTS".to_string(),
                                status: "等待中".to_string(),
                                progress: 0.0,
                            });
                        }
                        if ui.selectable_label(state.batch.task_list.first().map_or(false, |t| t.kind == "ASR"), "ASR 识别").clicked() {
                            state.batch.task_list.clear();
                            state.batch.task_list.push(crate::state::BatchTaskItem {
                                name: "批量 ASR".to_string(),
                                kind: "ASR".to_string(),
                                status: "等待中".to_string(),
                                progress: 0.0,
                            });
                        }
                    });
                ui.end_row();

                let is_tts = state.batch.task_list.first().map_or(true, |t| t.kind != "ASR");
                if is_tts {
                    ui.label("TTS 引擎:");
                    egui::ComboBox::from_id_salt("batch_tts_engine")
                        .selected_text(&state.batch.engine)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut state.batch.engine, "kokoro".to_string(), "Kokoro-82M");
                            ui.selectable_value(&mut state.batch.engine, "indextts25".to_string(), "IndexTTS-2.5（粤语）");
                            ui.selectable_value(&mut state.batch.engine, "qwen3".to_string(), "Qwen3-TTS");
                            ui.selectable_value(&mut state.batch.engine, "cosyvoice3".to_string(), "CosyVoice3");
                        });
                    ui.end_row();

                    ui.label("音色:");
                    ui.text_edit_singleline(&mut state.batch.voice);
                    ui.end_row();

                    ui.label("语速:");
                    ui.add(egui::Slider::new(&mut state.batch.speed, 0.5..=2.0).text("倍"));
                    ui.end_row();

                    ui.label("输出格式:");
                    egui::ComboBox::from_id_salt("batch_tts_format")
                        .selected_text(&state.batch.format)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut state.batch.format, "wav".to_string(), "WAV");
                            ui.selectable_value(&mut state.batch.format, "mp3".to_string(), "MP3");
                        });
                    ui.end_row();
                }

                if !is_tts {
                    ui.label("ASR 模型:");
                    egui::ComboBox::from_id_salt("batch_asr_model")
                        .selected_text(&state.batch.model)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut state.batch.model, "whisper-base".to_string(), "Whisper Base");
                            ui.selectable_value(&mut state.batch.model, "whisper-small".to_string(), "Whisper Small");
                            ui.selectable_value(&mut state.batch.model, "sensevoice".to_string(), "SenseVoice");
                        });
                    ui.end_row();

                    ui.label("识别语言:");
                    egui::ComboBox::from_id_salt("batch_asr_lang")
                        .selected_text(&state.batch.language)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut state.batch.language, "zh".to_string(), "中文");
                            ui.selectable_value(&mut state.batch.language, "zhen".to_string(), "中英混合");
                        });
                    ui.end_row();

                    ui.label("输出格式:");
                    egui::ComboBox::from_id_salt("batch_asr_format")
                        .selected_text(&state.batch.format)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut state.batch.format, "srt".to_string(), "SRT 字幕");
                            ui.selectable_value(&mut state.batch.format, "txt".to_string(), "TXT 纯文本");
                            ui.selectable_value(&mut state.batch.format, "lrc".to_string(), "LRC 歌词");
                        });
                    ui.end_row();
                }

                ui.label("并发数:");
                ui.add(egui::Slider::new(&mut state.batch.concurrency, 1..=8).text("线程"));
                ui.end_row();
            });
        });

        ui.add_space(12.0);

        // ======== 操作区 ========
        PageLayout::card(ui, None, |ui| {
            ui.horizontal(|ui| {
                if state.batch.is_running {
                    if PageLayout::danger_button(ui, "⏹ 停止", true).clicked() {
                        // 置位取消令牌，批量循环在条目边界中断
                        if let Some(ref token) = state.batch.cancel_token {
                            token.store(true, std::sync::atomic::Ordering::SeqCst);
                        }
                        state.batch.is_running = false;
                        state.batch.progress_text = "正在停止...".to_string();
                    }
                } else {
                    let can_run = !state.batch.input_dir.trim().is_empty();
                    if PageLayout::primary_button(ui, "▶ 执行批量任务", can_run).clicked() {
                        let tx = state.task_tx.as_ref().unwrap().clone();
                        let is_tts = state.batch.task_list.first().map_or(true, |t| t.kind != "ASR");
                        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
                        state.batch.cancel_token = Some(cancel.clone());

                        state.batch.is_running = true;
                        state.batch.progress = 0.0;
                        state.batch.progress_text = "准备中...".to_string();
                        state.batch.result_message = String::new();

                        if is_tts {
                            task_runner::spawn_batch_tts(tx, state.batch.input_dir.clone(), state.batch.output_dir.clone(), state.batch.engine.clone(), state.batch.voice.clone(), state.batch.speed, state.batch.format.clone(), state.batch.concurrency, cancel);
                        } else {
                            task_runner::spawn_batch_asr(tx, state.batch.input_dir.clone(), state.batch.output_dir.clone(), state.batch.model.clone(), state.batch.language.clone(), state.batch.format.clone(), true, state.batch.concurrency, cancel);
                        }
                    }
                }

                if !state.batch.result_message.is_empty() {
                    ui.add_space(16.0);
                    PageLayout::result_message(ui, &state.batch.result_message);
                }
            });

            if state.batch.is_running || state.batch.progress > 0.0 {
                ui.add_space(8.0);
                ui.add(egui::ProgressBar::new(state.batch.progress)
                    .fill(colors::WAVE_TEAL)
                    .animate(state.batch.is_running)
                    .text(&state.batch.progress_text));
            }
        });

        // ======== 任务列表 ========
        if state.batch.task_list.is_empty() {
            ui.add_space(12.0);
            PageLayout::card(ui, None, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(16.0);
                    ui.label(egui::RichText::new("选择任务类型和输入目录后执行").size(13.0).color(colors::STONE_GRAY));
                    ui.add_space(16.0);
                });
            });
        } else {
            ui.add_space(12.0);
            PageLayout::card(ui, Some("任务列表"), |ui| {
                egui::Grid::new("batch_list")
                    .min_col_width(80.0)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("序号").strong());
                        ui.label(egui::RichText::new("名称").strong());
                        ui.label(egui::RichText::new("类型").strong());
                        ui.label(egui::RichText::new("状态").strong());
                        ui.label(egui::RichText::new("进度").strong());
                        ui.end_row();

                        for (i, task) in state.batch.task_list.iter_mut().enumerate() {
                            ui.label(format!("{}", i + 1));
                            ui.label(&task.name);
                            ui.label(&task.kind);
                            PageLayout::status_label(ui, &task.status, &task.status);
                            ui.add(egui::ProgressBar::new(task.progress)
                                .fill(colors::WAVE_TEAL)
                                .show_percentage());
                            ui.end_row();
                        }
                    });
            });
        }
    }
}
