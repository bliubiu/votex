use std::sync::Arc;
use eframe::egui;

use crate::state::AppState;
use crate::task_runner;
use crate::widgets::audio_player::AudioPlayer;
use crate::pages::page_template::PageLayout;
use crate::theme::colors;

use votex_infra::tts::qwen3_model_selector;

/// TTS 语音合成页面
pub struct TtsPage;

impl TtsPage {
    pub fn show(ui: &mut egui::Ui, state: &mut AppState) {
        PageLayout::header(ui, "语音合成", "将文本转换为语音，支持中文普通话和方言");
        ui.add_space(4.0);

        // ======== 文本输入区 ========
        PageLayout::card(ui, Some("输入文本"), |ui| {
            ui.horizontal(|ui| {
                if ui.button("📂 打开文本文件").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("文本文件", &["txt", "md", "srt", "lrc", "json"])
                        .pick_file()
                    {
                        match votex_infra::encoding::detector::EncodingDetector::read_text_file(&path) {
                            Ok(content) => {
                                state.tts.input_text = content;
                                state.tts.result_message = format!("已加载文件: {}", path.display());
                            }
                            Err(e) => {
                                state.tts.result_message = format!("读取文件失败: {}", e);
                            }
                        }
                    }
                }
            });
            ui.add_space(8.0);
            PageLayout::text_editor(ui, &mut state.tts.input_text, "请输入要合成的文本，或点击上方按钮加载文件...", 160.0);
        });

        ui.add_space(12.0);

        // ======== 参数配置区 ========
        PageLayout::card(ui, Some("参数配置"), |ui| {
            PageLayout::param_grid(ui, |ui| {
                ui.label("TTS 引擎:");
                let engine_changed = {
                    let mut changed = false;
                    egui::ComboBox::from_id_salt("tts_engine")
                        .selected_text(&state.tts.engine)
                        .show_ui(ui, |ui| {
                            if ui.selectable_value(&mut state.tts.engine, "kokoro".to_string(), "Kokoro-82M").changed() { changed = true; }
                            if ui.selectable_value(&mut state.tts.engine, "indextts2".to_string(), "IndexTTS2").changed() { changed = true; }
                            if ui.selectable_value(&mut state.tts.engine, "qwen3".to_string(), "Qwen3-TTS").changed() { changed = true; }
                            if ui.selectable_value(&mut state.tts.engine, "cosyvoice3".to_string(), "CosyVoice3").changed() { changed = true; }
                        });
                    changed
                };
                if engine_changed && state.tts.engine == "qwen3" && state.tts.qwen3_variant.is_empty() {
                    state.tts.show_qwen3_selector = true;
                }
                // 引擎切换时更新设置页的选中引擎（用于资源建议）
                if engine_changed {
                    // 仅在第一次切换到 Qwen3 时弹出资源建议（用户关闭后不再弹出）
                    if state.tts.engine == "qwen3" && !state.tts.resource_tip_dismissed {
                        state.tts.show_resource_recommendation = true;
                    }
                }
                ui.end_row();

                ui.label("语言/方言:");
                egui::ComboBox::from_id_salt("tts_language")
                    .selected_text(&state.tts.language)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.tts.language, "zh".to_string(), "普通话");
                        ui.selectable_value(&mut state.tts.language, "yue".to_string(), "粤语");
                        ui.selectable_value(&mut state.tts.language, "nan".to_string(), "闽南语");
                    });
                ui.end_row();

                ui.label("音色:");
                egui::ComboBox::from_id_salt("tts_voice")
                    .selected_text(&state.tts.voice)
                    .show_ui(ui, |ui| {
                        if state.tts.engine == "kokoro" {
                            ui.selectable_value(&mut state.tts.voice, "zf_001".to_string(), "中文女声 001");
                            ui.selectable_value(&mut state.tts.voice, "zf_002".to_string(), "中文女声 002");
                            ui.selectable_value(&mut state.tts.voice, "zf_003".to_string(), "中文女声 003");
                            ui.selectable_value(&mut state.tts.voice, "zf_004".to_string(), "中文女声 004");
                            ui.selectable_value(&mut state.tts.voice, "zf_005".to_string(), "中文女声 005");
                            ui.separator();
                            ui.selectable_value(&mut state.tts.voice, "zm_009".to_string(), "中文男声 009");
                            ui.selectable_value(&mut state.tts.voice, "zm_010".to_string(), "中文男声 010");
                            ui.selectable_value(&mut state.tts.voice, "zm_011".to_string(), "中文男声 011");
                        } else {
                            ui.selectable_value(&mut state.tts.voice, "default".to_string(), "默认");
                        }
                    });
                ui.end_row();

                ui.label("语速:");
                ui.add(egui::Slider::new(&mut state.tts.speed, 0.5..=2.0).text("倍"));
                ui.end_row();

                ui.label("输出格式:");
                egui::ComboBox::from_id_salt("tts_format")
                    .selected_text(&state.tts.format)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.tts.format, "wav".to_string(), "WAV");
                        ui.selectable_value(&mut state.tts.format, "mp3".to_string(), "MP3");
                        ui.selectable_value(&mut state.tts.format, "m4b".to_string(), "M4B 有声书");
                    });
                ui.end_row();

                ui.label("输出路径:");
                ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut state.tts.output_path);
                    if ui.button("浏览").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("音频", &["wav", "mp3"])
                            .save_file()
                        {
                            state.tts.output_path = path.display().to_string();
                        }
                    }
                });
                ui.end_row();
            });
        });

        // ======== 当前 Qwen3 变体状态显示 ========
        if state.tts.engine == "qwen3" && !state.tts.qwen3_variant.is_empty() {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let variant_label = if state.tts.qwen3_variant == "0.6b" { "0.6B" } else { "1.7B int4" };
                ui.label(format!("模型变体: {}", variant_label));
                if ui.button("🔄 更换").clicked() {
                    state.tts.show_qwen3_selector = true;
                }
            });
        }

        ui.add_space(12.0);

        // ======== 操作区 ========
        PageLayout::card(ui, None, |ui| {
            ui.horizontal(|ui| {
                // 检查：Qwen3 引擎需要先选模型变体
                let needs_variant = state.tts.engine == "qwen3" && state.tts.qwen3_variant.is_empty();
                let can_run = !state.tts.input_text.trim().is_empty() && !state.tts.is_running && !needs_variant;
                if PageLayout::primary_button(ui, "▶ 开始合成", can_run).clicked() {
                    let tx = state.task_tx.as_ref().unwrap().clone();
                    let text = state.tts.input_text.clone();
                    let output_path = state.tts.output_path.clone();
                    let engine = state.tts.engine.clone();
                    let voice = state.tts.voice.clone();
                    let speed = state.tts.speed;
                    let format = state.tts.format.clone();
                    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
                    let model_override = if engine == "qwen3" {
                        Some(if state.tts.qwen3_variant == "0.6b" {
                            "qwen3-tts-0.6b".to_string()
                        } else {
                            "qwen3-tts-1.7b".to_string()
                        })
                    } else {
                        None
                    };

                    state.tts.is_running = true;
                    state.tts.progress = 0.0;
                    state.tts.progress_text = "准备中...".to_string();
                    state.tts.result_message = String::new();
                    // 保存取消令牌，供「取消」按钮触发
                    state.tts.cancel_token = Some(cancel.clone());

                    task_runner::spawn_tts(tx, text, output_path, engine, voice, speed, format, Some(state.tts.language.clone()), model_override, cancel);
                }

                if state.tts.is_running {
                    if PageLayout::danger_button(ui, "⏹ 取消", true).clicked() {
                        // 置位取消令牌，合成循环在下一段边界中断
                        if let Some(ref token) = state.tts.cancel_token {
                            token.store(true, std::sync::atomic::Ordering::SeqCst);
                        }
                        state.tts.is_running = false;
                        state.tts.progress_text = "正在取消...".to_string();
                    }
                }
            });

            // 进度条
            if state.tts.is_running || !state.tts.progress_text.is_empty() {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(&state.tts.progress_text).size(12.0).color(colors::STONE_GRAY));
                    ui.add(
                        egui::ProgressBar::new(state.tts.progress)
                            .fill(colors::WAVE_TEAL)
                            .animate(state.tts.is_running)
                            .show_percentage(),
                    );
                });
            }

            // 结果消息
            if !state.tts.result_message.is_empty() {
                ui.add_space(8.0);
                PageLayout::result_message(ui, &state.tts.result_message);
            }
        });

        // 音频试听
        if !state.audio_player.file_path.is_empty() {
            ui.add_space(12.0);
            AudioPlayer::show(ui, &mut state.audio_player);
        }

        // ======== Qwen3 模型变体选择弹窗 ========
        if state.tts.show_qwen3_selector {
            let models_base = std::env::current_dir()
                .unwrap_or_default()
                .join("models");
            let conditions = qwen3_model_selector::detect_device(&models_base);
            let recommendation = qwen3_model_selector::recommend(&conditions);

            // 默认选中推荐项
            let mut selected = recommendation.recommended_id.clone();

            egui::Window::new("选择 Qwen3-TTS 模型变体")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .fixed_size([460.0, 380.0])
                .show(ui.ctx(), |ui| {
                    ui.vertical_centered(|ui| {
                        ui.heading("🧠 设备检测与模型推荐");
                    });
                    ui.add_space(8.0);

                    // 设备信息概览
                    ui.label("检测结果：");
                    ui.horizontal(|ui| {
                        let ep_label = if conditions.has_gpu { "✅ GPU (DirectML)" } else { "💻 CPU" };
                        let ram_label = format!("{} MB 可用", conditions.available_ram_mb);
                        ui.label(format!("{}  |  {}", ep_label, ram_label));
                    });
                    ui.horizontal(|ui| {
                        ui.label(format!("0.6B 模型: {}", if conditions.model_06_exists { "✅ 已下载" } else { "❌ 未下载" }));
                        ui.label(format!("1.7B 模型: {}", if conditions.model_17_exists { "✅ 已下载" } else { "❌ 未下载" }));
                    });

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);

                    // 推荐说明
                    ui.label(egui::RichText::new("💡 推荐").size(14.0).color(colors::WAVE_TEAL));
                    ui.label(egui::RichText::new(&recommendation.reason).size(13.0).color(colors::STONE_GRAY));
                    ui.add_space(4.0);

                    // 手动选择（Radio 按钮）
                    ui.label(egui::RichText::new("手动选择：").size(14.0));
                    let mut sel_06 = selected == "qwen3-tts-0.6b";
                    let mut sel_17 = selected == "qwen3-tts-1.7b";

                    ui.horizontal(|ui| {
                        ui.add_space(16.0);
                        let is_rec_06 = recommendation.recommended_id == "qwen3-tts-0.6b";
                        let label_06 = if is_rec_06 { "0.6B（推荐）" } else { "0.6B" };
                        if ui.radio(sel_06, label_06).clicked() {
                            sel_06 = true;
                            sel_17 = false;
                            selected = "qwen3-tts-0.6b".to_string();
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.add_space(16.0);
                        let is_rec_17 = recommendation.recommended_id == "qwen3-tts-1.7b";
                        let label_17 = if is_rec_17 { "1.7B int4（推荐）" } else { "1.7B int4" };
                        if ui.radio(sel_17, label_17).clicked() {
                            sel_17 = true;
                            sel_06 = false;
                            selected = "qwen3-tts-1.7b".to_string();
                        }
                    });

                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(
                        "0.6B: 9种预设音色（含方言），加载快，内存占用低\n\
                         1.7B: 6种 VoiceDesign 音色，合成质量更高"
                    ).size(11.0).color(colors::STONE_GRAY));

                    ui.add_space(12.0);

                    // 确认按钮
                    ui.vertical_centered(|ui| {
                        if ui.add_sized([200.0, 36.0], egui::Button::new(
                            egui::RichText::new("✅ 确认选择").size(16.0)
                        )).clicked() {
                            let variant = if selected.contains("0.6b") { "0.6b".to_string() } else { "1.7b".to_string() };
                            state.tts.qwen3_variant = variant;
                            state.tts.show_qwen3_selector = false;
                        }
                    });
                });
        }

        // ======== 资源限制建议弹窗 ========
        if state.tts.show_resource_recommendation {
            egui::Window::new("💡 推理资源建议")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .fixed_size([420.0, 300.0])
                .show(ui.ctx(), |ui| {
                    ui.vertical_centered(|ui| {
                        ui.heading("Qwen3-TTS 资源建议");
                    });
                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);

                    ui.label(egui::RichText::new(
                        "Qwen3-TTS 需要加载 4 个 ONNX 模型（prefill、decode、\n\
                        code_predictor、vocoder），对 CPU 和内存有一定要求。"
                    ).size(13.0));
                    ui.add_space(8.0);

                    ui.label(egui::RichText::new("推荐配置").size(14.0).color(colors::WAVE_TEAL));
                    ui.add_space(4.0);

                    let specs = [
                        ("CPU 线程", "4 线程（可在设置页调整）"),
                        ("内存上限", "4 GB（可在设置页调整）"),
                        ("执行模式", "顺序执行（省内存，推荐）"),
                        ("0.6B 模型", "CPU 建议选择 0.6B 模型，内存占用 ~1.5GB"),
                        ("1.7B 模型", "GPU 建议选择 1.7B int4，内存占用 ~3GB"),
                    ];
                    for (label, desc) in &specs {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(format!("• {}: ", label)).strong());
                            ui.label(*desc);
                        });
                    }

                    ui.add_space(12.0);
                    let cores = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4) as u32;
                    let cpu_recommend = if cores >= 8 { 4 } else { cores.min(2) };
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(
                            &format!("你的设备有 {} 核心，建议设为 {} 线程", cores, cpu_recommend)
                        ).size(11.0).color(colors::STONE_GRAY));
                    });

                    ui.add_space(8.0);

                    ui.vertical_centered(|ui| {
                        if ui.add_sized([200.0, 32.0], egui::Button::new(
                            egui::RichText::new("✅ 知道了").size(15.0)
                        )).clicked() {
                            state.tts.show_resource_recommendation = false;
                            state.tts.resource_tip_dismissed = true;
                        }
                        if ui.add_sized([200.0, 28.0], egui::Button::new(
                            egui::RichText::new("去设置页调整").size(13.0)
                        )).clicked() {
                            state.tts.show_resource_recommendation = false;
                            state.current_page = crate::state::Page::Settings;
                        }
                    });
                });
        }
    }
}
