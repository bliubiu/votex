use eframe::egui;

use crate::state::AppState;
use crate::pages::page_template::PageLayout;
use crate::theme::colors;

/// 视频页面：AI 短视频生成 / 视频配音 两种模式
pub struct VideoPage;

impl VideoPage {
    pub fn show(ui: &mut egui::Ui, state: &mut AppState) {
        PageLayout::header(ui, "视频工具", "AI 短视频一键生成，或为现有视频配音（转写 → 翻译 → 配音 → 对齐）");
        ui.add_space(4.0);

        // ======== 模式切换 ========
        ui.horizontal(|ui| {
            let gen_sel = state.video.mode == "generate";
            let dub_sel = state.video.mode == "dub";
            if ui
                .add(
                    egui::Button::new(
                        egui::RichText::new("AI 短视频生成")
                            .color(if gen_sel { colors::WAVE_TEAL } else { colors::STONE_GRAY }),
                    )
                    .stroke(if gen_sel {
                        egui::Stroke::new(1.5, colors::WAVE_TEAL)
                    } else {
                        egui::Stroke::NONE
                    }),
                )
                .clicked()
            {
                state.video.mode = "generate".to_string();
            }
            if ui
                .add(
                    egui::Button::new(
                        egui::RichText::new("视频配音")
                            .color(if dub_sel { colors::WAVE_TEAL } else { colors::STONE_GRAY }),
                    )
                    .stroke(if dub_sel {
                        egui::Stroke::new(1.5, colors::WAVE_TEAL)
                    } else {
                        egui::Stroke::NONE
                    }),
                )
                .clicked()
            {
                state.video.mode = "dub".to_string();
            }
        });
        ui.add_space(8.0);

        match state.video.mode.as_str() {
            "dub" => self::show_dub(ui, state),
            _ => self::show_generate(ui, state),
        }

        // 共享进度与结果区
        if state.video.is_running || state.video.progress > 0.0 {
            ui.add_space(8.0);
            ui.add(egui::ProgressBar::new(state.video.progress)
                .fill(colors::WAVE_TEAL)
                .animate(state.video.is_running)
                .text(&state.video.progress_text));
        }

        if !state.video.result_message.is_empty() {
            ui.add_space(8.0);
            PageLayout::result_message(ui, &state.video.result_message);
        }
    }
}

/// 模式一：AI 短视频生成（原有链路：脚本 → TTS → ASR 字幕 → 素材合成）
fn show_generate(ui: &mut egui::Ui, state: &mut AppState) {
    // ======== 脚本输入 ========
    PageLayout::card(ui, Some("脚本内容"), |ui| {
        PageLayout::text_editor(ui, &mut state.video.script, "输入视频脚本内容，每段将对应一个视频片段...", 140.0);
        ui.add_space(8.0);
        if ui.button("AI 生成脚本").clicked() {
            if let Ok(api_key) = std::env::var("DEEPSEEK_API_KEY") {
                if !api_key.is_empty() {
                    use votex_app::use_case::script_generate::{ScriptGenerateRequest, ScriptGenerateUseCase};
                    let use_case = ScriptGenerateUseCase;
                    let req = ScriptGenerateRequest {
                        topic: state.video.script.clone(),
                        style: "科普".to_string(),
                        duration_seconds: Some(180),
                        engine: "deepseek-chat".to_string(),
                        extra_instructions: None,
                    };
                    match use_case.execute(req) {
                        Ok(resp) => state.video.script = resp.content,
                        Err(e) => state.video.result_message = format!("脚本生成失败: {}", e),
                    }
                }
            } else {
                state.video.script = "AI 脚本生成需要先配置 DEEPSEEK_API_KEY 环境变量\n请参考: 系统设置 > API Key 配置".to_string();
            }
        }
    });

    ui.add_space(12.0);

    // ======== 参数配置 ========
    PageLayout::card(ui, Some("参数配置"), |ui| {
        PageLayout::param_grid(ui, |ui| {
            ui.label("输出文件:");
            ui.text_edit_singleline(&mut state.video.output_path);
            ui.end_row();

            ui.label("TTS 引擎:");
            egui::ComboBox::from_id_salt("video_tts_engine")
                .selected_text(&state.video.tts_engine)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.video.tts_engine, "kokoro".to_string(), "Kokoro-82M");
                    ui.selectable_value(&mut state.video.tts_engine, "indextts25".to_string(), "IndexTTS-2.5（粤语）");
                    ui.selectable_value(&mut state.video.tts_engine, "qwen3".to_string(), "Qwen3-TTS");
                    ui.selectable_value(&mut state.video.tts_engine, "cosyvoice3".to_string(), "CosyVoice3");
                });
            ui.end_row();

            ui.label("配音音色:");
            ui.text_edit_singleline(&mut state.video.tts_voice);
            ui.end_row();

            ui.label("素材来源:");
            egui::ComboBox::from_id_salt("video_material_source")
                .selected_text(&state.video.material_source)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.video.material_source, "pexels".to_string(), "Pexels");
                    ui.selectable_value(&mut state.video.material_source, "pixabay".to_string(), "Pixabay");
                });
            ui.end_row();

            ui.label("关键词搜索:");
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut state.video.material_keyword);
                if ui.small_button("翻译").clicked() {
                    VideoPage::translate_keyword(state);
                }
            });
            if !state.video.translated_keyword.is_empty() {
                ui.label(format!("→ 英文: {}", state.video.translated_keyword));
            }
            ui.end_row();
        });
    });

    ui.add_space(12.0);

    // ======== 执行区 ========
    PageLayout::card(ui, None, |ui| {
        ui.horizontal(|ui| {
            if state.video.is_running {
                if PageLayout::danger_button(ui, "停止", true).clicked() {
                    // 置位取消令牌，后台任务在阶段边界中断，
                    // 终态由后台 Error 事件统一复位（与 TTS 页一致）
                    if let Some(ref token) = state.video.cancel_token {
                        token.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                    state.video.progress_text = "正在停止...".to_string();
                }
            } else if PageLayout::primary_button(ui, "▶ 生成视频", true).clicked() {
                state.video.is_running = true;
                state.video.progress = 0.0;
                state.video.progress_text = "开始生成视频...".to_string();
                state.video.result_message.clear();

                let tx = state.task_tx.as_ref().unwrap().clone();
                // 取消令牌由页面持有并复用，停止按钮才能拿到同一个令牌
                let token = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                state.video.cancel_token = Some(std::sync::Arc::clone(&token));
                crate::task_runner::spawn_video(tx, state.video.script.clone(), state.video.output_path.clone(), state.video.tts_engine.clone(), state.video.tts_voice.clone(), 1.0, true, "srt".to_string(), "whisper-base".to_string(), "1920x1080".to_string(), None, None, Some(token));
            }
        });
    });

    // 素材预览
    if !state.video.material_preview_url.is_empty() {
        ui.add_space(12.0);
        PageLayout::card(ui, Some("素材预览"), |ui| {
            ui.label(format!("素材 URL: {}", state.video.material_preview_url));
        });
    }
}

/// 模式二：视频配音（转写 → 翻译 → 逐段配音 → 时长拟合 → 质检 → mux）
fn show_dub(ui: &mut egui::Ui, state: &mut AppState) {
    // ======== 视频与配音参数 ========
    PageLayout::card(ui, Some("配音设置"), |ui| {
        PageLayout::param_grid(ui, |ui| {
            ui.label("源视频:");
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut state.video.dub_video_path);
                if ui.button("浏览").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("视频", &["mp4", "mkv", "mov", "avi", "webm"])
                        .pick_file()
                    {
                        state.video.dub_video_path = path.display().to_string();
                    }
                }
            });
            ui.end_row();

            ui.label("识别引擎:");
            egui::ComboBox::from_id_salt("dub_asr_engine")
                .selected_text(&state.video.dub_asr_engine)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.video.dub_asr_engine, "paraformer".to_string(), "Paraformer");
                    ui.selectable_value(&mut state.video.dub_asr_engine, "sensevoice".to_string(), "SenseVoice");
                    ui.selectable_value(&mut state.video.dub_asr_engine, "qwen3-asr".to_string(), "Qwen3-ASR");
                    ui.selectable_value(&mut state.video.dub_asr_engine, "wenet".to_string(), "WeNet");
                    ui.selectable_value(&mut state.video.dub_asr_engine, "firered-asr".to_string(), "FireRedASR");
                    ui.selectable_value(&mut state.video.dub_asr_engine, "whisper-base".to_string(), "Whisper-base");
                });
            ui.end_row();

            ui.label("配音引擎:");
            egui::ComboBox::from_id_salt("dub_tts_engine")
                .selected_text(&state.video.dub_tts_engine)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.video.dub_tts_engine, "kokoro".to_string(), "Kokoro-82M");
                    ui.selectable_value(&mut state.video.dub_tts_engine, "indextts25".to_string(), "IndexTTS-2.5（粤语）");
                    ui.selectable_value(&mut state.video.dub_tts_engine, "qwen3".to_string(), "Qwen3-TTS");
                    ui.selectable_value(&mut state.video.dub_tts_engine, "cosyvoice3".to_string(), "CosyVoice3");
                });
            ui.end_row();

            ui.label("配音音色:");
            ui.text_edit_singleline(&mut state.video.dub_voice);
            ui.end_row();

            ui.label("配音语速:");
            ui.add(egui::Slider::new(&mut state.video.dub_speed, 0.5..=2.0).text("倍"));
            ui.end_row();

            ui.label("翻译（可选）:");
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("dub_translate")
                    .selected_text(if state.video.dub_translate.is_empty() {
                        "不翻译（用原声文本配音）"
                    } else {
                        &state.video.dub_translate
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.video.dub_translate, String::new(), "不翻译（用原声文本配音）");
                        ui.selectable_value(&mut state.video.dub_translate, "zh-en".to_string(), "中 → 英");
                        ui.selectable_value(&mut state.video.dub_translate, "en-zh".to_string(), "英 → 中");
                        ui.selectable_value(&mut state.video.dub_translate, "zh-ja".to_string(), "中 → 日");
                    });
            });
            ui.end_row();

            ui.label("最大加速:");
            ui.horizontal(|ui| {
                ui.add(egui::Slider::new(&mut state.video.dub_max_tempo, 1.0..=4.0).text("x"));
                ui.label(
                    egui::RichText::new("超长段的变速上限，超出部分顺延")
                        .size(11.0)
                        .color(colors::STONE_GRAY),
                );
            });
            ui.end_row();

            ui.label("输出选项:");
            ui.horizontal(|ui| {
                ui.checkbox(&mut state.video.dub_verify, "二次 ASR 质检");
                ui.checkbox(&mut state.video.dub_keep_bg, "保留背景音");
            });
            ui.end_row();

            ui.label("输出目录:");
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut state.video.dub_output_dir);
                if ui.button("浏览").clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        state.video.dub_output_dir = path.display().to_string();
                    }
                }
            });
            ui.end_row();
        });
    });

    ui.add_space(12.0);

    // ======== 执行区 ========
    PageLayout::card(ui, None, |ui| {
        ui.horizontal(|ui| {
            if state.video.is_running {
                if PageLayout::danger_button(ui, "停止", true).clicked() {
                    if let Some(ref token) = state.video.cancel_token {
                        token.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                    state.video.progress_text = "正在停止...".to_string();
                }
            } else {
                let can_run = !state.video.dub_video_path.trim().is_empty();
                if PageLayout::primary_button(ui, "▶ 开始配音", can_run).clicked() {
                    state.video.is_running = true;
                    state.video.progress = 0.0;
                    state.video.progress_text = "准备中...".to_string();
                    state.video.result_message.clear();

                    let tx = state.task_tx.as_ref().unwrap().clone();
                    let token = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                    state.video.cancel_token = Some(std::sync::Arc::clone(&token));
                    crate::task_runner::spawn_dub(
                        tx,
                        state.video.dub_video_path.clone(),
                        state.video.dub_output_dir.clone(),
                        state.video.dub_asr_engine.clone(),
                        "zh".to_string(),
                        state.video.dub_tts_engine.clone(),
                        state.video.dub_voice.clone(),
                        state.video.dub_speed,
                        Some(state.video.dub_translate.clone()).filter(|s| !s.is_empty()),
                        "dict".to_string(),
                        state.video.dub_max_tempo,
                        state.video.dub_verify,
                        state.video.dub_keep_bg,
                        Some(token),
                    );
                }
            }
        });

        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(
                "配音流程：提取音轨 → 识别原声 →（可选）翻译 → 逐段配音 → 按原时间轴拟合 → 重建字幕 → 替换/混入音轨。\
                 需要系统安装 ffmpeg。"
            )
            .size(11.0)
            .color(colors::STONE_GRAY),
        );
    });
}

impl VideoPage {
    fn translate_keyword(state: &mut AppState) {
        let keyword = state.video.material_keyword.trim().to_string();
        if keyword.is_empty() {
            return;
        }
        let translator = crate::model_detector::create_keyword_translator();
        let result = translator.translate(&keyword);
        state.video.translated_keyword = result;
    }
}
