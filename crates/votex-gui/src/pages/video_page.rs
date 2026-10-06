use eframe::egui;

use crate::state::AppState;
use crate::pages::page_template::PageLayout;
use crate::theme::colors;

/// 视频生成页面
pub struct VideoPage;

impl VideoPage {
    pub fn show(ui: &mut egui::Ui, state: &mut AppState) {
        PageLayout::header(ui, "视频生成", "基于脚本一键生成配音视频");
        ui.add_space(4.0);

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
                        Self::translate_keyword(state);
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
                } else {
                    if PageLayout::primary_button(ui, "▶ 生成视频", true).clicked() {
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
                }
            });

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
        });

        // 素材预览
        if !state.video.material_preview_url.is_empty() {
            ui.add_space(12.0);
            PageLayout::card(ui, Some("素材预览"), |ui| {
                ui.label(format!("素材 URL: {}", state.video.material_preview_url));
            });
        }
    }

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
