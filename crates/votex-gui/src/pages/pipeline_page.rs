use eframe::egui;

use crate::state::{AppState, PipelineStep};
use crate::pages::page_template::PageLayout;
use crate::theme::colors;

/// 流水线模板页面
pub struct PipelinePage;

impl PipelinePage {
    pub fn show(ui: &mut egui::Ui, state: &mut AppState) {
        PageLayout::header(ui, "流水线", "将多个处理步骤组合为一键执行的流水线模板");
        ui.add_space(4.0);

        // ======== 基本配置 ========
        PageLayout::card(ui, Some("基本配置"), |ui| {
            PageLayout::param_grid(ui, |ui| {
                ui.label("模板名称:");
                ui.text_edit_singleline(&mut state.pipeline.template_name);
                ui.end_row();

                ui.label("输入文件:");
                ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut state.pipeline.input_path);
                    if ui.button("浏览").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("文本", &["txt", "md"])
                            .pick_file()
                        {
                            state.pipeline.input_path = path.display().to_string();
                        }
                    }
                });
                ui.end_row();

                ui.label("输出目录:");
                ui.text_edit_singleline(&mut state.pipeline.output_dir);
                ui.end_row();

                ui.label("TTS 引擎:");
                egui::ComboBox::from_id_salt("pipeline_tts_engine")
                    .selected_text(&state.pipeline.engine)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.pipeline.engine, "kokoro".to_string(), "Kokoro-82M");
                        ui.selectable_value(&mut state.pipeline.engine, "indextts25".to_string(), "IndexTTS-2.5（粤语）");
                        ui.selectable_value(&mut state.pipeline.engine, "qwen3".to_string(), "Qwen3-TTS");
                        ui.selectable_value(&mut state.pipeline.engine, "cosyvoice3".to_string(), "CosyVoice3");
                    });
                ui.end_row();

                ui.label("音色:");
                ui.text_edit_singleline(&mut state.pipeline.voice);
                ui.end_row();

                ui.label("语速:");
                ui.add(egui::Slider::new(&mut state.pipeline.speed, 0.5..=2.0).text("倍"));
                ui.end_row();
            });
        });

        ui.add_space(12.0);

        // ======== 预置模板 ========
        PageLayout::card(ui, Some("预置模板"), |ui| {
            ui.horizontal(|ui| {
                if ui.button("有声书制作").clicked() {
                    state.pipeline.steps = Self::audiobook_template();
                    state.pipeline.template_name = "有声书制作".to_string();
                }
                if ui.button("播客制作").clicked() {
                    state.pipeline.steps = Self::podcast_template();
                    state.pipeline.template_name = "播客制作".to_string();
                }
                if ui.button("视频配音").clicked() {
                    state.pipeline.steps = Self::video_dub_template();
                    state.pipeline.template_name = "视频配音".to_string();
                }
            });
        });

        ui.add_space(12.0);

        // ======== 流水线步骤 ========
        PageLayout::card(ui, Some("流水线步骤"), |ui| {
            let mut remove_idx = None;

            if state.pipeline.steps.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(12.0);
                    ui.label(egui::RichText::new("选择预置模板或手动添加步骤").size(13.0).color(colors::STONE_GRAY));
                    ui.add_space(12.0);
                });
            } else {
                egui::Grid::new("pipeline_steps")
                    .min_col_width(120.0)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("启用").strong());
                        ui.label(egui::RichText::new("步骤").strong());
                        ui.label(egui::RichText::new("类型").strong());
                        ui.label(egui::RichText::new("状态").strong());
                        ui.label(egui::RichText::new("操作").strong());
                        ui.end_row();

                        for i in 0..state.pipeline.steps.len() {
                            let enabled = &mut state.pipeline.steps[i].enabled;
                            ui.checkbox(enabled, "");
                            ui.label(&state.pipeline.steps[i].name);
                            ui.label(&state.pipeline.steps[i].kind);
                            PageLayout::status_label(ui, &state.pipeline.steps[i].status, &state.pipeline.steps[i].status);
                            if ui.small_button("删除").clicked() {
                                remove_idx = Some(i);
                            }
                            ui.end_row();
                        }
                    });
            }

            if let Some(idx) = remove_idx {
                state.pipeline.steps.remove(idx);
            }

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("+ TTS 合成").clicked() {
                    state.pipeline.steps.push(PipelineStep {
                        name: "文本转语音".to_string(),
                        kind: "TTS".to_string(),
                        enabled: true,
                        status: "待执行".to_string(),
                    });
                }
                if ui.button("+ ASR 转写").clicked() {
                    state.pipeline.steps.push(PipelineStep {
                        name: "语音转文字".to_string(),
                        kind: "ASR".to_string(),
                        enabled: true,
                        status: "待执行".to_string(),
                    });
                }
                if ui.button("+ 视频生成").clicked() {
                    state.pipeline.steps.push(PipelineStep {
                        name: "视频合成".to_string(),
                        kind: "Video".to_string(),
                        enabled: true,
                        status: "待执行".to_string(),
                    });
                }
            });
        });

        ui.add_space(12.0);

        // ======== 执行区 ========
        PageLayout::card(ui, None, |ui| {
            ui.horizontal(|ui| {
                if state.pipeline.is_running {
                    if PageLayout::danger_button(ui, "停止", true).clicked() {
                        // 置位取消令牌，后台任务在阶段/片段边界中断，
                        // 终态由后台 Error 事件统一复位（与 TTS 页一致）
                        if let Some(ref token) = state.pipeline.cancel_token {
                            token.store(true, std::sync::atomic::Ordering::SeqCst);
                        }
                        state.pipeline.progress_text = "正在停止...".to_string();
                    }
                } else {
                    if PageLayout::primary_button(ui, "▶ 执行流水线", !state.pipeline.steps.is_empty()).clicked() {
                        state.pipeline.is_running = true;
                        state.pipeline.progress = 0.0;
                        state.pipeline.progress_text = "流水线开始执行...".to_string();
                        state.pipeline.result_message.clear();

                        let tx = state.task_tx.as_ref().unwrap().clone();
                        let kind = if state.pipeline.steps.iter().any(|s| s.kind == "ASR") {
                            "subtitle".to_string()
                        } else {
                            "audiobook".to_string()
                        };
                        // 取消令牌由页面持有并复用，停止按钮才能拿到同一个令牌
                        let token = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                        state.pipeline.cancel_token = Some(std::sync::Arc::clone(&token));
                        crate::task_runner::spawn_pipeline(tx, kind, state.pipeline.input_path.clone(), state.pipeline.output_dir.clone(), state.pipeline.engine.clone(), state.pipeline.voice.clone(), state.pipeline.speed, state.pipeline_repo.clone(), Some(token));
                    }
                }
            });

            if state.pipeline.is_running || state.pipeline.progress > 0.0 {
                ui.add_space(8.0);
                ui.add(egui::ProgressBar::new(state.pipeline.progress)
                    .fill(colors::WAVE_TEAL)
                    .animate(state.pipeline.is_running)
                    .text(&state.pipeline.progress_text));
            }

            if !state.pipeline.result_message.is_empty() {
                ui.add_space(8.0);
                PageLayout::result_message(ui, &state.pipeline.result_message);
            }
        });
    }

    fn audiobook_template() -> Vec<PipelineStep> {
        vec![
            PipelineStep { name: "长文本分段".to_string(), kind: "预处理".to_string(), enabled: true, status: "待执行".to_string() },
            PipelineStep { name: "TTS 配音合成".to_string(), kind: "TTS".to_string(), enabled: true, status: "待执行".to_string() },
            PipelineStep { name: "音频降噪".to_string(), kind: "后处理".to_string(), enabled: true, status: "待执行".to_string() },
            PipelineStep { name: "导出音频文件".to_string(), kind: "导出".to_string(), enabled: true, status: "待执行".to_string() },
        ]
    }

    fn podcast_template() -> Vec<PipelineStep> {
        vec![
            PipelineStep { name: "LLM 生成文案".to_string(), kind: "LLM".to_string(), enabled: true, status: "待执行".to_string() },
            PipelineStep { name: "TTS 多角色配音".to_string(), kind: "TTS".to_string(), enabled: true, status: "待执行".to_string() },
            PipelineStep { name: "背景音乐混音".to_string(), kind: "音频".to_string(), enabled: true, status: "待执行".to_string() },
            PipelineStep { name: "导出播客".to_string(), kind: "导出".to_string(), enabled: true, status: "待执行".to_string() },
        ]
    }

    fn video_dub_template() -> Vec<PipelineStep> {
        vec![
            PipelineStep { name: "脚本生成".to_string(), kind: "LLM".to_string(), enabled: true, status: "待执行".to_string() },
            PipelineStep { name: "视频素材搜索".to_string(), kind: "素材".to_string(), enabled: true, status: "待执行".to_string() },
            PipelineStep { name: "TTS 配音".to_string(), kind: "TTS".to_string(), enabled: true, status: "待执行".to_string() },
            PipelineStep { name: "字幕生成".to_string(), kind: "ASR".to_string(), enabled: true, status: "待执行".to_string() },
            PipelineStep { name: "视频合成".to_string(), kind: "Video".to_string(), enabled: true, status: "待执行".to_string() },
        ]
    }
}
