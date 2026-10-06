use eframe::egui;

use crate::state::AppState;
use crate::pages::page_template::PageLayout;
use crate::theme::colors;

/// 翻译页面
pub struct TranslationPage;

impl TranslationPage {
    pub fn show(ui: &mut egui::Ui, state: &mut AppState) {
        PageLayout::header(ui, "文本翻译", "多引擎翻译，支持中英互译");
        ui.add_space(4.0);

        // ======== 引擎选择 ========
        PageLayout::card(ui, Some("翻译设置"), |ui| {
            ui.horizontal(|ui| {
                ui.label("翻译引擎:");
                egui::ComboBox::from_id_salt("translation_engine")
                    .selected_text(&state.translation.engine)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.translation.engine, "dict".to_string(), "离线词典");
                        ui.selectable_value(&mut state.translation.engine, "opus-mt".to_string(), "Opus-MT（离线）");
                        ui.selectable_value(&mut state.translation.engine, "nllb-200".to_string(), "NLLB-200（离线）");
                        ui.selectable_value(&mut state.translation.engine, "m2m-100".to_string(), "M2M-100（离线）");
                        ui.selectable_value(&mut state.translation.engine, "hy-mt-1.5".to_string(), "HY-MT1.5（离线）");
                        ui.selectable_value(&mut state.translation.engine, "qwen-mt".to_string(), "Qwen-MT（在线）");
                        ui.selectable_value(&mut state.translation.engine, "llm".to_string(), "LLM（在线）");
                    });

                ui.add_space(16.0);
                ui.label("翻译方向:");
                egui::ComboBox::from_id_salt("translation_direction")
                    .selected_text(&state.translation.direction)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.translation.direction, "zh-en".to_string(), "中文 → 英文");
                        ui.selectable_value(&mut state.translation.direction, "en-zh".to_string(), "英文 → 中文");
                        ui.selectable_value(&mut state.translation.direction, "auto".to_string(), "自动检测");
                    });
            });
        });

        ui.add_space(12.0);

        // ======== 源文本 ========
        PageLayout::card(ui, Some("源文本"), |ui| {
            PageLayout::text_editor(ui, &mut state.translation.source_text, "请输入要翻译的文本...", 140.0);
        });

        ui.add_space(12.0);

        // ======== 操作区 ========
        PageLayout::card(ui, None, |ui| {
            ui.horizontal(|ui| {
                if !state.translation.is_running {
                    if PageLayout::primary_button(ui, "翻译", !state.translation.source_text.trim().is_empty()).clicked() {
                        if state.translation.source_text.trim().is_empty() {
                            state.translation.result_message = "请输入源文本".to_string();
                        } else {
                            state.translation.result_message = String::new();
                            state.translation.is_running = true;
                            state.translation.progress = 0.0;
                            state.translation.progress_text = "翻译中...".to_string();

                            let tx = state.task_tx.as_ref().unwrap().clone();
                            let source_text = state.translation.source_text.clone();
                            let engine = state.translation.engine.clone();
                            let direction = state.translation.direction.clone();
                            // 取消令牌由页面持有并复用，取消按钮才能拿到同一个令牌
                            let token = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                            state.translation.cancel_token = Some(std::sync::Arc::clone(&token));
                            crate::task_runner::spawn_translate(tx, source_text, engine, direction, Some(token));
                        }
                    }
                } else {
                    if PageLayout::danger_button(ui, "取消", true).clicked() {
                        // 置位取消令牌，后台任务在下一个片段边界中断；
                        // 终态由后台 Success/Error 事件统一复位（与 TTS 页一致）
                        if let Some(ref token) = state.translation.cancel_token {
                            token.store(true, std::sync::atomic::Ordering::SeqCst);
                        }
                        state.translation.progress_text = "正在取消...".to_string();
                    }
                }
            });

            if state.translation.is_running {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(egui::RichText::new(&state.translation.progress_text).size(12.0).color(colors::STONE_GRAY));
                });
                ui.add(
                    egui::ProgressBar::new(state.translation.progress)
                        .fill(colors::WAVE_TEAL)
                        .animate(true)
                        .text(format!("{:.0}%", state.translation.progress * 100.0)),
                );
            }

            if !state.translation.result_message.is_empty() {
                ui.add_space(8.0);
                PageLayout::result_message(ui, &state.translation.result_message);
            }
        });

        // ======== 翻译结果 ========
        if !state.translation.target_text.is_empty() {
            ui.add_space(12.0);
            PageLayout::card(ui, Some("翻译结果"), |ui| {
                PageLayout::text_editor(ui, &mut state.translation.target_text, "翻译结果将显示在这里...", 140.0);
            });
        }
    }
}
