use eframe::egui;
use crate::model_detector;
use crate::state::{AppState, Page};
use crate::theme::colors;
use crate::pages::page_template::PageLayout;

/// 首页仪表盘
pub struct DashboardPage;

impl DashboardPage {
    pub fn show(ui: &mut egui::Ui, state: &mut AppState) {
        PageLayout::header(ui, "首页", "快速开始你的语音创作");
        ui.add_space(4.0);

        // ======== 快捷操作卡片区 ========
        PageLayout::card(ui, Some("快捷操作"), |ui| {
            let available_width = ui.available_width();
            let card_width = (available_width - 16.0) / 3.0;

            egui::Grid::new("quick_actions")
                .min_col_width(card_width)
                .max_col_width(card_width)
                .spacing([8.0, 8.0])
                .show(ui, |ui| {
                    Self::action_card(ui, "▷", "语音合成", "文本转语音，支持普通话和方言", colors::WAVE_TEAL, || {
                        state.current_page = Page::Tts;
                    });
                    Self::action_card(ui, "♪", "语音识别", "音频转文字，支持中英混合", colors::BAMBOO_GOLD, || {
                        state.current_page = Page::Asr;
                    });
                    Self::action_card(ui, "◎", "图文识别", "图片/PDF 转文字", colors::STONE_GRAY, || {
                        state.current_page = Page::Ocr;
                    });
                    ui.end_row();

                    Self::action_card(ui, "▣", "批量任务", "批量 TTS / ASR 处理", colors::JADE_GREEN, || {
                        state.current_page = Page::Batch;
                    });
                    Self::action_card(ui, "⚡", "流水线", "一键有声书 / 播客制作", colors::WAVE_TEAL, || {
                        state.current_page = Page::Pipeline;
                    });
                    Self::action_card(ui, "◆", "视频生成", "脚本 + 配音 + 素材合成视频", colors::BAMBOO_GOLD, || {
                        state.current_page = Page::Video;
                    });
                    ui.end_row();
                });
        });

        ui.add_space(12.0);

        // ======== 模型状态概览 ========
        PageLayout::card(ui, Some("模型状态"), |ui| {
            let models = Self::get_model_overview(state);
            let ready_count = models.iter().filter(|m| m.status.contains("就绪")).count();
            let total = models.len();

            ui.horizontal(|ui| {
                // 大数字显示就绪数
                ui.label(egui::RichText::new(format!("{}", ready_count))
                    .size(32.0)
                    .strong()
                    .color(colors::WAVE_TEAL));
                ui.add_space(4.0);
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new("模型就绪").size(13.0).color(colors::STONE_GRAY));
                    ui.label(egui::RichText::new(format!("共 {} 个", total)).size(11.0).color(colors::STONE_GRAY));
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("管理模型").clicked() {
                        state.current_page = Page::Settings;
                    }
                });
            });

            ui.add_space(8.0);

            // 模型列表 — 横向滚动标签
            if models.is_empty() {
                ui.label(egui::RichText::new("暂无模型数据").size(12.0).color(colors::STONE_GRAY));
            } else {
                egui::ScrollArea::horizontal()
                    .max_height(36.0)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            for m in &models {
                                let (bg, fg) = if m.status.contains("就绪") {
                                    (colors::JADE_GREEN.linear_multiply(0.1), colors::JADE_GREEN)
                                } else if m.status.contains("下载") {
                                    (colors::BAMBOO_GOLD.linear_multiply(0.1), colors::BAMBOO_GOLD)
                                } else {
                                    (colors::STONE_GRAY.linear_multiply(0.1), colors::STONE_GRAY)
                                };
                egui::Frame::new()
                    .fill(bg)
                    .corner_radius(egui::CornerRadius::same(12))
                    .inner_margin(egui::Margin::symmetric(10, 4))
                                    .show(ui, |ui| {
                                        ui.label(egui::RichText::new(format!("{} · {}", m.name, m.status))
                                            .size(11.0)
                                            .color(fg));
                                    });
                                ui.add_space(4.0);
                            }
                        });
                    });
            }
        });

        ui.add_space(12.0);

        // ======== 使用提示 ========
        PageLayout::card(ui, Some("使用提示"), |ui| {
            let tips = [
                ("首次使用", "先在「系统设置」中下载所需模型"),
                ("TTS 合成", "支持中文普通话和方言（粤语、闽南语）"),
                ("ASR 识别", "支持简体中文和中英混合识别"),
                ("隐私安全", "所有处理均在本地完成，无需联网"),
            ];
            for (title, desc) in tips {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("·").size(16.0).color(colors::WAVE_TEAL));
                    ui.vertical(|ui| {
                        ui.label(egui::RichText::new(title).size(13.0).strong().color(colors::INK));
                        ui.label(egui::RichText::new(desc).size(12.0).color(colors::STONE_GRAY));
                    });
                });
                ui.add_space(6.0);
            }
        });
    }

    fn action_card(ui: &mut egui::Ui, icon: &str, title: &str, desc: &str, accent: egui::Color32, on_click: impl FnOnce()) {
        let response = egui::Frame::new()
            .fill(colors::CARD_BG)
            .corner_radius(egui::CornerRadius::same(8))
            .stroke(egui::Stroke::new(1.0, accent.linear_multiply(0.2)))
            .inner_margin(egui::Margin::symmetric(14, 12))
            .show(ui, |ui| {
                ui.set_min_size(egui::vec2(0.0, 72.0));
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new(icon).size(22.0).color(accent));
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(title).size(14.0).strong().color(colors::INK));
                    ui.label(egui::RichText::new(desc).size(11.0).color(colors::STONE_GRAY));
                });
            })
            .response
            .interact(egui::Sense::click());

        if response.clicked() {
            on_click();
        }
        // 悬浮效果
        if response.hovered() {
            ui.painter().rect_filled(response.rect, egui::CornerRadius::same(8), accent.linear_multiply(0.05));
        }
    }

    fn get_model_overview(state: &AppState) -> Vec<ModelOverview> {
        let models_dir = if state.settings.models_dir.is_empty() {
            model_detector::default_models_dir()
        } else {
            std::path::PathBuf::from(&state.settings.models_dir)
        };

        model_detector::all_models()
            .iter()
            .map(|info| {
                let ready = model_detector::is_model_ready(&models_dir, &info.kind, &info.id);
                ModelOverview {
                    name: info.name.clone(),
                    kind: info.kind.clone(),
                    status: if ready { "就绪".to_string() } else { "未下载".to_string() },
                }
            })
            .collect()
    }
}

struct ModelOverview {
    name: String,
    #[allow(dead_code)]
    kind: String,
    status: String,
}
