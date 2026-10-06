use eframe::egui;

use crate::state::AppState;
use crate::task_runner;

/// 首次启动引导页面
pub struct OnboardingPage;

impl OnboardingPage {
    pub fn show(ctx: &egui::Context, state: &mut AppState) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(40.0);
                ui.heading("欢迎使用声阅");
                ui.label("离线桌面语音工具 · 首次使用引导");
                ui.add_space(24.0);

                // 步骤指示器
                let step = state.onboarding.step;
                let total = state.onboarding.total_steps;

                ui.horizontal(|ui| {
                    for i in 0..total {
                        let is_current = i == step;
                        let is_done = i < step;
                        let label = if is_done { "✓" } else if is_current { "●" } else { "○" };
                        let color = if is_done {
                            egui::Color32::GREEN
                        } else if is_current {
                            egui::Color32::from_rgb(0, 120, 215)
                        } else {
                            egui::Color32::GRAY
                        };
                        ui.label(egui::RichText::new(label).size(20.0).color(color));
                        if i < total - 1 {
                            ui.label("───");
                        }
                    }
                });

                ui.add_space(8.0);
                let step_titles = ["选择模型", "下载模型", "开始使用"];
                ui.label(egui::RichText::new(step_titles[step]).strong());
                ui.add_space(24.0);

                // 步骤内容
                match step {
                    0 => Self::step_select_models(ui, state),
                    1 => Self::step_download(ui, state),
                    2 => Self::step_finish(ui, state),
                    _ => {}
                }

                ui.add_space(24.0);

                // 导航按钮
                ui.horizontal(|ui| {
                    if step > 0 {
                        if ui.button("← 上一步").clicked() {
                            state.onboarding.step = step - 1;
                        }
                    }
                    if step < total - 1 {
                        if ui.button("下一步 →").clicked() {
                            if step == 0 {
                                // 进入下载步骤
                                state.onboarding.step = 1;
                                state.onboarding.downloading = true;
                                Self::start_downloads(state);
                            } else {
                                state.onboarding.step = step + 1;
                            }
                        }
                    }
                    if step == total - 1 {
                        if ui.button("开始使用").clicked() {
                            state.show_onboarding = false;
                        }
                    }
                    if ui.button("跳过引导").clicked() {
                        state.show_onboarding = false;
                    }
                });
            });
        });
    }

    fn start_downloads(state: &mut AppState) {
        let models_to_download: Vec<String> = {
            let mut list = Vec::new();
            if state.onboarding.selected_kokoro {
                list.push("kokoro-82m".to_string());
            }
            if state.onboarding.selected_whisper_base {
                list.push("whisper-base".to_string());
            }
            if state.onboarding.selected_whisper_small {
                list.push("whisper-small".to_string());
            }
            list
        };

        if models_to_download.is_empty() {
            state.onboarding.downloading = false;
            return;
        }

        // 为每个选择的模型启动下载
        for model_id in models_to_download {
            let tx = state.task_tx.as_ref().unwrap().clone();
            let mirror = "cn".to_string();
            let models_dir = votex_app::platform::paths::models_dir()
                .display()
                .to_string();

            task_runner::spawn_model_download(tx, model_id, mirror, models_dir, None);
        }
    }

    fn step_select_models(ui: &mut egui::Ui, state: &mut AppState) {
        ui.label("选择需要下载的模型：");
        ui.add_space(8.0);

        ui.checkbox(&mut state.onboarding.selected_kokoro, "Kokoro-82M — TTS 语音合成（推荐）");
        ui.checkbox(&mut state.onboarding.selected_whisper_base, "Whisper Base — ASR 语音识别（推荐）");
        ui.checkbox(&mut state.onboarding.selected_whisper_small, "Whisper Small — ASR 高精度识别（可选）");

        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("推荐模型约 200MB，下载时间取决于网络速度")
                .small()
                .color(egui::Color32::GRAY),
        );
    }

    fn step_download(ui: &mut egui::Ui, state: &mut AppState) {
        if state.onboarding.downloading {
            ui.label("正在下载模型...");
            ui.add_space(8.0);
            ui.add(
                egui::ProgressBar::new(state.onboarding.download_progress)
                    .show_percentage(),
            );
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(format!("当前: {}", state.onboarding.download_model))
                    .small(),
            );
        } else {
            ui.label("点击「下一步」开始下载所选模型");
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("将使用国内镜像源加速下载")
                    .small()
                    .color(egui::Color32::GRAY),
            );
        }
    }

    fn step_finish(ui: &mut egui::Ui, _state: &mut AppState) {
        ui.label("设置完成！你可以开始使用声阅了。");
        ui.add_space(12.0);
        ui.label("• 在「语音合成」页面输入文本生成语音");
        ui.label("• 在「语音识别」页面导入音频生成字幕");
        ui.label("• 在「系统设置」中管理模型和配置");
    }
}
