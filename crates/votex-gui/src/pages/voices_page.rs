use eframe::egui;

use crate::state::AppState;
use crate::pages::page_template::PageLayout;
use crate::theme::colors;
use crate::widgets::audio_player::AudioPlayer;

use votex_app::platform::tts as voice_lib;

/// 音色库页面：克隆音色的入库 / 试听 / 删除
///
/// 音色入库后对 IndexTTS-2.5 与 CosyVoice3 两个克隆引擎同时可用
/// （IndexTTS-2.5 按名回落到统一音色库目录）。
/// 列表/添加/删除为毫秒级文件操作，在 UI 线程同步执行；
/// 删除走二次确认（AGENTS.md：禁止自动删除音色文件）。
pub struct VoicesPage;

impl VoicesPage {
    pub fn show(ui: &mut egui::Ui, state: &mut AppState) {
        // 进入页面时刷新列表（以 operation 后的显式刷新为准，这里做兜底）
        if state.voices.voices.is_empty() {
            state.voices.voices = voice_lib::list_voices();
        }

        PageLayout::header(
            ui,
            "音色库",
            "克隆音色管理：入库后 IndexTTS-2.5 / CosyVoice3 均可直接使用",
        );
        ui.add_space(4.0);

        // ======== 添加音色 ========
        PageLayout::card(ui, Some("添加克隆音色"), |ui| {
            PageLayout::param_grid(ui, |ui| {
                ui.label("音色名:");
                ui.add(egui::TextEdit::singleline(&mut state.voices.new_name).desired_width(220.0));
                ui.end_row();

                ui.label("参考音频:");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut state.voices.new_reference)
                            .desired_width(300.0),
                    );
                    if ui.button("浏览").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("音频", &["wav"])
                            .pick_file()
                        {
                            state.voices.new_reference = path.display().to_string();
                            // 未命名时用文件名兜底，减少一步输入
                            if state.voices.new_name.is_empty() {
                                if let Some(stem) = path.file_stem() {
                                    state.voices.new_name = stem.to_string_lossy().to_string();
                                }
                            }
                        }
                    }
                });
                ui.end_row();

                ui.label("转写文本:");
                ui.add(
                    egui::TextEdit::singleline(&mut state.voices.new_transcript)
                        .hint_text("参考音频说的话（CosyVoice 必需，IndexTTS-2.5 可省略）")
                        .desired_width(300.0),
                );
                ui.end_row();

                ui.label("降噪:");
                ui.checkbox(&mut state.voices.new_denoise, "入库时自动降噪");
                ui.end_row();

                ui.label("时长上限:");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut state.voices.new_max_ref_seconds)
                            .range(0..=600)
                            .suffix("s"),
                    );
                    ui.label(
                        egui::RichText::new("超限时无转写自动截取（best_window），有转写拒绝")
                            .size(11.0)
                            .color(colors::STONE_GRAY),
                    );
                });
                ui.end_row();
            });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let can_add = !state.voices.new_name.trim().is_empty()
                    && !state.voices.new_reference.trim().is_empty();
                if PageLayout::primary_button(ui, "＋ 入库音色", can_add).clicked() {
                    match voice_lib::add_voice(
                        state.voices.new_name.trim(),
                        std::path::Path::new(&state.voices.new_reference),
                        state.voices.new_denoise,
                        Some(state.voices.new_transcript.trim()).filter(|t| !t.is_empty()),
                        (state.voices.new_max_ref_seconds > 0)
                            .then_some(state.voices.new_max_ref_seconds),
                    ) {
                        Ok(meta) => {
                            let trimmed_note = meta
                                .trimmed_from_ms
                                .map(|from| format!("（已从 {}ms 截取）", from))
                                .unwrap_or_default();
                            state.voices.message =
                                Some(format!("音色「{}」入库成功{}", meta.name, trimmed_note));
                            state.voices.new_name.clear();
                            state.voices.new_reference.clear();
                            state.voices.new_transcript.clear();
                        }
                        Err(e) => {
                            state.voices.message = Some(format!("入库失败: {}", e));
                        }
                    }
                    state.voices.voices = voice_lib::list_voices();
                }
            });
        });

        ui.add_space(12.0);

        // ======== 音色列表 ========
        PageLayout::card(ui, Some("音色列表"), |ui| {
            if state.voices.voices.is_empty() {
                ui.label(
                    egui::RichText::new("音色库为空：录制一段 5~15 秒的清晰人声 WAV，在上方入库即可克隆")
                        .color(colors::STONE_GRAY),
                );
                return;
            }

            egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                for voice in &state.voices.voices {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(&voice.name)
                                .strong()
                                .color(colors::INK),
                        );
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!(
                                "{}ms · {}Hz{}",
                                voice.duration_ms,
                                voice.sample_rate,
                                if voice.denoised { " · 已降噪" } else { "" }
                            ))
                            .size(12.0)
                            .color(colors::STONE_GRAY),
                        );
                        match &voice.transcript {
                            Some(_) => {
                                ui.colored_label(colors::JADE_GREEN, "转写: 有");
                            }
                            None => {
                                ui.colored_label(colors::BAMBOO_GOLD, "转写: 无（CosyVoice 不可用）");
                            }
                        }

                        // 试听
                        let preview_path = voice_lib::voice_reference_path(&voice.name)
                            .map(|p| p.display().to_string())
                            .unwrap_or_default();
                        if ui
                            .add_enabled(
                                !preview_path.is_empty(),
                                egui::Button::new(egui::RichText::new("▶ 试听").size(12.0)),
                            )
                            .clicked()
                        {
                            state.audio_player.open_file(&preview_path);
                            state.voices.previewing = Some(voice.name.clone());
                        }

                        // 删除（先挂待确认，再点一次才真正删除）
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new("删除").size(12.0).color(colors::SEAL_RED),
                                ),
                            )
                            .clicked()
                        {
                            state.voices.pending_remove = Some(voice.name.clone());
                        }
                    });
                }
            });
        });

        // ======== 删除二次确认 ========
        if let Some(name) = state.voices.pending_remove.clone() {
            egui::Window::new("确认删除音色")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .fixed_size([380.0, 150.0])
                .show(ui.ctx(), |ui| {
                    ui.label(format!("确定删除音色「{}」吗？", name));
                    ui.label(
                        egui::RichText::new("将同时删除其参考音频与元数据，此操作不可撤销。")
                            .size(12.0)
                            .color(colors::SEAL_RED),
                    );
                    ui.add_space(12.0);
                    ui.horizontal(|ui| {
                        if PageLayout::danger_button(ui, "确认删除", true).clicked() {
                            state.voices.message = Some(match voice_lib::remove_voice(&name, true) {
                                Ok(true) => format!("已删除音色「{}」", name),
                                Ok(false) => format!("音色「{}」不存在", name),
                                Err(e) => format!("删除失败: {}", e),
                            });
                            state.voices.pending_remove = None;
                            state.voices.voices = voice_lib::list_voices();
                        }
                        if PageLayout::secondary_button(ui, "取消").clicked() {
                            state.voices.pending_remove = None;
                        }
                    });
                });
        }

        // ======== 操作消息 ========
        if let Some(msg) = &state.voices.message {
            ui.add_space(8.0);
            PageLayout::result_message(ui, msg);
        }

        // ======== 试听播放器 ========
        if !state.audio_player.file_path.is_empty() {
            ui.add_space(12.0);
            AudioPlayer::show(ui, &mut state.audio_player);
            if let Some(name) = &state.voices.previewing {
                ui.label(
                    egui::RichText::new(format!("正在试听: {}", name))
                        .size(11.0)
                        .color(colors::STONE_GRAY),
                );
            }
        }
    }
}
