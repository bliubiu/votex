use eframe::egui;

/// 字幕条目
#[derive(Debug, Clone)]
pub struct SubtitleEntry {
    pub index: usize,
    pub start_time: String,
    pub end_time: String,
    pub text: String,
}

/// 字幕可视化编辑器
pub struct SubtitleEditor;

impl SubtitleEditor {
    pub fn show(ui: &mut egui::Ui, entries: &mut Vec<SubtitleEntry>) {
        ui.vertical(|ui| {
            ui.heading("字幕编辑");
            ui.add_space(4.0);

            if entries.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(20.0);
                    ui.label(
                        egui::RichText::new("暂无字幕数据")
                            .color(egui::Color32::GRAY),
                    );
                });
                return;
            }

            // 工具栏
            ui.horizontal(|ui| {
                ui.label(format!("共 {} 条字幕", entries.len()));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("+ 添加").clicked() {
                        let last_end = entries.last().map(|e| e.end_time.clone()).unwrap_or_default();
                        entries.push(SubtitleEntry {
                            index: entries.len() + 1,
                            start_time: last_end.clone(),
                            end_time: last_end,
                            text: String::new(),
                        });
                    }
                });
            });

            ui.add_space(4.0);

            // 字幕列表
            egui::ScrollArea::vertical()
                .max_height(400.0)
                .show(ui, |ui| {
                    let mut remove_idx = None;
                    for (i, entry) in entries.iter_mut().enumerate() {
                        egui::Frame::group(ui.style())
                            .inner_margin(4.0)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(format!("#{}", i + 1))
                                            .strong()
                                            .color(egui::Color32::from_rgb(0, 120, 215)),
                                    );
                                    ui.add(
                                        egui::TextEdit::singleline(&mut entry.start_time)
                                            .desired_width(100.0)
                                            .hint_text("00:00:00,000"),
                                    );
                                    ui.label("→");
                                    ui.add(
                                        egui::TextEdit::singleline(&mut entry.end_time)
                                            .desired_width(100.0)
                                            .hint_text("00:00:05,000"),
                                    );
                                    if ui.small_button("✕").clicked() {
                                        remove_idx = Some(i);
                                    }
                                });
                                ui.add(
                                    egui::TextEdit::multiline(&mut entry.text)
                                        .desired_width(f32::INFINITY)
                                        .desired_rows(1),
                                );
                            });
                        ui.add_space(2.0);
                    }

                    if let Some(idx) = remove_idx {
                        entries.remove(idx);
                        // 重新编号
                        for (i, entry) in entries.iter_mut().enumerate() {
                            entry.index = i + 1;
                        }
                    }
                });
        });
    }
}
