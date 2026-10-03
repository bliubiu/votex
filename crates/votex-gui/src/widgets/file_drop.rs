use eframe::egui;
use std::path::PathBuf;

/// 文件拖拽组件
pub struct FileDrop;

impl FileDrop {
    /// 显示拖拽区域，返回拖入的文件路径
    pub fn show(ui: &mut egui::Ui, label: &str) -> Option<PathBuf> {
        let mut dropped_path = None;

        // 检测拖拽文件
        let dropped = ui.ctx().input(|i| i.raw.dropped_files.clone());
        if !dropped.is_empty() {
            if let Some(file) = dropped.first() {
                if let Some(path) = &file.path {
                    dropped_path = Some(path.clone());
                }
            }
        }

        // 拖拽区域
        let response = ui.vertical_centered(|ui| {
            ui.add_space(20.0);
            ui.label(
                egui::RichText::new(label)
                    .color(egui::Color32::GRAY),
            );
            ui.label(
                egui::RichText::new("拖拽文件到此处")
                    .small()
                    .color(egui::Color32::GRAY),
            );
            ui.add_space(20.0);
        }).response;

        // 悬停高亮
        if ui.ctx().input(|i| i.raw.hovered_files.len()) > 0 {
            ui.painter().rect_stroke(
                response.rect,
                4.0,
                egui::Stroke::new(2.0, egui::Color32::from_rgb(0, 120, 215)),
                egui::StrokeKind::Outside,
            );
        }

        dropped_path
    }
}
