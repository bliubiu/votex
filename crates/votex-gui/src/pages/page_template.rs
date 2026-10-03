use eframe::egui;
use crate::theme::colors;

/// 页面布局模板 — 统一的卡片式三段布局
pub struct PageLayout;

impl PageLayout {
    pub fn header(ui: &mut egui::Ui, title: &str, description: &str) {
        ui.horizontal(|ui| {
            ui.heading(egui::RichText::new(title).size(20.0).color(colors::INK));
            if !description.is_empty() {
                ui.add_space(12.0);
                ui.label(egui::RichText::new(description).size(13.0).color(colors::STONE_GRAY));
            }
        });
        ui.add_space(4.0);
        let rect = ui.available_rect_before_wrap();
        ui.painter().line_segment(
            [
                egui::pos2(rect.left(), rect.top()),
                egui::pos2(rect.right(), rect.top()),
            ],
            egui::Stroke::new(1.0, colors::STONE_GRAY.linear_multiply(0.15)),
        );
        ui.add_space(12.0);
    }

    pub fn card(ui: &mut egui::Ui, title: Option<&str>, add_contents: impl FnOnce(&mut egui::Ui)) {
        egui::Frame::new()
            .fill(colors::CARD_BG)
            .corner_radius(egui::CornerRadius::same(10))
            .stroke(egui::Stroke::new(1.0, colors::STONE_GRAY.linear_multiply(0.15)))
            .inner_margin(egui::Margin::symmetric(16, 14))
            .show(ui, |ui| {
                if let Some(title) = title {
                    ui.label(egui::RichText::new(title).size(14.0).strong().color(colors::INK));
                    ui.add_space(10.0);
                }
                add_contents(ui);
            });
    }

    pub fn primary_button(ui: &mut egui::Ui, text: &str, enabled: bool) -> egui::Response {
        let btn = egui::Button::new(
            egui::RichText::new(text).size(14.0).color(egui::Color32::WHITE)
        )
        .fill(colors::WAVE_TEAL)
        .corner_radius(egui::CornerRadius::same(6))
        .min_size(egui::vec2(110.0, 34.0));
        ui.add_enabled(enabled, btn)
    }

    pub fn secondary_button(ui: &mut egui::Ui, text: &str) -> egui::Response {
        let btn = egui::Button::new(
            egui::RichText::new(text).size(14.0).color(colors::INK)
        )
        .fill(egui::Color32::TRANSPARENT)
        .stroke(egui::Stroke::new(1.0, colors::STONE_GRAY.linear_multiply(0.3)))
        .corner_radius(egui::CornerRadius::same(6))
        .min_size(egui::vec2(90.0, 34.0));
        ui.add(btn)
    }

    pub fn danger_button(ui: &mut egui::Ui, text: &str, enabled: bool) -> egui::Response {
        let btn = egui::Button::new(
            egui::RichText::new(text).size(14.0).color(egui::Color32::WHITE)
        )
        .fill(colors::SEAL_RED)
        .corner_radius(egui::CornerRadius::same(6))
        .min_size(egui::vec2(90.0, 34.0));
        ui.add_enabled(enabled, btn)
    }

    pub fn status_label(ui: &mut egui::Ui, text: &str, status: &str) {
        let color = match status {
            "就绪" | "完成" | "成功" => colors::JADE_GREEN,
            "执行中" | "运行" => colors::WAVE_TEAL,
            "失败" | "错误" => colors::SEAL_RED,
            "等待" | "待执行" => colors::STONE_GRAY,
            "下载" => colors::BAMBOO_GOLD,
            _ => colors::STONE_GRAY,
        };
        ui.colored_label(color, text);
    }

    pub fn result_message(ui: &mut egui::Ui, message: &str) {
        if message.is_empty() {
            return;
        }
        let is_success = message.contains("成功") || message.contains("完成");
        let bg = if is_success {
            colors::JADE_GREEN.linear_multiply(0.1)
        } else {
            colors::SEAL_RED.linear_multiply(0.1)
        };
        let fg = if is_success { colors::JADE_GREEN } else { colors::SEAL_RED };

        egui::Frame::new()
            .fill(bg)
            .corner_radius(egui::CornerRadius::same(6))
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                ui.colored_label(fg, message);
            });
    }

    pub fn text_editor(ui: &mut egui::Ui, text: &mut String, hint: &str, max_height: f32) {
        egui::ScrollArea::vertical()
            .max_height(max_height)
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(text)
                        .hint_text(hint)
                        .desired_width(f32::INFINITY)
                        .desired_rows(6)
                        .frame(true),
                );
            });
    }

    pub fn param_grid(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
        egui::Grid::new(ui.next_auto_id())
            .min_col_width(100.0)
            .spacing([12.0, 8.0])
            .show(ui, add_contents);
    }
}
