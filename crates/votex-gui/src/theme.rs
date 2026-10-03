use eframe::egui;

/// 声阅主题色系 —「墨韵声波」(Ink & Wave)
pub mod colors {
    use egui::Color32;

    pub const INK: Color32 = Color32::from_rgb(28, 27, 34);
    pub const RICE_PAPER: Color32 = Color32::from_rgb(244, 241, 234);
    pub const WAVE_TEAL: Color32 = Color32::from_rgb(45, 212, 191);
    pub const BAMBOO_GOLD: Color32 = Color32::from_rgb(196, 168, 130);
    pub const STONE_GRAY: Color32 = Color32::from_rgb(107, 114, 128);
    pub const SEAL_RED: Color32 = Color32::from_rgb(239, 68, 68);
    pub const JADE_GREEN: Color32 = Color32::from_rgb(52, 211, 153);
    pub const CARD_BG: Color32 = Color32::from_rgb(255, 255, 255);
    pub const HOVER_BG: Color32 = Color32::from_rgb(241, 245, 249);
    pub const SELECTED_BG: Color32 = Color32::from_rgb(236, 252, 249);
    pub const CHARCOAL: Color32 = Color32::from_rgb(55, 54, 64);
    pub const NAV_TEXT: Color32 = Color32::from_rgb(180, 180, 190);
    pub const NAV_ACTIVE_TEXT: Color32 = Color32::from_rgb(255, 255, 255);
    pub const NAV_GROUP_LABEL: Color32 = Color32::from_rgb(120, 120, 130);
}

/// 应用全局样式
pub fn configure_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    style.visuals.widgets.noninteractive.bg_fill = colors::RICE_PAPER;
    style.visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, colors::STONE_GRAY);
    style.visuals.widgets.noninteractive.corner_radius = egui::CornerRadius::same(8);
    style.visuals.widgets.inactive.bg_fill = colors::CARD_BG;
    style.visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(8);
    style.visuals.widgets.hovered.bg_fill = colors::HOVER_BG;
    style.visuals.widgets.hovered.corner_radius = egui::CornerRadius::same(8);
    style.visuals.widgets.active.bg_fill = colors::SELECTED_BG;
    style.visuals.widgets.active.corner_radius = egui::CornerRadius::same(8);
    style.visuals.selection.bg_fill = colors::WAVE_TEAL;

    style.spacing.item_spacing = egui::vec2(10.0, 8.0);

    ctx.set_style(style);
}

/// 绘制导航选中态左侧声波条
pub fn draw_nav_active_bar(ui: &egui::Ui, rect: &egui::Rect) {
    let painter = ui.painter();
    let bar_rect = egui::Rect::from_min_size(
        egui::pos2(rect.left(), rect.top() + 4.0),
        egui::vec2(3.0, rect.height() - 8.0),
    );
    painter.rect_filled(bar_rect, egui::CornerRadius::same(2), colors::WAVE_TEAL);
}

/// EQ 风格进度条
pub fn eq_progress_bar(ui: &mut egui::Ui, progress: f32, text: &str) -> egui::Response {
    let height = 6.0;
    let desired = ui.available_size();
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(desired.x.max(100.0), height),
        egui::Sense::hover(),
    );

    if ui.is_rect_visible(rect) {
        ui.painter().rect_filled(rect, egui::CornerRadius::same(3), egui::Color32::from_rgb(229, 231, 235));

        let bar_count = 20;
        let gap = 2.0;
        let bar_width = (rect.width() - gap * (bar_count - 1) as f32) / bar_count as f32;
        let filled_count = (progress * bar_count as f32).ceil() as usize;

        for i in 0..bar_count {
            let x = rect.left() + i as f32 * (bar_width + gap);
            let bar_height = if i < filled_count {
                let h_ratio = 0.3 + 0.7 * ((i as f32 / bar_count as f32) * std::f32::consts::PI).sin();
                (rect.height() * h_ratio).max(2.0)
            } else {
                2.0
            };
            let bar_rect = egui::Rect::from_min_size(
                egui::pos2(x, rect.bottom() - bar_height),
                egui::vec2(bar_width, bar_height),
            );
            let color = if i < filled_count {
                colors::WAVE_TEAL
            } else {
                egui::Color32::from_rgb(229, 231, 235)
            };
            ui.painter().rect_filled(bar_rect, egui::CornerRadius::same(1), color);
        }

        if !text.is_empty() {
            let text_pos = egui::pos2(rect.center().x, rect.center().y - 8.0);
            ui.painter().text(
                text_pos,
                egui::Align2::CENTER_CENTER,
                text,
                egui::FontId::proportional(11.0),
                colors::STONE_GRAY,
            );
        }
    }

    response
}
