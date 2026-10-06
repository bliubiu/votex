use std::sync::Arc;
use eframe::egui;

use crate::state::AppState;
use crate::task_runner;
use crate::pages::page_template::PageLayout;
use crate::theme::colors;

/// OCR 图文识别页面
pub struct OcrPage;

impl OcrPage {
    pub fn show(ui: &mut egui::Ui, state: &mut AppState) {
        PageLayout::header(ui, "图文识别", "图片/PDF 转文字，支持多语言识别");
        ui.add_space(4.0);

        // ======== 引擎与模式 ========
        PageLayout::card(ui, Some("识别设置"), |ui| {
            ui.horizontal(|ui| {
                ui.label("OCR 引擎:");
                egui::ComboBox::from_id_salt("ocr_engine")
                    .selected_text(match state.ocr.engine.as_str() {
                        "paddleocr" | "paddleocr-v6-medium" => "PaddleOCR v6 Medium（推荐）",
                        "paddleocr-v6-small" => "PaddleOCR v6 Small",
                        "paddleocr-v6-tiny" => "PaddleOCR v6 Tiny（轻量）",
                        "paddleocr-v4" => "PaddleOCR v4 Mobile",
                        "paddleocr-v5-mobile" => "PaddleOCR v5 Mobile",
                        "paddleocr-v5-server" => "PaddleOCR v5 Server",
                        "easyocr" => "EasyOCR（多语言）",
                        _ => &state.ocr.engine,
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.ocr.engine, "paddleocr-v6-medium".to_string(), "PaddleOCR v6 Medium（推荐）");
                        ui.selectable_value(&mut state.ocr.engine, "paddleocr-v6-small".to_string(), "PaddleOCR v6 Small");
                        ui.selectable_value(&mut state.ocr.engine, "paddleocr-v6-tiny".to_string(), "PaddleOCR v6 Tiny（轻量）");
                        ui.selectable_value(&mut state.ocr.engine, "paddleocr-v4".to_string(), "PaddleOCR v4 Mobile");
                        ui.selectable_value(&mut state.ocr.engine, "paddleocr-v5-mobile".to_string(), "PaddleOCR v5 Mobile");
                        ui.selectable_value(&mut state.ocr.engine, "paddleocr-v5-server".to_string(), "PaddleOCR v5 Server");
                        ui.selectable_value(&mut state.ocr.engine, "easyocr".to_string(), "EasyOCR（多语言）");
                    });
            });
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                ui.label("模式:");
                ui.selectable_value(&mut state.ocr.use_batch, false, "单图识别");
                ui.selectable_value(&mut state.ocr.use_batch, true, "批量识别");
            });
        });

        ui.add_space(12.0);

        // ======== 输入配置 ========
        PageLayout::card(ui, Some(if state.ocr.use_batch { "批量输入" } else { "输入配置" }), |ui| {
            if state.ocr.use_batch {
                Self::show_batch_mode(ui, state);
            } else {
                Self::show_single_mode(ui, state);
            }
        });

        ui.add_space(12.0);

        // ======== 操作与进度 ========
        PageLayout::card(ui, None, |ui| {
            if !state.ocr.use_batch {
                ui.horizontal(|ui| {
                    if !state.ocr.is_running {
                        if PageLayout::primary_button(ui, "开始识别", !state.ocr.input_path.is_empty()).clicked() {
                            if state.ocr.input_path.is_empty() {
                                state.ocr.result_message = "请选择输入图片".to_string();
                            } else {
                                state.ocr.result_message = String::new();
                                state.ocr.is_running = true;
                                state.ocr.progress = 0.0;
                                state.ocr.progress_text = "识别中...".to_string();

                                let tx = state.task_tx.as_ref().unwrap().clone();
                                let input_path = state.ocr.input_path.clone();
                                let output_path = state.ocr.output_path.clone();
                                let output_format = state.ocr.output_format.clone();
                                let no_cls = state.ocr.no_cls;
                                let engine = state.ocr.engine.clone();
                                let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));

                                // 保存取消令牌供「取消」按钮使用。
                                // 必须存进 state：此前令牌只存在于闭包内，取消按钮拿不到
                                // 同一个令牌，点了只是改界面文字，后台识别仍会跑完。
                                state.ocr.cancel_token = Some(cancel.clone());
                                task_runner::spawn_ocr(tx, input_path, output_path, output_format, no_cls, engine, cancel);
                            }
                        }
                    } else {
                        if PageLayout::danger_button(ui, "取消", true).clicked() {
                            // 置位取消令牌，后台任务在下一个检查点中断
                            if let Some(ref token) = state.ocr.cancel_token {
                                token.store(true, std::sync::atomic::Ordering::SeqCst);
                            }
                            // 不在此处置 is_running = false：后台线程仍在跑，
                            // 下一个 Progress 事件会把「正在取消...」覆盖回「识别中 3/10」，
                            // UI 闪回且用户误以为没取消。终态由后台 Success/Error 事件统一复位。
                            state.ocr.progress_text = "正在取消...".to_string();
                        }
                    }
                });
            } else {
                ui.horizontal(|ui| {
                    if !state.ocr.is_running {
                        if PageLayout::primary_button(ui, "开始批量识别", !state.ocr.batch_inputs.is_empty()).clicked() {
                            if state.ocr.batch_inputs.is_empty() {
                                state.ocr.result_message = "请添加要识别的图片".to_string();
                            } else {
                                state.ocr.result_message = String::new();
                                state.ocr.is_running = true;
                                state.ocr.progress = 0.0;
                                state.ocr.progress_text = format!("批量识别 0/{}", state.ocr.batch_inputs.len());

                                let tx = state.task_tx.as_ref().unwrap().clone();
                                let image_paths = state.ocr.batch_inputs.clone();
                                let output_dir = state.ocr.batch_output_dir.clone();
                                let output_format = state.ocr.output_format.clone();
                                let no_cls = state.ocr.no_cls;
                                let engine = state.ocr.engine.clone();
                                let max_concurrency = state.ocr.max_concurrency;
                                let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));

                                // 保存取消令牌供「取消」按钮使用。
                                // 必须存进 state：此前令牌只存在于闭包内，取消按钮拿不到
                                // 同一个令牌，点了只是改界面文字，后台识别仍会跑完。
                                state.ocr.cancel_token = Some(cancel.clone());
                                task_runner::spawn_ocr_batch(tx, image_paths, output_dir, output_format, no_cls, engine, max_concurrency, cancel);
                            }
                        }
                    } else {
                        if PageLayout::danger_button(ui, "取消", true).clicked() {
                            // 置位取消令牌，后台任务在下一个检查点中断
                            if let Some(ref token) = state.ocr.cancel_token {
                                token.store(true, std::sync::atomic::Ordering::SeqCst);
                            }
                            // 不在此处置 is_running = false：后台线程仍在跑，
                            // 下一个 Progress 事件会把「正在取消...」覆盖回「识别中 3/10」，
                            // UI 闪回且用户误以为没取消。终态由后台 Success/Error 事件统一复位。
                            state.ocr.progress_text = "正在取消...".to_string();
                        }
                    }
                });
            }

            if state.ocr.is_running {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(egui::RichText::new(&state.ocr.progress_text).size(12.0).color(colors::STONE_GRAY));
                });
                ui.add(
                    egui::ProgressBar::new(state.ocr.progress)
                        .fill(colors::WAVE_TEAL)
                        .animate(true)
                        .text(format!("{:.0}%", state.ocr.progress * 100.0)),
                );
            }

            if !state.ocr.result_message.is_empty() {
                ui.add_space(8.0);
                PageLayout::result_message(ui, &state.ocr.result_message);
            }
        });

        // 页面列表
        if !state.ocr.pages.is_empty() {
            ui.add_space(12.0);
            PageLayout::card(ui, Some("页面列表"), |ui| {
                let mut total_blocks = 0;
                for page in &state.ocr.pages {
                    total_blocks += page.block_count;
                    ui.horizontal(|ui| {
                        ui.label(format!("第 {} 页", page.index + 1));
                        PageLayout::status_label(ui, &page.status, &page.status);
                        if page.block_count > 0 {
                            ui.label(format!("{} 区块", page.block_count));
                        }
                        if page.confidence > 0.0 {
                            ui.label(format!("{:.1}%", page.confidence * 100.0));
                        }
                    });
                }
                if total_blocks > 0 {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(format!("总识别区块: {}", total_blocks)).strong());
                }
            });
        }
    }

    fn show_single_mode(ui: &mut egui::Ui, state: &mut AppState) {
        ui.horizontal(|ui| {
            ui.label("输入图片:");
            ui.text_edit_singleline(&mut state.ocr.input_path);
            if ui.button("浏览").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("图片", &["png", "jpg", "jpeg", "bmp", "tiff"])
                    .pick_file()
                {
                    state.ocr.input_path = path.display().to_string();
                }
            }
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("输出文件:");
            ui.text_edit_singleline(&mut state.ocr.output_path);
            if ui.button("浏览").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("文本", &["txt", "json", "md"])
                    .save_file()
                {
                    state.ocr.output_path = path.display().to_string();
                }
            }
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("输出格式:");
            egui::ComboBox::from_id_salt("ocr_format")
                .selected_text(&state.ocr.output_format)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.ocr.output_format, "txt".to_string(), "TXT");
                    ui.selectable_value(&mut state.ocr.output_format, "json".to_string(), "JSON");
                    ui.selectable_value(&mut state.ocr.output_format, "md".to_string(), "Markdown");
                });
            ui.checkbox(&mut state.ocr.no_cls, "禁用方向分类");
        });
    }

    fn show_batch_mode(ui: &mut egui::Ui, state: &mut AppState) {
        ui.horizontal(|ui| {
            if ui.button("添加图片").clicked() {
                if let Some(files) = rfd::FileDialog::new()
                    .add_filter("图片", &["png", "jpg", "jpeg", "bmp", "tiff"])
                    .pick_files()
                {
                    for path in &files {
                        state.ocr.batch_inputs.push(path.display().to_string());
                    }
                }
            }
        });

        ui.add_space(4.0);

        let mut remove_idx = None;
        for (i, path) in state.ocr.batch_inputs.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(format!("{}. {}", i + 1, path));
                if ui.button("×").clicked() {
                    remove_idx = Some(i);
                }
            });
        }
        if let Some(idx) = remove_idx {
            state.ocr.batch_inputs.remove(idx);
        }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("输出目录:");
            ui.text_edit_singleline(&mut state.ocr.batch_output_dir);
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("输出格式:");
            egui::ComboBox::from_id_salt("batch_ocr_format")
                .selected_text(&state.ocr.output_format)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.ocr.output_format, "txt".to_string(), "TXT");
                    ui.selectable_value(&mut state.ocr.output_format, "json".to_string(), "JSON");
                    ui.selectable_value(&mut state.ocr.output_format, "md".to_string(), "Markdown");
                });

            ui.label("并发数:");
            ui.add(egui::Slider::new(&mut state.ocr.max_concurrency, 1..=8).integer());
            ui.checkbox(&mut state.ocr.no_cls, "禁用方向分类");
        });
    }
}
