use eframe::egui;
use std::path::Path;

use crate::model_detector;
use crate::state::AppState;
use crate::task_runner;
use crate::pages::page_template::PageLayout;
use crate::theme::colors;

/// 系统设置页面
pub struct SettingsPage;

impl SettingsPage {
    pub fn show(ui: &mut egui::Ui, state: &mut AppState) {
        PageLayout::header(ui, "系统设置", "管理模型、API Key 和系统配置");
        ui.add_space(4.0);

        // ======== 基础配置 ========
        PageLayout::card(ui, Some("基础配置"), |ui| {
            PageLayout::param_grid(ui, |ui| {
                ui.label("模型目录:");
                ui.text_edit_singleline(&mut state.settings.models_dir);
                ui.end_row();

                ui.label("日志级别:");
                egui::ComboBox::from_id_salt("log_level")
                    .selected_text(&state.settings.log_level)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.settings.log_level, "debug".to_string(), "DEBUG");
                        ui.selectable_value(&mut state.settings.log_level, "info".to_string(), "INFO");
                        ui.selectable_value(&mut state.settings.log_level, "error".to_string(), "ERROR");
                    });
                ui.end_row();

                ui.label("下载镜像:");
                egui::ComboBox::from_id_salt("mirror")
                    .selected_text(&state.settings.mirror)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.settings.mirror, "cn".to_string(), "国内优先（hf-mirror/ModelScope）");
                        ui.selectable_value(&mut state.settings.mirror, "default".to_string(), "默认（HuggingFace）");
                    });
                ui.end_row();
            });

            ui.add_space(8.0);
            if ui.button("保存配置").clicked() {
                Self::save_config(state);
            }
        });

        ui.add_space(12.0);

        // ======== 推理资源限制 ========
        PageLayout::card(ui, Some("推理资源限制"), |ui| {
            ui.label(egui::RichText::new("限制 ONNX Runtime 推理时的 CPU 和内存使用，适用于需要腾出资源给其他应用的场景。修改后点击「保存配置」生效。")
                .size(12.0).color(colors::STONE_GRAY));
            ui.add_space(8.0);

            PageLayout::param_grid(ui, |ui| {
                // CPU 线程数
                ui.label("CPU 线程数:");
                ui.horizontal(|ui| {
                    let max_cores = std::thread::available_parallelism()
                        .map(|n| n.get() as f64)
                        .unwrap_or(16.0);
                    let mut slider_val = state.settings.cpu_threads as f64;
                    ui.add(
                        egui::Slider::new(&mut slider_val, 0.0..=max_cores)
                            .step_by(1.0)
                            .integer()
                            .show_value(false)
                            .suffix(" 线程")
                    );
                    state.settings.cpu_threads = slider_val as u32;
                    let text = if state.settings.cpu_threads == 0 {
                        "自动（全部核心）".to_string()
                    } else {
                        format!("{} 线程", state.settings.cpu_threads)
                    };
                    ui.label(text);
                });
                ui.end_row();

                // 内存限制
                ui.label("内存上限:");
                ui.horizontal(|ui| {
                    let mem_options = [0u32, 512, 1024, 2048, 4096, 8192];
                    let labels = ["不限", "512 MB", "1 GB", "2 GB", "4 GB", "8 GB"];
                    egui::ComboBox::from_id_salt("memory_limit")
                        .selected_text(if state.settings.memory_limit_mb == 0 {
                            "不限".to_string()
                        } else {
                            format!("{} MB（{} GB）",
                                state.settings.memory_limit_mb,
                                state.settings.memory_limit_mb / 1024)
                        })
                        .show_ui(ui, |ui| {
                            for (i, &val) in mem_options.iter().enumerate() {
                                let selected = state.settings.memory_limit_mb == val;
                                if ui.selectable_label(selected, labels[i]).clicked() {
                                    state.settings.memory_limit_mb = val;
                                }
                            }
                        });
                    // 允许手动输入
                    let mut custom = state.settings.memory_limit_mb as f64;
                    ui.add(
                        egui::Slider::new(&mut custom, 0.0..=16384.0)
                            .step_by(256.0)
                            .integer()
                            .show_value(false)
                    );
                    state.settings.memory_limit_mb = custom as u32;
                });
                ui.end_row();

                // 执行模式
                ui.label("执行模式:");
                egui::ComboBox::from_id_salt("execution_mode")
                    .selected_text(match state.settings.execution_mode.as_str() {
                        "sequential" => "顺序执行（省内存）",
                        "parallel" => "并行执行（高性能）",
                        _ => &state.settings.execution_mode,
                    })
                    .show_ui(ui, |ui| {
                        if ui.selectable_value(&mut state.settings.execution_mode, "sequential".to_string(), "顺序执行（省内存）").changed() {}
                        if ui.selectable_value(&mut state.settings.execution_mode, "parallel".to_string(), "并行执行（高性能，高内存）").changed() {}
                    });
                ui.end_row();
            });

            ui.add_space(4.0);
            // 资源建议提示
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("💡").size(14.0));
                ui.label(egui::RichText::new(
                    "建议：CPU 线程设为 2~4 可显著降低 CPU 占用，内存设为 2GB 可防止 OOM。Qwen3-TTS 建议 4 线程 + 4GB 内存。"
                ).size(11.0).color(colors::STONE_GRAY));
            });
        });

        ui.add_space(12.0);

        // ======== API Key 配置 ========
        PageLayout::card(ui, Some("API Key 配置"), |ui| {
            ui.label(egui::RichText::new("配置在线服务 API Key 以启用 LLM 脚本生成、在线 TTS/ASR 等高级功能。")
                .size(12.0).color(colors::STONE_GRAY));
            ui.add_space(8.0);

            PageLayout::param_grid(ui, |ui| {
                ui.label("DeepSeek API Key:");
                ui.horizontal(|ui| {
                    let display_key = Self::masked_api_key("DEEPSEEK_API_KEY");
                    ui.label(display_key);
                    if ui.small_button("设置").clicked() {}
                    if ui.small_button("清除").clicked() {}
                });
                ui.end_row();

                ui.label("Azure Speech Key:");
                ui.horizontal(|ui| {
                    let display_key = Self::masked_api_key("AZURE_SPEECH_KEY");
                    ui.label(display_key);
                    if ui.small_button("设置").clicked() {}
                    if ui.small_button("清除").clicked() {}
                });
                ui.end_row();

                ui.label("阿里云 AK ID:");
                ui.horizontal(|ui| {
                    let display_key = Self::masked_api_key("ALIYUN_ACCESS_KEY_ID");
                    ui.label(display_key);
                    if ui.small_button("设置").clicked() {}
                    if ui.small_button("清除").clicked() {}
                });
                ui.end_row();

                ui.label("Pexels API Key:");
                ui.horizontal(|ui| {
                    let display_key = Self::masked_api_key("PEXELS_API_KEY");
                    ui.label(display_key);
                    if ui.small_button("设置").clicked() {}
                    if ui.small_button("清除").clicked() {}
                });
                ui.end_row();
            });

            ui.add_space(4.0);
            ui.label(egui::RichText::new("API Key 通过环境变量读取，设置后重启应用生效。加密存储功能即将推出。")
                .size(11.0).color(colors::STONE_GRAY));
        });

        ui.add_space(12.0);

        // ======== 模型管理 ========
        PageLayout::card(ui, Some("模型管理"), |ui| {
            if state.settings.model_list.is_empty() && state.initialized {
                Self::refresh_models(state);
            }

            ui.horizontal(|ui| {
                if ui.button("🔄 刷新模型列表").clicked() {
                    Self::refresh_models(state);
                }
            });

            ui.add_space(8.0);

            // 分类筛选标签
            let filters = ["全部", "TTS", "ASR", "OCR", "Translation"];
            ui.horizontal(|ui| {
                for &f in &filters {
                    let selected = state.settings.model_filter == f;
                    let btn = if selected {
                        egui::Button::new(egui::RichText::new(f).strong().color(colors::WAVE_TEAL))
                            .fill(colors::WAVE_TEAL.linear_multiply(0.1))
                    } else {
                        egui::Button::new(f)
                    };
                    if ui.add(btn).clicked() {
                        state.settings.model_filter = f.to_string();
                    }
                }
            });

            ui.add_space(8.0);

            let filter = state.settings.model_filter.to_lowercase();
            let filtered: Vec<usize> = state.settings.model_list.iter().enumerate()
                .filter(|(_, m)| filter == "全部" || m.kind.to_lowercase() == filter)
                .map(|(i, _)| i)
                .collect();
            let filtered_count = filtered.len();
            let total_count = state.settings.model_list.len();

            ui.label(egui::RichText::new(format!("共 {} 个模型（筛选后 {} 个）", total_count, filtered_count))
                .size(12.0).color(colors::STONE_GRAY));

            ui.add_space(4.0);

            if state.settings.model_list.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(12.0);
                    ui.label(egui::RichText::new("暂无模型数据，点击刷新").size(13.0).color(colors::STONE_GRAY));
                    ui.add_space(12.0);
                });
            } else {
                egui::ScrollArea::vertical()
                    .max_height(400.0)
                    .show(ui, |ui| {
                        egui::Grid::new("model_list")
                            .min_col_width(100.0)
                            .spacing([8.0, 6.0])
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new("模型").strong());
                                ui.label(egui::RichText::new("类型").strong());
                                ui.label(egui::RichText::new("引擎").strong());
                                ui.label(egui::RichText::new("状态").strong());
                                ui.label(egui::RichText::new("操作").strong());
                                ui.end_row();

                                for &idx in &filtered {
                                    let model = &mut state.settings.model_list[idx];
                                    ui.label(&model.name);
                                    ui.label(&model.kind);
                                    ui.label(&model.engine);
                                    PageLayout::status_label(ui, &model.status, &model.status);

                                    ui.horizontal(|ui| {
                                        if model.status == "未下载" {
                                            if ui.small_button("下载").clicked() {
                                                model.status = "下载中...".to_string();
                                                let tx = state.task_tx.as_ref().unwrap().clone();
                                                let model_id = model.id.clone();
                                                let mirror = state.settings.mirror.clone();
                                                let models_dir = state.settings.models_dir.clone();
                                                task_runner::spawn_model_download(tx, model_id, mirror, models_dir, state.download_repo.clone());
                                            }
                                        } else if model.status.contains("就绪") {
                                            if ui.small_button("删除").clicked() {
                                                let models_dir = if state.settings.models_dir.is_empty() {
                                                    model_detector::default_models_dir()
                                                } else {
                                                    Path::new(&state.settings.models_dir).to_path_buf()
                                                };
                                                let model_dir = models_dir.join(&model.id);
                                                if model_dir.exists() {
                                                    let _ = std::fs::remove_dir_all(&model_dir);
                                                }
                                                model.status = "未下载".to_string();
                                            }
                                        }
                                    });
                                    ui.end_row();
                                }
                            });
                    });
            }
        });
    }

    fn save_config(state: &AppState) {
        let config_path = Path::new("application.yml");
        let priority = if state.settings.mirror == "cn" {
            vec!["modelscope".into(), "hf-mirror".into(), "gitee".into()]
        } else {
            vec!["modelscope".into(), "hf-mirror".into(), "github".into(), "huggingface".into()]
        };

        let infer_cpu_threads = state.settings.cpu_threads;
        let infer_memory_limit = state.settings.memory_limit_mb;
        let infer_exec_mode = state.settings.execution_mode.clone();

        // 同时更新全局运行时推理配置
        votex_infra::shared::OrtSessionFactory::update_global_config(|cfg| {
            cfg.num_threads = infer_cpu_threads;
            cfg.inter_threads = 0;
            cfg.memory_limit_mb = infer_memory_limit;
            cfg.execution_mode = infer_exec_mode.clone();
            cfg.enable_memory_pattern = infer_memory_limit == 0;
        });

        let config = votex_domain::config::value_object::AppConfig {
            models: votex_domain::config::value_object::ModelsConfig {
                storage_path: if state.settings.models_dir.is_empty() {
                    "models".to_string()
                } else {
                    state.settings.models_dir.clone()
                },
                registry_path: None,
                download: votex_domain::config::value_object::DownloadConfig {
                    priority,
                    ..Default::default()
                },
            },
            log: votex_domain::config::value_object::LogConfig {
                level: state.settings.log_level.clone(),
                ..Default::default()
            },
            inference: votex_domain::config::value_object::InferenceConfig {
                num_threads: infer_cpu_threads,
                inter_threads: 0,
                quantization: "none".to_string(),
                kv_cache: true,
                execution_provider: "auto".to_string(),
                memory_limit_mb: infer_memory_limit,
                enable_memory_pattern: infer_memory_limit == 0,
                execution_mode: infer_exec_mode,
            },
            ..Default::default()
        };

        match votex_infra::config::loader::ConfigLoader::save_to_file(&config, config_path) {
            Ok(()) => {
                tracing::info!("配置已保存，推理资源限制已更新");
            }
            Err(e) => {
                tracing::error!("保存配置失败: {}", e);
            }
        }
    }

    fn masked_api_key(env_var: &str) -> String {
        match std::env::var(env_var) {
            Ok(key) if !key.is_empty() => {
                if key.len() > 8 {
                    format!("{}****", &key[..4])
                } else {
                    "****".to_string()
                }
            }
            _ => "未配置".to_string(),
        }
    }

    fn refresh_models(state: &mut AppState) {
        let models_dir = if state.settings.models_dir.is_empty() {
            model_detector::default_models_dir()
        } else {
            std::path::PathBuf::from(&state.settings.models_dir)
        };

        state.settings.model_list = model_detector::all_models()
            .iter()
            .map(|info| {
                let is_ready = model_detector::is_model_ready(&models_dir, &info.kind, &info.id);
                crate::state::ModelItem {
                    id: info.id.clone(),
                    name: info.name.clone(),
                    kind: info.kind.clone(),
                    engine: info.engine.clone(),
                    status: if is_ready { "就绪".to_string() } else { "未下载".to_string() },
                }
            })
            .collect();
    }
}
