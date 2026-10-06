//! EasyOCR 引擎（ONNX 推理）
//!
//! EasyOCR 支持 80+ 种语言识别，包括中文简繁体、英文、日文、韩文等。
//! 使用 CRAFT 文本检测 + CRNN 文本识别的 ONNX 模型。
//!
//! # 模型文件
//! - `models/EasyOCR/craft_det.onnx` - CRAFT 文本检测模型
//! - `models/EasyOCR/craft_refiner.onnx` - CRAFT 文本细化模型（可选）
//! - `models/EasyOCR/recognizer.onnx` - CRNN 文本识别模型
//! - `models/EasyOCR/char_dict.txt` - 字符字典文件
//!
//! # 集成状态
//! 【已完成】ONNX 推理管线集成

use std::path::Path;

use image::DynamicImage;
use ndarray::Array4;
use ort::session::Session;
use std::sync::Arc;

use votex_domain::error::OcrError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::ocr::provider::OcrProvider;
use votex_domain::ocr::value_object::{OcrParams, OcrResult, OcrTextBlock, TextBox};
use votex_domain::shared::value_object::{CancellationToken, ProgressCallback, ProgressEvent};

use crate::shared::{EngineState, ModelFileLocator, OrtSessionFactory};

/// EasyOCR Provider（ONNX 推理）
pub struct EasyOcrProvider {
    det_state: EngineState<Session>,
    rec_state: EngineState<Session>,
    /// 识别字典（字符表）
    ///
    /// 用 `Arc<[String]>` 而非 `Vec<String>`：识别路径按 token 下标取字符，
    /// 包在 `Arc` 里可让 `&str` 借用跨闭包存活，无需逐次加锁。
    char_dict: EngineState<Arc<[String]>>,
}

impl EasyOcrProvider {
    pub fn new() -> Self {
        Self {
            det_state: EngineState::new(),
            rec_state: EngineState::new(),
            char_dict: EngineState::new(),
        }
    }

    /// 从模型目录加载
    pub fn load_from_dir(&self, model_dir: &Path) -> Result<(), OcrError> {
        let det_path = ModelFileLocator::find_auxiliary_file(model_dir, "craft_det.onnx")
            .map_err(|e| OcrError::LoadFailed(format!("{}", e)))?;
        let rec_path = ModelFileLocator::find_auxiliary_file(model_dir, "recognizer.onnx")
            .map_err(|e| OcrError::LoadFailed(format!("{}", e)))?;
        let dict_path = ModelFileLocator::find_auxiliary_file(model_dir, "char_dict.txt")
            .map_err(|e| OcrError::LoadFailed(format!("{}", e)))?;

        let det_session = OrtSessionFactory::create(&det_path)
            .map_err(|e| OcrError::LoadFailed(format!("加载检测模型失败: {}", e)))?;
        self.det_state.load(det_session);

        let rec_session = OrtSessionFactory::create(&rec_path)
            .map_err(|e| OcrError::LoadFailed(format!("加载识别模型失败: {}", e)))?;
        self.rec_state.load(rec_session);

        let dict_content = std::fs::read_to_string(&dict_path)
            .map_err(|e| OcrError::LoadFailed(format!("读取字典文件失败: {}", e)))?;
        let dict: Vec<String> = dict_content.lines().map(|l| l.to_string()).collect();
        self.char_dict.load(Arc::from(dict));

        tracing::info!(
            "EasyOCR 模型加载完成（det+rec），字典条目数: {}",
            self.char_dict.with(|d| d.len()).unwrap_or(0)
        );
        Ok(())
    }

    /// CRAFT 文本检测
    fn detect_text(&self, img: &DynamicImage) -> Result<Vec<(TextBox, DynamicImage)>, OcrError> {
        let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);
        let max_side = 1280.0;
        let ratio = if orig_h > max_side || orig_w > max_side {
            (max_side / orig_h).min(max_side / orig_w)
        } else {
            1.0
        };
        let new_w = ((orig_w * ratio) as u32 / 32) * 32;
        let new_h = ((orig_h * ratio) as u32 / 32) * 32;

        let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // 归一化：ImageNet 均值和标准差
        let mean = [0.485f32, 0.456, 0.406];
        let std_dev = [0.229f32, 0.224, 0.225];
        let mut input_tensor = vec![0.0f32; 3 * (new_w as usize) * (new_h as usize)];

        for y in 0..new_h {
            for x in 0..new_w {
                let pixel = rgb.get_pixel(x, y);
                for c in 0..3 {
                    let val = pixel[c] as f32 / 255.0;
                    let normalized = (val - mean[c]) / std_dev[c];
                    let idx = c * (new_h as usize) * (new_w as usize)
                        + y as usize * new_w as usize
                        + x as usize;
                    input_tensor[idx] = normalized;
                }
            }
        }

        let input = Array4::from_shape_vec(
            (1, 3, new_h as usize, new_w as usize),
            input_tensor,
        )
        .map_err(|e| OcrError::RecognizeFailed(format!("创建输入张量失败: {}", e)))?;

        let input_value = ort::value::Value::from_array(input)
            .map_err(|e| OcrError::RecognizeFailed(format!("创建输入值失败: {}", e)))?;

        // 使用 EngineState 执行检测推理，提取 region_map 为 owned 数据
        let region_map_owned = self.det_state.with_mut(|session| {
            let outputs = session
                .run(ort::inputs![input_value])
                .map_err(|e| OcrError::RecognizeFailed(format!("CRAFT 推理失败: {}", e)))?;

            let region_map = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| OcrError::RecognizeFailed(format!("提取区域图失败: {}", e)))?;

            Ok::<Vec<f32>, OcrError>(region_map.iter().copied().collect())
        }).map_err(|_| OcrError::EngineNotLoaded)??;

        // 阈值分割提取文本区域
        let thresh = 0.7f32;
        let map_h = (new_h as usize) / 2; // CRAFT 输出分辨率通常为输入的一半
        let map_w = (new_w as usize) / 2;

        let mut visited = vec![false; map_h * map_w];
        let mut text_regions: Vec<(usize, usize, usize, usize)> = Vec::new();

        for y in 0..map_h {
            for x in 0..map_w {
                let idx = y * map_w + x;
                if idx < region_map_owned.len() && region_map_owned[idx] > thresh && !visited[idx] {
                    // BFS 找连通域
                    let mut queue = std::collections::VecDeque::new();
                    queue.push_back((x, y));
                    visited[idx] = true;
                    let mut min_x = x;
                    let mut min_y = y;
                    let mut max_x = x;
                    let mut max_y = y;

                    while let Some((cx, cy)) = queue.pop_front() {
                        for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                            let nx = cx as i32 + dx;
                            let ny = cy as i32 + dy;
                            if nx >= 0 && nx < map_w as i32 && ny >= 0 && ny < map_h as i32 {
                                let nx = nx as usize;
                                let ny = ny as usize;
                                let nidx = ny * map_w + nx;
                                if nidx < region_map_owned.len()
                                    && !visited[nidx]
                                    && region_map_owned[nidx] > thresh
                                {
                                    visited[nidx] = true;
                                    queue.push_back((nx, ny));
                                    min_x = min_x.min(nx);
                                    min_y = min_y.min(ny);
                                    max_x = max_x.max(nx);
                                    max_y = max_y.max(ny);
                                }
                            }
                        }
                    }

                    text_regions.push((min_x, min_y, max_x, max_y));
                }
            }
        }

        // 按 y 坐标排序，再按 x 排序
        text_regions.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let scale_x = orig_w / new_w as f32;
        let scale_y = orig_h / new_h as f32;
        let padding = 5.0f32;

        let mut results = Vec::new();

        for (start_x, start_y, end_x, end_y) in text_regions {
            let x0 = (start_x as f32 * scale_x - padding).max(0.0);
            let y0 = (start_y as f32 * scale_y - padding).max(0.0);
            let x1 = ((end_x + 1) as f32 * scale_x + padding).min(orig_w);
            let y1 = ((end_y + 1) as f32 * scale_y + padding).min(orig_h);

            let crop_x = x0 as u32;
            let crop_y = y0 as u32;
            let crop_w = (x1 - x0).max(1.0) as u32;
            let crop_h = (y1 - y0).max(1.0) as u32;

            if crop_x + crop_w <= img.width()
                && crop_y + crop_h <= img.height()
                && crop_w > 5
                && crop_h > 5
            {
                let crop = img.clone().crop(crop_x, crop_y, crop_w, crop_h);
                results.push((
                    TextBox {
                        x0,
                        y0,
                        x1,
                        y1: y0,
                        x2: x1,
                        y2: y1,
                        x3: x0,
                        y3: y1,
                    },
                    crop,
                ));
            }
        }

        tracing::info!("EasyOCR 检测到 {} 个文本区域", results.len());
        Ok(results)
    }

    /// CRNN 文本识别
    fn recognize_text(&self, crop: &DynamicImage) -> Result<(String, f32), OcrError> {
        let img_h = 64u32;
        let img_w = ((crop.width() as f32 / crop.height() as f32) * img_h as f32).min(512.0) as u32;
        let img_w = ((img_w + 3) / 4) * 4; // 确保是 4 的倍数

        let resized = crop.resize_exact(img_w, img_h, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // 归一化：[-1, 1]
        let mut input_tensor = vec![0.0f32; 3 * (img_h as usize) * (img_w as usize)];

        for y in 0..img_h {
            for x in 0..img_w {
                let pixel = rgb.get_pixel(x, y);
                for c in 0..3 {
                    let val = pixel[c] as f32 / 127.5 - 1.0;
                    let idx = c * (img_h as usize) * (img_w as usize)
                        + y as usize * img_w as usize
                        + x as usize;
                    input_tensor[idx] = val;
                }
            }
        }

        let input = Array4::from_shape_vec(
            (1, 3, img_h as usize, img_w as usize),
            input_tensor,
        )
        .map_err(|e| OcrError::RecognizeFailed(format!("创建输入张量失败: {}", e)))?;

        let input_value = ort::value::Value::from_array(input)
            .map_err(|e| OcrError::RecognizeFailed(format!("创建输入值失败: {}", e)))?;

        // 使用 EngineState 执行识别推理，提取 predictions 为 owned 数据
        let preds_owned = self.rec_state.with_mut(|session| {
            let outputs = session
                .run(ort::inputs![input_value])
                .map_err(|e| OcrError::RecognizeFailed(format!("CRNN 推理失败: {}", e)))?;

            let preds = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| OcrError::RecognizeFailed(format!("提取输出失败: {}", e)))?;

            Ok::<Vec<f32>, OcrError>(preds.iter().copied().collect())
        }).map_err(|_| OcrError::EngineNotLoaded)??;

        let seq_len = preds_owned.len() / (1 * 1); // 简化：假设 shape [1, seq_len, num_classes]
        let num_classes = if seq_len > 0 {
            preds_owned.len() / seq_len
        } else {
            0
        };

        let mut text = String::new();
        let mut total_conf = 0.0f32;
        let mut char_count = 0usize;
        let mut last_idx = 0usize;

        // CTC 解码
        for t in 0..seq_len {
            let mut max_idx = 0;
            let mut max_val = preds_owned.get(t * num_classes).copied().unwrap_or(0.0);
            for c in 1..num_classes {
                let idx = t * num_classes + c;
                if let Some(&val) = preds_owned.get(idx) {
                    if val > max_val {
                        max_val = val;
                        max_idx = c;
                    }
                }
            }

            // 跳过空白符（0）和重复字符
            if max_idx != 0 && max_idx != last_idx {
                let dict_guard = self.char_dict.get();
                let dict_ref = dict_guard.as_ref().and_then(|d| d.as_ref());
                if max_idx >= 1 && (max_idx as usize - 1) < dict_ref.map_or(0, |d| d.len()) {
                    text.push_str(&dict_ref.expect("上一步已确认长度")[max_idx as usize - 1]);
                    total_conf += max_val;
                    char_count += 1;
                }
            }
            last_idx = max_idx;
        }

        let avg_conf = if char_count > 0 {
            total_conf / char_count as f32
        } else {
            0.0
        };

        Ok((text, avg_conf))
    }

    /// 识别图片中的文字（含取消和进度回调）
    #[allow(unused_variables)]
    pub fn recognize_with_cancel(
        &self,
        image_path: &Path,
        params: &OcrParams,
        cancel_token: &CancellationToken,
        on_progress: ProgressCallback,
    ) -> Result<OcrResult, OcrError> {
        if !self.det_state.is_loaded() {
            return Err(OcrError::EngineNotLoaded);
        }

        if cancel_token.is_cancelled() {
            return Err(OcrError::RecognizeFailed("任务已取消".to_string()));
        }

        if let Some(ref cb) = on_progress {
            cb(ProgressEvent::PhaseChanged { phase: "loading_image".to_string() });
        }

        let img = image::open(image_path)
            .map_err(|e| OcrError::ImageReadFailed(format!("打开图片失败: {:?}: {}", image_path, e)))?;

        if cancel_token.is_cancelled() {
            return Err(OcrError::RecognizeFailed("任务已取消".to_string()));
        }

        if let Some(ref cb) = on_progress {
            cb(ProgressEvent::PhaseChanged { phase: "detecting".to_string() });
        }

        let text_boxes = self.detect_text(&img)?;

        if cancel_token.is_cancelled() {
            return Err(OcrError::RecognizeFailed("任务已取消".to_string()));
        }

        tracing::info!("EasyOCR 检测到 {} 个文本区域", text_boxes.len());

        if let Some(ref cb) = on_progress {
            cb(ProgressEvent::BlockProgress { current: 0, total: text_boxes.len() });
        }

        let mut blocks = Vec::new();
        for (i, (box_pts, crop)) in text_boxes.iter().enumerate() {
            if cancel_token.is_cancelled() {
                return Err(OcrError::RecognizeFailed("任务已取消".to_string()));
            }

            if let Some(ref cb) = on_progress {
                cb(ProgressEvent::PhaseChanged { phase: "recognizing".to_string() });
            }

            let (text, confidence) = self.recognize_text(crop)?;

            if !text.is_empty() {
                tracing::debug!("EasyOCR 区域 {}: text={}, conf={:.2}", i, text, confidence);
                blocks.push(OcrTextBlock {
                    text,
                    confidence,
                    box_points: box_pts.clone(),
                });
            }

            if let Some(ref cb) = on_progress {
                cb(ProgressEvent::BlockProgress { current: i + 1, total: text_boxes.len() });
            }
        }

        Ok(OcrResult {
            blocks,
            output_path: image_path.with_extension("txt"),
        })
    }
}

impl OcrProvider for EasyOcrProvider {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::EasyOcr
    }

    fn load(&self, model: &Model) -> Result<(), OcrError> {
        let model_dir = ModelFileLocator::locate_model_dir(model)
            .map_err(|e| OcrError::LoadFailed(format!("{}", e)))?;
        self.load_from_dir(&model_dir)
    }

    fn unload(&self) -> Result<(), OcrError> {
        self.det_state.unload();
        self.rec_state.unload();
        self.char_dict.unload();
        tracing::info!("EasyOCR 引擎已释放");
        Ok(())
    }

    fn recognize_with_cancel(
        &self,
        image_path: &Path,
        params: &OcrParams,
        cancel_token: &CancellationToken,
        on_progress: ProgressCallback,
    ) -> Result<OcrResult, OcrError> {
        // 必须用完全限定名调用 inherent 方法。
        // 裸写 `self.recognize_with_cancel(..)` 会解析回本 trait 方法 → 无限递归 → 栈溢出。
        EasyOcrProvider::recognize_with_cancel(self, image_path, params, cancel_token, on_progress)
    }

    fn recognize(
        &self,
        image_path: &Path,
        _params: &OcrParams,
    ) -> Result<OcrResult, OcrError> {
        if !self.det_state.is_loaded() {
            return Err(OcrError::EngineNotLoaded);
        }

        let img = image::open(image_path)
            .map_err(|e| OcrError::RecognizeFailed(format!("打开图片失败: {}", e)))?;

        let text_boxes = self.detect_text(&img)?;
        tracing::info!("检测到 {} 个文本区域", text_boxes.len());

        let mut blocks = Vec::new();
        for (box_pts, crop) in text_boxes {
            let (text, confidence) = self.recognize_text(&crop)?;

            if !text.is_empty() {
                tracing::debug!("text={}, conf={:.2}", text, confidence);
                blocks.push(OcrTextBlock {
                    text,
                    confidence,
                    box_points: box_pts,
                });
            }
        }

        Ok(OcrResult {
            blocks,
            output_path: image_path.with_extension("txt"),
        })
    }

    fn is_loaded(&self) -> bool {
        self.det_state.is_loaded() && self.rec_state.is_loaded()
    }
}
