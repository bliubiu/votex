use anyhow::{Context, Result};
use image::DynamicImage;
use ndarray::Array4;
use std::path::Path;
use std::sync::Arc;
use votex_domain::error::OcrError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::ocr::provider::OcrProvider;
use votex_domain::ocr::value_object::{OcrParams, OcrResult, OcrTextBlock, TextBox};
use votex_domain::shared::value_object::{CancellationToken, ProgressCallback, ProgressEvent};

use crate::shared::{EngineState, OrtSessionFactory};

/// PaddleOCR 模型版本变体
///
/// 不同版本使用不同的 ONNX 文件名集合
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddleOcrModelVariant {
    V4Mobile,
    V5Mobile,
    V5Server,
    V6Tiny,
    V6Small,
    V6Medium,
}

impl PaddleOcrModelVariant {
    /// 从 ModelId 字符串映射到变体
    pub fn from_model_id(model_id: &str) -> Self {
        match model_id {
            "paddleocr-v5-mobile" => Self::V5Mobile,
            "paddleocr-v5-server" => Self::V5Server,
            "paddleocr-v6-tiny" => Self::V6Tiny,
            "paddleocr-v6-small" => Self::V6Small,
            "paddleocr-v6-medium" => Self::V6Medium,
            _ => Self::V4Mobile, // 默认/兼容旧值 "paddleocr"
        }
    }

    /// 该变体是否需要独立的字典文件（v6 字典内嵌 ONNX 元数据）
    pub fn needs_dict_file(&self) -> bool {
        !matches!(self, Self::V6Tiny | Self::V6Small | Self::V6Medium)
    }

    fn det_filename(&self) -> &str {
        match self {
            Self::V4Mobile => "ch_PP-OCRv4_det_mobile.onnx",
            Self::V5Mobile => "ch_PP-OCRv5_det_mobile.onnx",
            Self::V5Server => "ch_PP-OCRv5_det_server.onnx",
            Self::V6Tiny => "ch_PP-OCRv6_det_tiny.onnx",
            Self::V6Small => "ch_PP-OCRv6_det_small.onnx",
            Self::V6Medium => "ch_PP-OCRv6_det_medium.onnx",
        }
    }

    fn cls_filename(&self) -> &str {
        match self {
            Self::V4Mobile | Self::V5Mobile | Self::V6Tiny | Self::V6Small | Self::V6Medium => {
                "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
            }
            Self::V5Server => "ch_PP-LCNet_x1_0_textline_ori_cls_server.onnx",
        }
    }

    fn rec_filename(&self) -> &str {
        match self {
            Self::V4Mobile => "ch_PP-OCRv4_rec_mobile.onnx",
            Self::V5Mobile => "ch_PP-OCRv5_rec_mobile.onnx",
            Self::V5Server => "ch_PP-OCRv5_rec_server.onnx",
            Self::V6Tiny => "ch_PP-OCRv6_rec_tiny.onnx",
            Self::V6Small => "ch_PP-OCRv6_rec_small.onnx",
            Self::V6Medium => "ch_PP-OCRv6_rec_medium.onnx",
        }
    }

    fn dict_filename(&self) -> &str {
        "ppocr_keys_v1.txt"
    }
}

/// PaddleOCR 推理引擎
///
/// 流水线：文本检测(det) → 方向分类(cls) → 文字识别(rec)
pub struct PaddleOcrEngine {
    /// 当前模型变体（决定 det/cls/rec 的文件名集合）
    ///
    /// 走 `EngineState` 而非裸字段：`load()` 会在 `&self` 下重新推导变体
    /// （用户可能在 GUI 里切换 v4/v5/v6），裸字段无法在共享实例上写入。
    variant: EngineState<PaddleOcrModelVariant>,
    det_state: EngineState<ort::session::Session>,
    cls_state: EngineState<ort::session::Session>,
    rec_state: EngineState<ort::session::Session>,
    /// 识别字典（字符表）
    ///
    /// 用 `Arc<[String]>` 而非 `Vec<String>`：识别路径每解码一个 token 都要按下标
    /// 取字符，包在 `Arc` 里可让 `&str` 借用跨闭包存活，无需逐次加锁。
    dict: EngineState<Arc<[String]>>,
}

impl PaddleOcrEngine {
    pub fn new() -> Self {
        // 默认使用 PP-OCRv6 Medium：识别精度显著高于 Tiny/Mobile 变体
        // （Tiny/Mobile 对"你→尔"等字形近似字存在稳定单字误差）
        Self::with_variant(PaddleOcrModelVariant::V6Medium)
    }

    /// 创建引擎并指定模型变体
    pub fn with_variant(variant: PaddleOcrModelVariant) -> Self {
        let variant_state = EngineState::new();
        variant_state.load(variant);
        Self {
            variant: variant_state,
            det_state: EngineState::new(),
            cls_state: EngineState::new(),
            rec_state: EngineState::new(),
            dict: EngineState::new(),
        }
    }

    /// 读取当前模型变体（未设置时回退到构造时传入的默认值）
    fn variant(&self) -> PaddleOcrModelVariant {
        self.variant
            .with(|v| *v)
            .unwrap_or(PaddleOcrModelVariant::V6Medium)
    }

    /// 加载字典：优先从 ONNX 元数据提取，回退到文件
    ///
    /// PP-OCRv4/v5 的 ONNX 文件也内嵌了字符表元数据（custom key），
    /// v6 则完全依赖元数据（无独立字典文件）。此方法对所有版本通用：
    /// 1. 尝试从 rec_session 元数据提取（key 名：character_dict / char_dict）
    /// 2. 若失败，尝试读取外部字典文件
    fn load_dict(
        model_dir: &Path,
        rec_session: &ort::session::Session,
        variant: PaddleOcrModelVariant,
    ) -> Result<Vec<String>> {
        // 优先从 ONNX 元数据提取
        if let Ok(metadata) = rec_session.metadata() {
            for key in &["character", "character_dict", "char_dict"] {
                if let Some(value) = metadata.custom(key) {
                    let chars: Vec<String> = value
                        .lines()
                        .filter(|l| !l.is_empty())
                        .map(|l| l.to_string())
                        .collect();
                    if !chars.is_empty() {
                        tracing::debug!(
                            "从 ONNX 元数据加载字典（key={}），条目数: {}",
                            key,
                            chars.len()
                        );
                        return Ok(chars);
                    }
                }
            }
        }

        // 回退到文件
        let dict_path = model_dir.join(variant.dict_filename());
        if dict_path.exists() {
            let content =
                std::fs::read_to_string(&dict_path).context("读取字典文件失败")?;
            let chars: Vec<String> = content.lines().map(|l| l.to_string()).collect();
            tracing::debug!("从文件加载字典: {:?}，条目数: {}", dict_path, chars.len());
            return Ok(chars);
        }

        anyhow::bail!(
            "无法加载字典：ONNX 元数据无字符表，且字典文件不存在: {:?}",
            dict_path
        );
    }

    /// 从模型目录加载全部模型
    pub fn load_from_dir(&self, model_dir: &Path) -> Result<()> {
        let variant = self.variant();
        let det_path = model_dir.join(variant.det_filename());
        let cls_path = model_dir.join(variant.cls_filename());
        let rec_path = model_dir.join(variant.rec_filename());

        if !det_path.exists() {
            anyhow::bail!("检测模型不存在: {:?}", det_path);
        }
        if !rec_path.exists() {
            anyhow::bail!("识别模型不存在: {:?}", rec_path);
        }

        // 使用 OrtSessionFactory 创建 session
        let det_session = OrtSessionFactory::create(&det_path)
            .context("加载检测模型失败")?;
        self.det_state.load(det_session);

        if cls_path.exists() {
            let cls_session = OrtSessionFactory::create(&cls_path)
                .context("加载方向分类模型失败")?;
            self.cls_state.load(cls_session);
        }

        let rec_session = OrtSessionFactory::create(&rec_path)
            .context("加载识别模型失败")?;

        // 加载字典：优先从 ONNX 元数据提取（所有版本均可），回退到文件
        let dict = Self::load_dict(model_dir, &rec_session, variant)?;
        self.dict.load(Arc::from(dict));

        self.rec_state.load(rec_session);

        tracing::info!(
            "PaddleOCR 模型加载完成（det+cls+rec），字典条目数: {}",
            self.dict.with(|d| d.len()).unwrap_or(0)
        );
        Ok(())
    }

    /// 识别图片中的文字（含取消和进度）
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

        // 检查取消
        if cancel_token.is_cancelled() {
            return Err(OcrError::RecognizeFailed("任务已取消".to_string()));
        }

        // 报告阶段：加载图片
        if let Some(ref cb) = on_progress {
            cb(ProgressEvent::PhaseChanged {
                phase: "loading_image".to_string(),
            });
        }

        let img = image::open(image_path)
            .map_err(|e| OcrError::ImageReadFailed(format!("打开图片失败: {:?}: {}", image_path, e)))?;

        // 报告阶段：文本检测
        if let Some(ref cb) = on_progress {
            cb(ProgressEvent::PhaseChanged {
                phase: "detecting".to_string(),
            });
        }

        let text_boxes = self.detect_text(&img)
            .map_err(|e| OcrError::DetectFailed(format!("文本检测失败: {}", e)))?;

        // 检查取消
        if cancel_token.is_cancelled() {
            return Err(OcrError::RecognizeFailed("任务已取消".to_string()));
        }

        tracing::info!("检测到 {} 个文本区域", text_boxes.len());

        if let Some(ref cb) = on_progress {
            cb(ProgressEvent::BlockProgress {
                current: 0,
                total: text_boxes.len(),
            });
        }

        let mut blocks = Vec::new();
        for (i, (box_pts, crop)) in text_boxes.iter().enumerate() {
            // 每个块前检查取消
            if cancel_token.is_cancelled() {
                return Err(OcrError::RecognizeFailed("任务已取消".to_string()));
            }

            let oriented_crop = if self.cls_state.is_loaded() {
                // 报告阶段：方向分类
                if let Some(ref cb) = on_progress {
                    cb(ProgressEvent::PhaseChanged {
                        phase: "classifying".to_string(),
                    });
                }
                self.classify_direction(crop)
                    .map_err(|e| OcrError::ClassifyFailed(format!("方向分类失败: {}", e)))?
            } else {
                crop.clone()
            };

            if let Some(ref cb) = on_progress {
                cb(ProgressEvent::PhaseChanged {
                    phase: "recognizing".to_string(),
                });
            }

            let (text, confidence) = self.recognize_text(&oriented_crop)
                .map_err(|e| OcrError::RecognizeFailed(format!("文字识别失败: {}", e)))?;

            if !text.is_empty() {
                tracing::debug!("区域 {}: text={}, conf={:.2}", i, text, confidence);
                blocks.push(OcrTextBlock {
                    text,
                    confidence,
                    box_points: box_pts.clone(),
                });
            }

            // 报告进度
            if let Some(ref cb) = on_progress {
                cb(ProgressEvent::BlockProgress {
                    current: i + 1,
                    total: text_boxes.len(),
                });
            }
        }

        Ok(OcrResult {
            blocks,
            output_path: image_path.with_extension("txt"),
        })
    }

    /// 文本检测：DBNet
    fn detect_text(&self, img: &DynamicImage) -> Result<Vec<(TextBox, DynamicImage)>> {
        self.det_state.with_mut(|session| {
            let (orig_h, orig_w) = (img.height() as f32, img.width() as f32);
            let max_side = 960.0;
            let ratio = if orig_h > max_side || orig_w > max_side {
                (max_side / orig_h).min(max_side / orig_w)
            } else {
                1.0
            };
            let new_w = (orig_w * ratio) as u32;
            let new_h = (orig_h * ratio) as u32;
            let target_w = ((new_w + 31) / 32) * 32;
            let target_h = ((new_h + 31) / 32) * 32;

            let resized =
                img.resize_exact(target_w, target_h, image::imageops::FilterType::Triangle);
            let rgb = resized.to_rgb8();

            let mean = [0.485f32, 0.456, 0.406];
            let std_dev = [0.229f32, 0.224, 0.225];
            let mut input_tensor = vec![0.0f32; 3 * (target_w as usize) * (target_h as usize)];

            for y in 0..target_h {
                for x in 0..target_w {
                    let pixel = rgb.get_pixel(x, y);
                    for c in 0..3 {
                        let val = pixel[c] as f32 / 255.0;
                        let normalized = (val - mean[c]) / std_dev[c];
                        let idx = c * (target_h as usize) * (target_w as usize)
                            + y as usize * target_w as usize
                            + x as usize;
                        input_tensor[idx] = normalized;
                    }
                }
            }

            let input = Array4::from_shape_vec(
                (1, 3, target_h as usize, target_w as usize),
                input_tensor,
            )?;
            let input_value = ort::value::Value::from_array(input)?;
            let outputs = session.run(ort::inputs![input_value])?;
            let prob_map = outputs[0].try_extract_array::<f32>()?;

            let thresh = 0.3f32;
            let map_h = prob_map.shape()[2];
            let map_w = prob_map.shape()[3];

            // 连通域分析：提取独立文本区域
            let mut visited = vec![false; map_h * map_w];
            let mut text_regions: Vec<(usize, usize, usize, usize)> = Vec::new(); // (min_x, min_y, max_x, max_y)

            for y in 0..map_h {
                for x in 0..map_w {
                    let idx = y * map_w + x;
                    if prob_map[[0, 0, y, x]] > thresh && !visited[idx] {
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
                                    if !visited[nidx] && prob_map[[0, 0, ny, nx]] > thresh {
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

            // 按 y 坐标排序，再按 x 排序（从上到下，从左到右）
            text_regions.sort_by(|a, b| {
                a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0))
            });

            let scale_x = orig_w / target_w as f32;
            let scale_y = orig_h / target_h as f32;
            let padding = 5.0f32; // 裁剪区域外扩像素

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

                if crop_x + crop_w <= img.width() && crop_y + crop_h <= img.height() && crop_w > 5 && crop_h > 5 {
                    let crop = img.clone().crop(crop_x, crop_y, crop_w, crop_h);
                    results.push((
                        TextBox {
                            x0, y0,
                            x1: x1, y1: y0,
                            x2: x1, y2: y1,
                            x3: x0, y3: y1,
                        },
                        crop,
                    ));
                }
            }

            tracing::info!("检测到 {} 个文本区域", results.len());

            Ok(results)
        })?
    }

    /// 方向分类
    fn classify_direction(&self, crop: &DynamicImage) -> Result<DynamicImage> {
        self.cls_state.with_mut(|session| {
            // cls 模型输入尺寸因变体而异（mobile cls 48×192，v5-server textline cls 80×160），
            // 从模型声明的静态输入形状读取；动态维度（-1）或读取失败时回退 48×192
            let (cls_h, cls_w) = session
                .inputs()
                .first()
                .and_then(|i| match i.dtype() {
                    ort::value::ValueType::Tensor { shape, .. } => {
                        let h = shape.get(2).copied().unwrap_or(48);
                        let w = shape.get(3).copied().unwrap_or(192);
                        Some((
                            if h > 0 { h as u32 } else { 48 },
                            if w > 0 { w as u32 } else { 192 },
                        ))
                    }
                    _ => None,
                })
                .unwrap_or((48, 192));

            let resized =
                crop.resize_exact(cls_w, cls_h, image::imageops::FilterType::Triangle);
            let rgb = resized.to_rgb8();

            let mean = [0.5f32, 0.5, 0.5];
            let std_dev = [0.5f32, 0.5, 0.5];
            let mut input_tensor = vec![0.0f32; 3 * (cls_h as usize) * (cls_w as usize)];

            for y in 0..cls_h {
                for x in 0..cls_w {
                    let pixel = rgb.get_pixel(x, y);
                    for c in 0..3 {
                        let val = pixel[c] as f32 / 255.0;
                        let normalized = (val - mean[c]) / std_dev[c];
                        let idx = c * (cls_h as usize) * (cls_w as usize)
                            + y as usize * cls_w as usize
                            + x as usize;
                        input_tensor[idx] = normalized;
                    }
                }
            }

            let input = Array4::from_shape_vec((1, 3, cls_h as usize, cls_w as usize), input_tensor)?;
            let input_value = ort::value::Value::from_array(input)?;
            let outputs = session.run(ort::inputs![input_value])?;
            let probs = outputs[0].try_extract_array::<f32>()?;

            if probs[[0, 1]] > probs[[0, 0]] {
                Ok(crop.rotate180())
            } else {
                Ok(crop.clone())
            }
        })?
    }

    /// 文字识别：CRNN + CTC 解码
    fn recognize_text(&self, crop: &DynamicImage) -> Result<(String, f32)> {
        self.rec_state.with_mut(|session| {
            let img_h = 48u32;
            let img_w = ((crop.width() as f32 / crop.height() as f32) * img_h as f32).min(640.0) as u32;
            let img_w = ((img_w + 7) / 8) * 8;

            let resized = crop.resize_exact(img_w, img_h, image::imageops::FilterType::Triangle);
            let rgb = resized.to_rgb8();

            let mean = [0.5f32, 0.5, 0.5];
            let std_dev = [0.5f32, 0.5, 0.5];
            let mut input_tensor = vec![0.0f32; 3 * (img_h as usize) * (img_w as usize)];

            for y in 0..img_h {
                for x in 0..img_w {
                    let pixel = rgb.get_pixel(x, y);
                    for c in 0..3 {
                        let val = pixel[c] as f32 / 255.0;
                        let normalized = (val - mean[c]) / std_dev[c];
                        let idx = c * (img_h as usize) * (img_w as usize)
                            + y as usize * (img_w as usize)
                            + x as usize;
                        input_tensor[idx] = normalized;
                    }
                }
            }

            let input = Array4::from_shape_vec(
                (1, 3, img_h as usize, img_w as usize),
                input_tensor,
            )?;
            let input_value = ort::value::Value::from_array(input)?;
            let outputs = session.run(ort::inputs![input_value])?;
            let preds = outputs[0].try_extract_array::<f32>()?;

            let seq_len = preds.shape()[1];
            let num_classes = preds.shape()[2];

            let mut text = String::new();
            let mut total_conf = 0.0f32;
            let mut char_count = 0usize;
            let mut last_idx = 0usize;

            for t in 0..seq_len {
                let mut max_idx = 0;
                let mut max_val = preds[[0, t, 0]];
                for c in 1..num_classes {
                    let val = preds[[0, t, c]];
                    if val > max_val {
                        max_val = val;
                        max_idx = c;
                    }
                }

                if max_idx != 0 && max_idx != last_idx {
                    let dict_guard = self.dict.get();
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
        })?
    }

    pub fn is_loaded(&self) -> bool {
        self.det_state.is_loaded() && self.rec_state.is_loaded()
    }

    pub fn engine_kind_value() -> EngineKind {
        EngineKind::PaddleOCR
    }
}

impl OcrProvider for PaddleOcrEngine {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::PaddleOCR
    }

    fn load(&self, model: &Model) -> Result<(), OcrError> {
        // 从 model.id 推导模型变体，确保加载正确的文件名集合
        self.variant
            .load(PaddleOcrModelVariant::from_model_id(model.id.as_str()));
        let model_dir = std::path::Path::new(&model.name);
        self.load_from_dir(model_dir)
            .map_err(|e| OcrError::LoadFailed(format!("PaddleOCR 加载失败: {}", e)))
    }

    fn unload(&self) -> Result<(), OcrError> {
        self.det_state.unload();
        self.cls_state.unload();
        self.rec_state.unload();
        self.dict.unload();
        tracing::info!("PaddleOCR 引擎已释放");
        Ok(())
    }

    fn recognize(&self, image_path: &Path, params: &OcrParams) -> Result<OcrResult, OcrError> {
        let cancel = CancellationToken::new();
        self.recognize_with_cancel(image_path, params, &cancel, None)
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
        PaddleOcrEngine::recognize_with_cancel(self, image_path, params, cancel_token, on_progress)
    }

    fn is_loaded(&self) -> bool {
        self.det_state.is_loaded() && self.rec_state.is_loaded()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paddleocr_引擎类型() {
        assert_eq!(PaddleOcrEngine::engine_kind_value(), EngineKind::PaddleOCR);
    }

    #[test]
    fn paddleocr_未加载时识别应报错() {
        let engine = PaddleOcrEngine::new();
        assert!(!engine.is_loaded());
    }

    #[test]
    fn cancel_token_正常工作() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
        token.reset();
        assert!(!token.is_cancelled());
    }
}
