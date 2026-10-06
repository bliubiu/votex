use crate::error::OcrError;
use crate::model::entity::Model;
use crate::model::value_object::EngineKind;
use crate::ocr::value_object::{OcrParams, OcrResult};
use crate::shared::value_object::{CancellationToken, ProgressCallback};

/// OCR 引擎 Provider 接口
pub trait OcrProvider: Send + Sync {
    /// 返回引擎类型
    fn engine_kind(&self) -> EngineKind;

    /// 加载模型到内存
    fn load(&self, model: &Model) -> Result<(), OcrError>;

    /// 释放模型资源
    fn unload(&self) -> Result<(), OcrError>;

    /// 识别单张图片
    fn recognize(
        &self,
        image_path: &std::path::Path,
        params: &OcrParams,
    ) -> Result<OcrResult, OcrError>;

    /// 识别单张图片（带取消和进度回调）
    fn recognize_with_cancel(
        &self,
        image_path: &std::path::Path,
        params: &OcrParams,
        cancel_token: &CancellationToken,
        on_progress: ProgressCallback,
    ) -> Result<OcrResult, OcrError> {
        // 默认实现：忽略取消和进度，调用基础 recognize
        let _ = cancel_token;
        let _ = on_progress;
        self.recognize(image_path, params)
    }

    /// 是否已加载
    fn is_loaded(&self) -> bool;
}

/// OCR 导出器接口（统一导出格式）
pub trait OcrExporter: Send + Sync {
    /// 导出格式名称
    fn format_name(&self) -> &'static str;

    /// 导出的文件扩展名
    fn extension(&self) -> &'static str;

    /// 执行导出
    fn export(
        &self,
        result: &OcrResult,
        output_path: &std::path::Path,
    ) -> Result<(), OcrError>;
}

/// OCR 导出器注册表
pub struct OcrExporterRegistry {
    exporters: Vec<Box<dyn OcrExporter>>,
}

impl OcrExporterRegistry {
    pub fn new() -> Self {
        Self {
            exporters: Vec::new(),
        }
    }

    /// 注册导出器
    pub fn register(&mut self, exporter: Box<dyn OcrExporter>) {
        self.exporters.push(exporter);
    }

    /// 按格式名查找导出器
    pub fn find_by_format(&self, format_name: &str) -> Option<&Box<dyn OcrExporter>> {
        self.exporters
            .iter()
            .find(|e| e.format_name() == format_name)
    }

    /// 列出所有注册的格式
    pub fn available_formats(&self) -> Vec<&'static str> {
        self.exporters.iter().map(|e| e.format_name()).collect()
    }

    /// 导出数量
    pub fn count(&self) -> usize {
        self.exporters.len()
    }
}
