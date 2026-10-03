use crate::error::AsrError;
use crate::model::entity::Model;
use crate::model::value_object::EngineKind;
use crate::asr::value_object::{AsrParams, RecognizeOutput};
use crate::shared::value_object::AudioData;

/// ASR 引擎 Provider 接口
pub trait AsrProvider: Send + Sync {
    /// 返回引擎类型
    fn engine_kind(&self) -> EngineKind;

    /// 加载模型到内存
    fn load(&mut self, model: &Model) -> Result<(), AsrError>;

    /// 释放模型资源
    fn unload(&mut self) -> Result<(), AsrError>;

    /// 识别单段音频
    fn recognize(
        &self,
        audio: &AudioData,
        params: &AsrParams,
    ) -> Result<RecognizeOutput, AsrError>;

    /// 期望输入采样率
    fn sample_rate(&self) -> u32;

    /// 是否已加载
    fn is_loaded(&self) -> bool;
}
