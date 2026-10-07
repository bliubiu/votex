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
    fn load(&self, model: &Model) -> Result<(), AsrError>;

    /// 释放模型资源
    fn unload(&self) -> Result<(), AsrError>;

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

    /// 引擎能力自描述
    ///
    /// 默认从 `engine_kind` / `sample_rate` 组装，无需加载模型即可调用。
    /// 供 `votex model list --json` 与 `votex serve` 的程序化发现复用。
    fn capability(&self) -> crate::model::capability::EngineCapability {
        crate::model::capability::EngineCapability::profile(self.engine_kind())
            .with_sample_rate(self.sample_rate())
    }
}
