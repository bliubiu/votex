use crate::error::TtsError;
use crate::model::entity::Model;
use crate::model::value_object::EngineKind;
use crate::shared::value_object::AudioData;
use crate::tts::dialect::{Dialect, DialectSupport};
use crate::tts::value_object::{TtsParams, VoiceId};

/// TTS 引擎 Provider 接口
pub trait TtsProvider: Send + Sync {
    /// 返回引擎类型
    fn engine_kind(&self) -> EngineKind;

    /// 加载模型到内存
    fn load(&mut self, model: &Model) -> Result<(), TtsError>;

    /// 释放模型资源
    fn unload(&mut self) -> Result<(), TtsError>;

    /// 合成单段语音
    fn synthesize(
        &self,
        text: &str,
        voice: &VoiceId,
        params: &TtsParams,
    ) -> Result<AudioData, TtsError>;

    /// 列出可用音色
    fn list_voices(&self) -> Vec<VoiceId>;

    /// 输出采样率
    fn sample_rate(&self) -> u32;

    /// 是否已加载
    fn is_loaded(&self) -> bool;

    /// 返回引擎支持的方言列表
    fn supported_dialects(&self) -> Vec<DialectSupport> {
        // 默认只支持普通话
        vec![DialectSupport::new(Dialect::Mandarin, crate::tts::dialect::DialectQuality::Native)]
    }
}
