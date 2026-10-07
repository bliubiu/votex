//! 引擎能力自描述
//!
//! 把「某个引擎能做什么」从散落在各 provider、CLI 帮助文本与 GUI 表单里的
//! 隐式知识，收敛为一份可序列化的能力元数据，供：
//!
//! - CLI（`votex model list --json`）
//! - 本地服务（`votex serve` 的 `/v1/capabilities`）
//! - Agent（MCP `tools/list` / `engines_list`）
//!
//! 程序化发现，无需阅读源码或加载模型。
//!
//! # 分层
//!
//! 本模块是**纯领域知识**（零 IO、零 infra 依赖）：
//! [`EngineCapability::profile`] 给出静态画像（分类 / 本地或在线 / 是否支持克隆），
//! provider 再通过 [`crate::tts::provider::TtsProvider::capability`] 覆盖
//! 运行期才确切的采样率与方言列表。

use serde::{Deserialize, Serialize};

use crate::model::value_object::EngineKind;
use crate::provider::engine_kind_to_capability;
use crate::tts::dialect::DialectSupport;

/// 引擎运行位置
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineLocation {
    /// 本地 ONNX 推理，内容零上传
    Local,
    /// 在线 API（需要 API Key，文本/音频会离开本机）
    Online,
}

impl EngineLocation {
    pub fn as_str(&self) -> &'static str {
        match self {
            EngineLocation::Local => "local",
            EngineLocation::Online => "online",
        }
    }
}

/// 引擎能力自描述
///
/// 字段刻意保持「扁平 + 可序列化」，避免消费方（脚本 / Agent）理解嵌套结构。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineCapability {
    /// 引擎标识（kebab-case，与 CLI `--engine` 一致）
    pub engine: String,
    /// 中文显示名
    pub display_name: String,
    /// 能力类别：tts / asr / ocr / llm / translation / video / runtime
    pub category: String,
    /// 运行位置
    pub location: EngineLocation,
    /// 是否本地推理（`location == Local` 的便捷布尔）
    pub local: bool,
    /// 是否支持流式输出（当前实现统一按整段处理，保留字段供后续演进）
    pub streaming: bool,
    /// 是否支持零样本音色克隆
    pub zero_shot_clone: bool,
    /// 采样率：TTS 为输出采样率，ASR 为期望输入采样率；未知为 `None`
    pub sample_rate: Option<u32>,
    /// 支持的方言（含等级），非 TTS 引擎为空
    pub dialects: Vec<DialectSupport>,
    /// 备注（中文，简短）
    pub notes: String,
}

impl EngineCapability {
    /// 由引擎类型构造静态能力画像
    ///
    /// `sample_rate` / `dialects` 留空，由具体 provider 覆盖。
    pub fn profile(engine: EngineKind) -> Self {
        let location = if engine.is_online() {
            EngineLocation::Online
        } else {
            EngineLocation::Local
        };
        Self {
            engine: engine.as_str().to_string(),
            display_name: engine.display_name().to_string(),
            category: engine_kind_to_capability(&engine).code().to_string(),
            location,
            local: location == EngineLocation::Local,
            streaming: false,
            zero_shot_clone: engine.supports_zero_shot_clone(),
            sample_rate: None,
            dialects: Vec::new(),
            notes: capability_notes(engine).to_string(),
        }
    }

    /// 覆盖采样率（provider 侧运行期确切值）
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    /// 覆盖方言列表（provider 侧声明，空列表不覆盖）
    pub fn with_dialects(mut self, dialects: Vec<DialectSupport>) -> Self {
        if !dialects.is_empty() {
            self.dialects = dialects;
        }
        self
    }
}

/// 全部引擎的静态能力画像（未合并 provider 运行期信息）
pub fn all_engine_capabilities() -> Vec<EngineCapability> {
    EngineKind::all()
        .into_iter()
        .map(EngineCapability::profile)
        .collect()
}

/// 引擎备注（中文，简短；无特殊说明返回空串）
fn capability_notes(engine: EngineKind) -> &'static str {
    match engine {
        EngineKind::Kokoro => "轻量中文/英文合成，CPU 即可流畅运行",
        EngineKind::IndexTTS25 => "零样本克隆，粤语原生支持",
        EngineKind::CosyVoice3 => "零样本克隆，克隆需配套参考文本",
        EngineKind::Qwen3Tts => "高质量合成，首次加载需 10~60 秒",
        EngineKind::AzureTts | EngineKind::AzureAsr => "在线 API，需 API Key",
        EngineKind::AliyunTts | EngineKind::AliyunAsr => "在线 API，需 API Key",
        EngineKind::Whisper => "多语言识别，中英混合友好",
        EngineKind::SenseVoice => "中文识别，标点与情感识别",
        EngineKind::Paraformer => "中文识别，速度快",
        EngineKind::Qwen3Asr => "大模型识别，长音频效果好",
        EngineKind::FireRedAsr => "中文识别，CTC 架构",
        EngineKind::WeNet => "Conformer 中文识别",
        EngineKind::OnnxRuntime => "推理运行时动态库，非可加载模型",
        EngineKind::Ollama => "本地 LLM 服务",
        EngineKind::DeepSeek
        | EngineKind::OpenAi
        | EngineKind::QwenLlm
        | EngineKind::Gemini
        | EngineKind::AzureLlm => "在线 LLM，需 API Key",
        EngineKind::OpusMt
        | EngineKind::QwenMt
        | EngineKind::Nllb
        | EngineKind::M2m100
        | EngineKind::HyMt1_5 => "翻译引擎",
        EngineKind::CTranslate2 => "翻译加速后端（需 --features ct2 编译启用，模型需本地转换）",
        EngineKind::Pexels | EngineKind::Pixabay | EngineKind::Coverr => "在线视频素材源",
        EngineKind::SpeakerDiarization => "说话人分割 + 嵌入提取 + 聚类，离线",
        EngineKind::StreamingZipformer => "流式识别，实时听写，端点检测自动分句",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::value_object::ModelKind;

    #[test]
    fn 能力画像_覆盖全部引擎() {
        // `EngineKind::all()` 必须与画像数量一致，遗漏新增引擎会在此暴露
        let caps = all_engine_capabilities();
        let unique: std::collections::HashSet<_> = caps.iter().map(|c| c.engine.clone()).collect();
        assert_eq!(unique.len(), caps.len(), "引擎标识应唯一");
        assert!(caps.len() >= 30, "应覆盖全部引擎，实际 {}", caps.len());
    }

    #[test]
    fn 能力画像_本地在线区分() {
        let kokoro = EngineCapability::profile(EngineKind::Kokoro);
        assert!(kokoro.local);
        assert_eq!(kokoro.location, EngineLocation::Local);
        assert_eq!(kokoro.category, "tts");

        let azure = EngineCapability::profile(EngineKind::AzureTts);
        assert!(!azure.local);
        assert_eq!(azure.location, EngineLocation::Online);
    }

    #[test]
    fn 能力画像_克隆与方言() {
        assert!(EngineCapability::profile(EngineKind::IndexTTS25).zero_shot_clone);
        assert!(EngineCapability::profile(EngineKind::CosyVoice3).zero_shot_clone);
        assert!(!EngineCapability::profile(EngineKind::Kokoro).zero_shot_clone);
    }

    #[test]
    fn 能力画像_类别不再是model_kind兜底() {
        // model_kind 对翻译引擎兜底为 Tts；能力类别必须给出真实分类
        assert_eq!(EngineCapability::profile(EngineKind::Nllb).category, "translation");
        assert_eq!(EngineKind::Nllb.model_kind(), ModelKind::Tts);
        assert_eq!(EngineCapability::profile(EngineKind::DeepSeek).category, "llm");
        assert_eq!(EngineCapability::profile(EngineKind::Pexels).category, "video");
        assert_eq!(
            EngineCapability::profile(EngineKind::OnnxRuntime).category,
            "runtime"
        );
    }

    #[test]
    fn 能力画像_可JSON往返() {
        let cap = EngineCapability::profile(EngineKind::Kokoro).with_sample_rate(24000);
        let json = serde_json::to_string(&cap).unwrap();
        let back: EngineCapability = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cap);
    }

    #[test]
    fn 引擎显示名非空() {
        for engine in EngineKind::all() {
            assert!(
                !engine.display_name().is_empty(),
                "{:?} 缺少中文显示名",
                engine
            );
        }
    }
}
