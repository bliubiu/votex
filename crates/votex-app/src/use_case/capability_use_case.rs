//! 能力自描述用例
//!
//! 汇总「这台机器现在能做什么」，供 CLI（`model list --json`）、
//! 本地服务（`votex serve`）与 Agent（MCP）统一消费：
//!
//! - **models**：registry 模型清单 + 本地文件是否就绪
//! - **engines**：全部引擎能力画像（静态画像 + provider 运行期覆盖采样率/方言）
//! - **voices**：引擎内置音色池（磁盘扫描，不加载模型）
//! - **clone_voices**：克隆音色库条目
//!
//! 不加载任何模型，可安全地在启动时或每次请求时调用。

use std::path::Path;

use serde::{Deserialize, Serialize};
use votex_domain::model::capability::{all_engine_capabilities, EngineCapability};
use votex_domain::model::registry::ModelRegistryEntry;
use votex_domain::model::value_object::ModelStatus;
use votex_domain::tts::value_object::VoiceMeta;

use crate::use_case::asr_use_case::AsrUseCase;
use crate::use_case::model_use_case::ModelUseCase;
use crate::use_case::tts_use_case::TtsUseCase;

/// 模型能力条目（registry 驱动 + 本地可用性）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapability {
    pub id: String,
    pub name: String,
    /// 模型类型：Tts / Asr / Ocr / Translation / Runtime
    pub kind: String,
    /// 引擎标识
    pub engine: String,
    /// 引擎中文显示名
    pub engine_display_name: String,
    /// 本地文件是否就绪（required 文件齐全）
    pub available: bool,
    /// 状态机标签（稳定英文串，便于脚本判断）
    pub status: String,
}

/// 克隆音色能力条目（隔离 infra 的 `CloneVoiceMeta`，只暴露稳定字段）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloneVoiceCapability {
    pub name: String,
    pub duration_ms: u64,
    pub denoised: bool,
    pub sample_rate: u32,
    /// 是否带参考文本（CosyVoice 零样本克隆必需）
    pub has_transcript: bool,
}

/// 能力报告（可序列化）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityReport {
    /// 应用版本
    pub version: String,
    pub models: Vec<ModelCapability>,
    pub engines: Vec<EngineCapability>,
    pub voices: Vec<VoiceMeta>,
    pub clone_voices: Vec<CloneVoiceCapability>,
}

/// 能力自描述用例
pub struct CapabilityUseCase;

impl CapabilityUseCase {
    /// 构建能力报告
    ///
    /// `models_dir`：模型根目录；`registry`：模型清单（调用方经
    /// `votex_app::platform::registry::load_registry_entries` 加载）。
    pub fn build(models_dir: &Path, registry: Vec<ModelRegistryEntry>) -> CapabilityReport {
        let models = ModelUseCase::new(models_dir, registry)
            .list_models()
            .into_iter()
            .map(|m| ModelCapability {
                id: m.id.as_str().to_string(),
                name: m.name.clone(),
                kind: m.kind.display_name().to_string(),
                engine: m.engine.as_str().to_string(),
                engine_display_name: m.engine.display_name().to_string(),
                available: m.is_ready(),
                status: status_label(&m.status).to_string(),
            })
            .collect();

        // 引擎能力：静态画像打底，provider 覆盖采样率与方言
        let mut engines = all_engine_capabilities();
        for cap in TtsUseCase::new()
            .capabilities()
            .into_iter()
            .chain(AsrUseCase::new().capabilities())
        {
            if let Some(slot) = engines.iter_mut().find(|e| e.engine == cap.engine) {
                *slot = cap;
            }
        }

        // 音色池（磁盘扫描，不加载模型）
        let voices = crate::platform::tts::list_engine_voices(models_dir);
        let clone_voices = crate::platform::tts::list_voices()
            .into_iter()
            .map(|v| CloneVoiceCapability {
                name: v.name,
                duration_ms: v.duration_ms,
                denoised: v.denoised,
                sample_rate: v.sample_rate,
                has_transcript: v.transcript.is_some(),
            })
            .collect();

        CapabilityReport {
            version: env!("CARGO_PKG_VERSION").to_string(),
            models,
            engines,
            voices,
            clone_voices,
        }
    }
}

/// 模型状态 → 稳定英文标签
fn status_label(status: &ModelStatus) -> &'static str {
    match status {
        ModelStatus::NotDownloaded => "not_downloaded",
        ModelStatus::Downloading(_) => "downloading",
        ModelStatus::DownloadPaused => "paused",
        ModelStatus::Verifying => "verifying",
        ModelStatus::VerifyFailed => "verify_failed",
        ModelStatus::Ready => "ready",
        ModelStatus::Loading => "loading",
        ModelStatus::Loaded => "loaded",
        ModelStatus::LoadFailed(_) => "load_failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 能力报告_空registry与空缺模型目录可构建() {
        let dir = tempfile::tempdir().unwrap();
        let report = CapabilityUseCase::build(dir.path(), Vec::new());
        assert!(report.models.is_empty());
        // 引擎能力在无模型时依然完整（静态画像 + provider 常量）
        assert!(report.engines.iter().any(|e| e.engine == "kokoro"));
        assert!(report.engines.iter().any(|e| e.engine == "whisper"));
        // 无模型时音色池为空，但字段存在
        assert!(report.voices.is_empty());
    }

    #[test]
    fn 能力报告_引擎含运行期采样率与方言() {
        let dir = tempfile::tempdir().unwrap();
        let report = CapabilityUseCase::build(dir.path(), Vec::new());

        let kokoro = report.engines.iter().find(|e| e.engine == "kokoro").unwrap();
        assert_eq!(kokoro.sample_rate, Some(24000));
        assert!(kokoro.local);
        assert_eq!(kokoro.category, "tts");

        // IndexTTS-2.5 众方言由 provider 声明
        let indextts = report
            .engines
            .iter()
            .find(|e| e.engine == "indextts25")
            .unwrap();
        assert!(indextts.zero_shot_clone);
    }

    #[test]
    fn 能力报告_JSON可序列化() {
        let dir = tempfile::tempdir().unwrap();
        let report = CapabilityUseCase::build(dir.path(), Vec::new());
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"engines\""));
        assert!(json.contains("\"kokoro\""));
        let _: serde_json::Value = serde_json::from_str(&json).unwrap();
    }
}
