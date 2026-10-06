//! 推理引擎加载：ONNX 模型文件定位与前置检查。
//!
//! # 候选文件名从哪来
//!
//! 早期实现把 Kokoro 的模型名（`kokoro-v1.0.int8.onnx` 等）硬编码在
//! `find_onnx_file` 里，其他引擎只能走「目录内任意 `.onnx`」的兜底分支。
//! 兜底分支还有两个问题：
//!
//! 1. **顺序不确定**——`read_dir` 的返回顺序未定义，
//!    目录里有多个 `.onnx` 时每次可能选到不同的文件。
//! 2. **静默降级**——选中哪个文件不告知用户，引擎加载失败时
//!    排查方向被误导（以为是权重损坏，实际是选错了文件）。
//!
//! 现在改为：
//!
//! - 候选名按 [`EngineKind`] 分组（[`onnx_candidates`]），每个引擎有自己的优先序
//! - 兜底分支**按文件名排序**后取首个，结果确定
//! - 走到兜底分支时打 `warn` 日志，说明选中了非预期文件
//!
//! # 与模型清单的关系
//!
//! 清单（`models/registry/*.yaml`）里的 `files[].name` 是**下载与校验**的依据；
//! 本模块的候选表是**运行时定位**的依据。两者可能不一致
//! （例如清单里有多个量化版本，运行时按引擎优先级选其一）。
//! 彻底统一需在 registry 里增加「运行时主文件」声明字段，
//! 当前以本表 + warn 日志作为过渡。

use std::path::{Path, PathBuf};

use anyhow::Result;
use votex_domain::model::value_object::EngineKind;

/// 推理引擎加载器（统一使用 ONNX Runtime）
pub struct EngineLoader;

impl EngineLoader {
    /// 根据引擎类型加载模型（统一 ONNX 推理）
    pub fn load(model_dir: &Path, engine: EngineKind) -> Result<()> {
        Self::load_onnx_for(model_dir, engine)
    }

    /// 加载 ONNX 模型（不区分引擎，跨引擎候选表）
    pub fn load_onnx(model_dir: &Path) -> Result<()> {
        Self::check_dir(model_dir)?;
        let onnx_file = Self::find_onnx_file(model_dir, &onnx_candidates(None))?;
        tracing::info!("ONNX 模型文件检查通过: {:?}", onnx_file);
        Ok(())
    }

    /// 按引擎类型定位并检查 ONNX 模型
    pub fn load_onnx_for(model_dir: &Path, engine: EngineKind) -> Result<()> {
        Self::check_dir(model_dir)?;
        let onnx_file = Self::find_onnx_file(model_dir, &onnx_candidates(Some(engine)))?;
        tracing::info!(
            "ONNX 模型文件检查通过（引擎 {:?}）: {:?}",
            engine,
            onnx_file
        );
        Ok(())
    }

    fn check_dir(model_dir: &Path) -> Result<()> {
        if !model_dir.exists() {
            anyhow::bail!("模型目录不存在: {:?}", model_dir);
        }
        if !model_dir.is_dir() {
            anyhow::bail!("模型路径不是目录: {:?}", model_dir);
        }
        Ok(())
    }

    /// 在目录中按候选顺序查找 ONNX 文件
    ///
    /// 找不到候选名时回退到「目录内任意 `.onnx`」，
    /// 但必须**排序后取首个**（`read_dir` 顺序未定义），
    /// 并打 warn 说明这是兜底选择。
    pub fn find_onnx_file(model_dir: &Path, candidates: &[&str]) -> Result<PathBuf> {
        for name in candidates {
            let path = model_dir.join(name);
            if path.is_file() {
                return Ok(path);
            }
        }

        // 兜底：收集全部 .onnx，按文件名排序后取首个，保证结果确定
        let mut fallback: Vec<PathBuf> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|x| x.to_str())
                        .is_some_and(|x| x.eq_ignore_ascii_case("onnx"))
            })
            .collect();
        fallback.sort();

        if let Some(first) = fallback.first() {
            tracing::warn!(
                "未匹配到 {} 的预期 ONNX 文件名，回退选择: {:?}（目录内共 {} 个 .onnx，按文件名排序取首个）",
                model_dir.display(),
                first.file_name().unwrap_or_default(),
                fallback.len()
            );
            return Ok(first.clone());
        }

        anyhow::bail!(
            "未找到 ONNX 模型文件: {}（该目录下没有任何 .onnx 文件）",
            model_dir.display()
        )
    }
}

/// 按引擎类型返回 ONNX 候选文件名（按优先级排列）
///
/// `engine` 为 `None` 时返回跨引擎的通用候选表，
/// 用于不关心具体引擎的场景（如仅做前置检查）。
pub fn onnx_candidates(engine: Option<EngineKind>) -> Vec<&'static str> {
    match engine {
        Some(EngineKind::Kokoro) => vec![
            // int8 体积最小、推理最快，优先
            "kokoro-v1.0.int8.onnx",
            "kokoro-v1.0.fp16.onnx",
            "kokoro-v1.0.onnx",
            "kokoro-v1.1-zh.onnx",
            "kokoro-v1.1-zh.fixed.onnx",
        ],
        // IndexTTS-2.5：分图布局（registry: indextts-2.5-onnx.yaml，16 文件），
        // 与 indextts25::engines::load 的目录布局逐字一致
        Some(EngineKind::IndexTTS25) => vec![
            "gpt_prefill/gpt_prefill.onnx",
            "gpt_step/gpt_step.onnx",
            "cfm_estimator/cfm_estimator.onnx",
            "semantic_model/semantic_model.onnx",
            "bigvgan.onnx",
            "emo_vec.onnx",
            "campplus.onnx",
            "length_regulator.onnx",
            "semantic_codec_decode.onnx",
        ],
        Some(EngineKind::CosyVoice3) => vec![
            "llm.onnx",
            "flow.onnx",
            "hift.onnx",
            "speech_tokenizer_v3.onnx",
            "campplus.onnx",
        ],
        Some(EngineKind::Qwen3Tts) => vec![
            "talker_prefill.onnx",
            "talker_decode.onnx",
            "code_predictor.onnx",
            "vocoder.onnx",
        ],
        Some(EngineKind::PaddleOCR) => vec![
            "det.onnx",
            "rec.onnx",
            "cls.onnx",
            "det_model.onnx",
            "rec_model.onnx",
            "cls_model.onnx",
        ],
        Some(EngineKind::Whisper)
        | Some(EngineKind::SenseVoice)
        | Some(EngineKind::Paraformer)
        | Some(EngineKind::Qwen3Asr)
        | Some(EngineKind::FireRedAsr) => vec![
            "model.onnx",
            "model_int8.onnx",
            "model_quant.onnx",
            "model_quantized.onnx",
            "encoder_model.onnx",
            "decoder_model.onnx",
        ],
        // 云端引擎与其他：走通用表
        _ => vec![
            "model.onnx",
            "model_quant.onnx",
            "model_quantized.onnx",
            "encoder_model.onnx",
            "decoder_model.onnx",
            "bigvgan.onnx",
            "speaker_encoder.onnx",
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str) {
        std::fs::write(dir.join(name), b"fake onnx content").unwrap();
    }

    // ===== 目录检查 =====

    #[test]
    fn load_onnx_目录不存在应报错() {
        let result = EngineLoader::load_onnx(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }

    #[test]
    fn load_onnx_路径是文件应报错() {
        // 传入文件而非目录时，原实现只判 exists() 会漏过
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("model.onnx");
        std::fs::write(&f, b"x").unwrap();

        let err = EngineLoader::load_onnx(&f).unwrap_err();
        assert!(
            err.to_string().contains("不是目录"),
            "应提示路径不是目录，实际: {}",
            err
        );
    }

    #[test]
    fn load_onnx_空目录应报错且信息可操作() {
        let dir = tempfile::tempdir().unwrap();
        let err = EngineLoader::load_onnx(dir.path()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("未找到"), "实际: {}", msg);
        // 错误信息应说明目录里一个 .onnx 都没有，而不是笼统的「未找到」
        assert!(
            msg.contains("没有任何 .onnx"),
            "错误信息应指出目录内无任何 onnx，实际: {}",
            msg
        );
    }

    // ===== 候选命中 =====

    #[test]
    fn load_onnx_模型文件存在时通过() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "model.onnx");
        assert!(EngineLoader::load_onnx(dir.path()).is_ok());
    }

    #[test]
    fn load_onnx_自动查找已知文件名() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "kokoro-v1.0.int8.onnx");
        assert!(EngineLoader::load_onnx(dir.path()).is_ok());
    }

    #[test]
    fn load_按引擎类型分发() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "kokoro-v1.0.int8.onnx");
        assert!(EngineLoader::load(dir.path(), EngineKind::Kokoro).is_ok());
    }

    // ===== 新增：候选优先级 =====

    #[test]
    fn kokoro_优先选int8版本() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "kokoro-v1.0.onnx");
        write(dir.path(), "kokoro-v1.0.int8.onnx");

        let picked = EngineLoader::find_onnx_file(
            dir.path(),
            &onnx_candidates(Some(EngineKind::Kokoro)),
        )
        .unwrap();
        assert_eq!(
            picked.file_name().unwrap(),
            "kokoro-v1.0.int8.onnx",
            "int8 体积最小，应优先于完整精度版本"
        );
    }

    #[test]
    fn paddleocr_优先选det模型() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "rec.onnx");
        write(dir.path(), "det.onnx");

        let picked = EngineLoader::find_onnx_file(
            dir.path(),
            &onnx_candidates(Some(EngineKind::PaddleOCR)),
        )
        .unwrap();
        assert_eq!(picked.file_name().unwrap(), "det.onnx");
    }

    // ===== 新增：兜底确定性 =====

    #[test]
    fn 兜底选择按文件名排序结果确定() {
        // 原实现直接返回 read_dir 首个结果，顺序未定义；
        // 多文件目录下可能每次选中不同文件，导致加载行为随机。
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "zzz.onnx");
        write(dir.path(), "aaa.onnx");
        write(dir.path(), "mmm.onnx");

        let cands = onnx_candidates(Some(EngineKind::Kokoro));
        let first = EngineLoader::find_onnx_file(dir.path(), &cands).unwrap();
        for _ in 0..20 {
            let again = EngineLoader::find_onnx_file(dir.path(), &cands).unwrap();
            assert_eq!(first, again, "兜底选择必须稳定");
        }
        assert_eq!(
            first.file_name().unwrap(),
            "aaa.onnx",
            "应取文件名序最小者"
        );
    }

    #[test]
    fn 兜底忽略大小写扩展名与子目录() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "model.ONNX");
        std::fs::create_dir(dir.path().join("nested.onnx")).unwrap();
        std::fs::write(dir.path().join("notes.txt"), b"x").unwrap();

        // 传空候选表，强制走兜底分支
        let picked = EngineLoader::find_onnx_file(dir.path(), &[]).unwrap();
        assert_eq!(
            picked.file_name().unwrap(),
            "model.ONNX",
            ".ONNX 大写扩展名应被识别，子目录与 .txt 应被排除"
        );
    }

    #[test]
    fn 候选全部缺失时回退到任意onnx() {
        // Whisper 的候选名与实际文件名不符时应能兜底，
        // 而不是直接失败——清单与运行时命名不一致是常态。
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "my_custom_export.onnx");

        let picked = EngineLoader::find_onnx_file(
            dir.path(),
            &onnx_candidates(Some(EngineKind::Whisper)),
        )
        .unwrap();
        assert_eq!(picked.file_name().unwrap(), "my_custom_export.onnx");
    }

    #[test]
    fn 候选表按引擎区分() {
        // 回归守护：Kokoro 候选不应包含 PaddleOCR 的文件名
        let kokoro = onnx_candidates(Some(EngineKind::Kokoro));
        assert!(kokoro.iter().all(|n| !n.starts_with("det")));
        let paddle = onnx_candidates(Some(EngineKind::PaddleOCR));
        assert!(paddle.contains(&"det.onnx"));
    }
}
