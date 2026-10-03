use anyhow::Result;
use std::path::Path;
use votex_domain::model::value_object::EngineKind;

/// 推理引擎加载器（统一使用 ONNX Runtime）
pub struct EngineLoader;

impl EngineLoader {
    /// 根据引擎类型加载模型（统一 ONNX 推理）
    pub fn load(model_dir: &Path, _engine: EngineKind) -> Result<()> {
        Self::load_onnx(model_dir)
    }

    /// 加载 ONNX 模型
    pub fn load_onnx(model_dir: &Path) -> Result<()> {
        if !model_dir.exists() {
            anyhow::bail!("模型目录不存在: {:?}", model_dir);
        }

        let onnx_file = Self::find_onnx_file(model_dir)?;
        tracing::info!("ONNX 模型文件检查通过: {:?}", onnx_file);

        Ok(())
    }

    /// 在目录中查找 ONNX 文件
    fn find_onnx_file(model_dir: &Path) -> Result<std::path::PathBuf> {
        // 优先查找已知文件名
        let candidates = [
            "kokoro-v1.0.int8.onnx",
            "kokoro-v1.0.fp16.onnx",
            "kokoro-v1.0.onnx",
            "kokoro-v1.1-zh.onnx",
            "kokoro-v1.1-zh.fixed.onnx",
            "bigvgan.onnx",
            "speaker_encoder.onnx",
            "model_quantized.onnx",
            "model_quant.onnx",
            "model.onnx",
            "encoder_model.onnx",
            "decoder_model.onnx",
        ];
        for name in &candidates {
            let path = model_dir.join(name);
            if path.exists() {
                return Ok(path);
            }
        }

        // 回退：查找目录中任意 .onnx 文件
        for entry in std::fs::read_dir(model_dir)? {
            let entry = entry?;
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".onnx") {
                    return Ok(entry.path());
                }
            }
        }

        anyhow::bail!("未找到 ONNX 模型文件: {:?}", model_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_onnx_目录不存在应报错() {
        let result = EngineLoader::load_onnx(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }

    #[test]
    fn load_onnx_模型文件存在时通过() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.onnx");
        std::fs::write(&model_file, b"fake onnx content").unwrap();
        assert!(EngineLoader::load_onnx(dir.path()).is_ok());
    }

    #[test]
    fn load_onnx_自动查找已知文件名() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("kokoro-v1.0.int8.onnx");
        std::fs::write(&model_file, b"fake onnx content").unwrap();
        assert!(EngineLoader::load_onnx(dir.path()).is_ok());
    }

    #[test]
    fn load_按引擎类型分发() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("kokoro-v1.0.int8.onnx");
        std::fs::write(&model_file, b"fake onnx content").unwrap();
        assert!(EngineLoader::load(dir.path(), EngineKind::Kokoro).is_ok());
    }

    #[test]
    fn load_indextts2_onnx模型直接加载() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("bigvgan.onnx");
        std::fs::write(&model_file, b"fake onnx content").unwrap();
        assert!(EngineLoader::load(dir.path(), EngineKind::IndexTTS2).is_ok());
    }
}
