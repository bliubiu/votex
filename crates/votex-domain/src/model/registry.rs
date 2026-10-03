use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 分词器配置（可选，仅需特殊处理的模型需要声明）
///
/// 不同模型来源（HuggingFace、魔塔社区、hf-mirror）的 tokenizer
/// 可能有不同的 ID 映射（如 NLLB 的 +1 偏移），在此声明后可
/// 由 SentencePieceBpe 统一处理，无需每个 provider 硬编码。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenizerConfig {
    /// 分词器类型：sentencepiece | huggingface_fast | bpe
    #[serde(rename = "type")]
    pub tokenizer_type: Option<String>,
    /// 内容 token ID 偏移量（如 NLLB 的原始 SP ID → HF 映射空间需要 +1）
    #[serde(default)]
    pub id_offset: i64,
    /// 编码器输入末尾是否自动追加 EOS
    #[serde(default)]
    pub eos_on_input: bool,
    /// 解码器起始 token（设为 eos 表示使用 EOS token）
    pub decoder_start_token: Option<String>,
    /// added_tokens 的 ID 值（如 FLORES-200 语言代码）
    #[serde(default)]
    pub added_tokens: HashMap<String, i64>,
}

/// 模型清单条目，对应 `models/registry/{id}.yaml`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistryEntry {
    /// 模型唯一标识（也是默认子目录名）
    pub id: String,
    /// 显示名称
    pub name: String,
    /// 模型类型：Tts / Asr / Ocr / Translation
    pub kind: String,
    /// 引擎名：Kokoro / IndexTTS2 / Whisper / PaddleOCR / EasyOcr / SenseVoice / Nllb
    pub engine: String,
    /// 存储子目录（覆盖 id），如 paddleocr-v5-mobile → "paddleocr-v5"
    #[serde(default)]
    pub sub_dir: Option<String>,
    /// 分词器配置（可选，默认无偏移）
    #[serde(default)]
    pub tokenizer: Option<TokenizerConfig>,
    /// 该模型所需下载的文件列表
    pub files: Vec<ModelFileEntry>,
}

/// 单个模型文件条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFileEntry {
    /// 相对于模型子目录的路径（如 "kokoro-v1.0.int8.onnx" 或 "voices/af_heart.bin"）
    pub name: String,
    /// 可选的 SHA256 校验值
    #[serde(default)]
    pub sha256: Option<String>,
    /// 预期文件大小（字节），用于校验下载完整性
    /// 设置后下载器会在断点续传前校验本地文件大小是否匹配
    #[serde(default)]
    pub size: Option<u64>,
    /// 是否为必需文件（模型就绪检测时，所有 required 文件必须存在）
    /// 用于 embeddings 等非 ONNX 但引擎启动必需的附属文件
    #[serde(default)]
    pub required: bool,
    /// 镜像源列表，键为镜像名（如 "huggingface"），值为下载 URL
    pub sources: HashMap<String, String>,
}

impl ModelRegistryEntry {
    /// 根据 kind 获取分类目录名
    fn kind_dir(&self) -> &str {
        match self.kind.as_str() {
            "Tts" => "tts",
            "Asr" => "asr",
            "Ocr" => "ocr",
            "Translation" => "translation",
            _ => "other",
        }
    }

    /// 获取完整存储路径（自动按 kind 分类到 tts/asr/ocr/translation 子目录）
    pub fn storage_dir(&self) -> String {
        let leaf = self.sub_dir.as_deref().unwrap_or(&self.id);
        format!("{}/{}", self.kind_dir(), leaf)
    }

    /// 获取 kind 字段的显示名
    pub fn kind_display(&self) -> &str {
        self.kind.as_str()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_entry_storage_dir_使用sub_dir() {
        let entry = ModelRegistryEntry {
            id: "paddleocr-v6-tiny".into(),
            name: "PaddleOCR v6 Tiny".into(),
            kind: "Ocr".into(),
            engine: "PaddleOCR".into(),
            sub_dir: Some("paddleocr-v6".into()),
            tokenizer: None,
            files: vec![],
        };
        assert_eq!(entry.storage_dir(), "ocr/paddleocr-v6");
    }

    #[test]
    fn registry_entry_storage_dir_默认等于id() {
        let entry = ModelRegistryEntry {
            id: "kokoro-82m".into(),
            name: "Kokoro-82M".into(),
            kind: "Tts".into(),
            engine: "Kokoro".into(),
            sub_dir: None,
            tokenizer: None,
            files: vec![],
        };
        assert_eq!(entry.storage_dir(), "tts/kokoro-82m");
    }

    #[test]
    fn registry_entry_storage_dir_translation() {
        let entry = ModelRegistryEntry {
            id: "nllb-200-distilled-600m".into(),
            name: "NLLB-200 Distilled 600M".into(),
            kind: "Translation".into(),
            engine: "Nllb".into(),
            sub_dir: None,
            tokenizer: None,
            files: vec![],
        };
        assert_eq!(entry.storage_dir(), "translation/nllb-200-distilled-600m");
    }

    #[test]
    fn registry_entry_storage_dir_asr() {
        let entry = ModelRegistryEntry {
            id: "whisper-base".into(),
            name: "Whisper Base".into(),
            kind: "Asr".into(),
            engine: "Whisper".into(),
            sub_dir: None,
            tokenizer: None,
            files: vec![],
        };
        assert_eq!(entry.storage_dir(), "asr/whisper-base");
    }

    #[test]
    fn registry_entry_yaml_序列化反序列化() {
        let yaml = r#"
id: whisper-base
name: Whisper Base
kind: Asr
engine: Whisper
files:
  - name: encoder.onnx
    sources:
      huggingface: https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base/resolve/main/encoder.onnx
      hf-mirror: https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-base/resolve/main/encoder.onnx
  - name: decoder.onnx
    sources:
      huggingface: https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base/resolve/main/decoder.onnx
      hf-mirror: https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-base/resolve/main/decoder.onnx
  - name: tokens.txt
    sources:
      huggingface: https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base/resolve/main/tokens.txt
      hf-mirror: https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-base/resolve/main/tokens.txt
"#;
        let entry: ModelRegistryEntry = serde_yml::from_str(yaml).unwrap();
        assert_eq!(entry.id, "whisper-base");
        assert_eq!(entry.storage_dir(), "asr/whisper-base");
        assert_eq!(entry.files.len(), 3);
        assert_eq!(entry.files[0].sources.len(), 2);
        assert!(entry.files[0].sources.contains_key("huggingface"));
    }

    #[test]
    fn registry_entry_sub_dir_yaml() {
        let yaml = r#"
id: easyocr
name: EasyOCR
kind: Ocr
engine: EasyOcr
sub_dir: EasyOCR
files:
  - name: craft_det.onnx
    sources:
      huggingface: https://huggingface.co/itextresearch/itext-EasyOCR-craft_mlt_25k/resolve/main/itext-EasyOCR-craft_mlt_25k.onnx
"#;
        let entry: ModelRegistryEntry = serde_yml::from_str(yaml).unwrap();
        assert_eq!(entry.storage_dir(), "ocr/EasyOCR");
    }

    #[test]
    fn registry_entry_sha256_可选() {
        let yaml = r#"
id: test
name: Test
kind: Tts
engine: Test
files:
  - name: test.onnx
    sha256: a1b2c3d4
    sources:
      huggingface: https://example.com/test.onnx
"#;
        let entry: ModelRegistryEntry = serde_yml::from_str(yaml).unwrap();
        assert_eq!(entry.files[0].sha256.as_deref(), Some("a1b2c3d4"));
    }
}
