//! 文本门面：编码探测与文档正文提取
//!
//! AGENTS.md 要求「输入文本编码支持 UTF-8、GBK（自动检测转换），输出统一 UTF-8」。

use std::path::Path;

/// 读取文本文件（自动探测 UTF-8 / GBK / UTF-16）
pub fn read_text_file(path: &Path) -> anyhow::Result<String> {
    Ok(votex_infra::encoding::detector::EncodingDetector::read_text_file(
        path,
    )?)
}

/// 探测字节序列的编码
pub fn detect_encoding(bytes: &[u8]) -> votex_domain::tts::value_object::FileEncoding {
    votex_infra::encoding::detector::EncodingDetector::detect(bytes)
}

/// 按探测结果解码为 UTF-8 字符串
pub fn decode_bytes(bytes: &[u8]) -> Option<String> {
    votex_infra::encoding::detector::EncodingDetector::decode(bytes)
}

/// 提取文档正文（TXT / MD / EPUB / DOCX / PDF）
pub fn extract_document_text(path: &Path) -> anyhow::Result<String> {
    votex_infra::document::extract_text(path)
}

/// 读取输入文本：路径存在则按文档提取，否则视为内联文本
///
/// CLI 与 GUI 共用，避免两边对「输入是路径还是字面量」的判断出现分歧。
pub fn load_input_text(input: &str) -> anyhow::Result<String> {
    let p = Path::new(input);
    if p.exists() {
        extract_document_text(p)
    } else {
        Ok(input.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn 内联文本原样返回() {
        assert_eq!(
            load_input_text("你好，这是一段内联文本").unwrap(),
            "你好，这是一段内联文本"
        );
    }

    #[test]
    fn 存在的txt文件走文档提取() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        fs::write(&path, "第一行\n第二行").unwrap();

        let text = load_input_text(path.to_str().unwrap()).unwrap();
        assert!(text.contains("第一行"));
        assert!(text.contains("第二行"));
    }

    #[test]
    fn gbk文本被正确解码() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gbk.txt");
        // “中文”的 GBK 编码
        fs::write(&path, [0xD6, 0xD0, 0xCE, 0xC4]).unwrap();

        let text = read_text_file(&path).unwrap();
        assert_eq!(text, "中文");
    }
}
