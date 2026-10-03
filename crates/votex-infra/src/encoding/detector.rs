/// 文本编码检测
pub struct EncodingDetector;

impl EncodingDetector {
    /// 读取文本文件，自动检测编码（UTF-8 / GBK）并转为 UTF-8 字符串
    ///
    /// 供 CLI/GUI 所有文本入口使用，替代裸 `fs::read_to_string`
    /// （后者遇到 GBK 文件直接报 UTF-8 校验错误）。
    pub fn read_text_file(path: &std::path::Path) -> anyhow::Result<String> {
        let bytes = std::fs::read(path)?;
        match Self::decode(&bytes) {
            Some(text) => Ok(text),
            None => {
                let head_hex: String = bytes
                    .iter()
                    .take(16)
                    .map(|b| format!("{:02x}", b))
                    .collect::<Vec<_>>()
                    .join(" ");
                anyhow::bail!(
                    "无法识别文件编码（已尝试 UTF-8 / GBK）: {}，文件头字节: {}",
                    path.display(),
                    head_hex
                )
            }
        }
    }

    /// 检测文本编码（UTF-8 / GBK）
    pub fn detect(bytes: &[u8]) -> votex_domain::tts::value_object::FileEncoding {
        // 先尝试 UTF-8
        if Self::is_valid_utf8(bytes) {
            return votex_domain::tts::value_object::FileEncoding::Utf8;
        }

        // 再尝试 GBK
        let (cow, _, had_errors) = encoding_rs::GBK.decode(bytes);
        if !had_errors && !cow.is_empty() {
            return votex_domain::tts::value_object::FileEncoding::Gbk;
        }

        votex_domain::tts::value_object::FileEncoding::Unknown
    }

    /// 检查是否为有效 UTF-8
    fn is_valid_utf8(bytes: &[u8]) -> bool {
        std::str::from_utf8(bytes).is_ok()
    }

    /// 将字节解码为 String
    pub fn decode(bytes: &[u8]) -> Option<String> {
        let encoding = Self::detect(bytes);
        match encoding {
            votex_domain::tts::value_object::FileEncoding::Utf8 => {
                std::str::from_utf8(bytes).ok().map(|s| s.to_string())
            }
            votex_domain::tts::value_object::FileEncoding::Gbk => {
                let (cow, _, had_errors) = encoding_rs::GBK.decode(bytes);
                if had_errors {
                    None
                } else {
                    Some(cow.to_string())
                }
            }
            votex_domain::tts::value_object::FileEncoding::Unknown => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_detector_检测utf8() {
        let text = "你好世界";
        let bytes = text.as_bytes();
        assert_eq!(
            EncodingDetector::detect(bytes),
            votex_domain::tts::value_object::FileEncoding::Utf8
        );
    }

    #[test]
    fn encoding_detector_检测gbk() {
        let text = "你好世界";
        let (bytes, _, _) = encoding_rs::GBK.encode(text);
        assert_eq!(
            EncodingDetector::detect(&bytes),
            votex_domain::tts::value_object::FileEncoding::Gbk
        );
    }

    #[test]
    fn encoding_detector_解码utf8() {
        let text = "你好世界";
        let bytes = text.as_bytes();
        let decoded = EncodingDetector::decode(bytes);
        assert_eq!(decoded, Some("你好世界".to_string()));
    }

    #[test]
    fn encoding_detector_解码gbk() {
        let text = "你好世界";
        let (bytes, _, _) = encoding_rs::GBK.encode(text);
        let decoded = EncodingDetector::decode(&bytes);
        assert_eq!(decoded, Some("你好世界".to_string()));
    }
}
