use serde::{Deserialize, Serialize};
use std::fmt;

/// 翻译方向
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranslationDirection {
    /// 中文 → 英文
    ZhToEn,
    /// 英文 → 中文
    EnToZh,
    /// 自动检测
    Auto,
    /// 指定语言对（源语言 → 目标语言），如 ByLanguagePair("zh", "en")
    ByLanguagePair {
        source: String,
        target: String,
    },
}

impl TranslationDirection {
    /// 返回标准化的方向标识
    pub fn as_str(&self) -> String {
        match self {
            TranslationDirection::ZhToEn => "zh-en".to_string(),
            TranslationDirection::EnToZh => "en-zh".to_string(),
            TranslationDirection::Auto => "auto".to_string(),
            TranslationDirection::ByLanguagePair { source, target } => {
                format!("{}-{}", source, target)
            }
        }
    }

    /// 返回目标语言代码（用于 M2M-100 / HY-MT1.5 等模型）
    pub fn target_lang(&self) -> Option<&str> {
        match self {
            TranslationDirection::ZhToEn => Some("en"),
            TranslationDirection::EnToZh => Some("zh"),
            TranslationDirection::Auto => None,
            TranslationDirection::ByLanguagePair { target, .. } => Some(target.as_str()),
        }
    }

    /// 返回源语言代码
    pub fn source_lang(&self) -> Option<&str> {
        match self {
            TranslationDirection::ZhToEn => Some("zh"),
            TranslationDirection::EnToZh => Some("en"),
            TranslationDirection::Auto => None,
            TranslationDirection::ByLanguagePair { source, .. } => Some(source.as_str()),
        }
    }

    /// 从字符串解析翻译方向
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "zh-en" => Some(TranslationDirection::ZhToEn),
            "en-zh" => Some(TranslationDirection::EnToZh),
            "auto" => Some(TranslationDirection::Auto),
            _ => {
                let parts: Vec<&str> = s.splitn(2, '-').collect();
                if parts.len() == 2 && parts[0].len() <= 5 && parts[1].len() <= 5 {
                    Some(TranslationDirection::ByLanguagePair {
                        source: parts[0].to_string(),
                        target: parts[1].to_string(),
                    })
                } else {
                    None
                }
            }
        }
    }
}

impl fmt::Display for TranslationDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
