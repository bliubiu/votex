//! 方言元数据模块
//!
//! 定义 TTS 引擎支持的方言类型以及方言支持等级。
//! 参考 PilotTTS 14 种中文方言分类体系 (2026-05)。

use serde::{Deserialize, Serialize};

// ============================================================
// 方言类型（14 种中文方言，参考 PilotTTS）
// ============================================================

/// 中文方言
///
/// 基于 ISO 693-3 语言代码和 PilotTTS 方言分类，涵盖 14 种主要中文方言/语言。
///
/// | # | 方言 | ISO 代码 | 主要分布区域 |
/// |---|------|---------|-------------|
/// | 1 | 普通话 | cmn | 全国通用 |
/// | 2 | 粤语 | yue | 广东、香港、澳门 |
/// | 3 | 闽南语 | nan | 福建南部、台湾、潮汕 |
/// | 4 | 吴语 | wuu | 上海、江苏南部、浙江 |
/// | 5 | 客家话 | hak | 广东东部、福建西部、江西南部 |
/// | 6 | 湘语 | hsn | 湖南（长沙等） |
/// | 7 | 赣语 | gan | 江西（南昌等） |
/// | 8 | 晋语 | cjy | 山西、陕西北部、内蒙古 |
/// | 9 | 闽北语 | mnp | 福建北部（建瓯等） |
/// | 10 | 闽东语 | cdo | 福建东部（福州等） |
/// | 11 | 莆仙语 | cpx | 福建莆田、仙游 |
/// | 12 | 闽中语 | czo | 福建中部（三明等） |
/// | 13 | 徽语 | czh | 安徽南部、江西东北部 |
/// | 14 | 平话 | (无) | 广西部分地区 |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dialect {
    /// 普通话
    Mandarin,
    /// 粤语（广东话）
    Cantonese,
    /// 闽南语（福建南部/台湾/潮汕）
    SouthernMin,
    /// 吴语（上海话/苏州话/浙江话）
    Wu,
    /// 客家话
    Hakka,
    /// 湘语（湖南话）
    Xiang,
    /// 赣语（江西话）
    Gan,
    /// 晋语（山西话）
    Jin,
    /// 闽北语
    NorthernMin,
    /// 闽东语（福州话）
    EasternMin,
    /// 莆仙语（莆田话）
    Puxian,
    /// 闽中语
    CentralMin,
    /// 徽语
    Huizhou,
    /// 平话（广西平话）
    Pinghua,
}

/// 方言语族分组
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DialectFamily {
    /// 官话（包括普通话、晋语）
    Mandarin,
    /// 粤语
    Yue,
    /// 闽语（闽南、闽北、闽东、莆仙、闽中）
    Min,
    /// 吴语
    Wu,
    /// 客家话
    Hakka,
    /// 湘语
    Xiang,
    /// 赣语
    Gan,
    /// 徽语
    Huizhou,
    /// 平话
    Pinghua,
}

impl DialectFamily {
    pub fn display_name(&self) -> &'static str {
        match self {
            DialectFamily::Mandarin => "官话",
            DialectFamily::Yue => "粤语",
            DialectFamily::Min => "闽语",
            DialectFamily::Wu => "吴语",
            DialectFamily::Hakka => "客家话",
            DialectFamily::Xiang => "湘语",
            DialectFamily::Gan => "赣语",
            DialectFamily::Huizhou => "徽语",
            DialectFamily::Pinghua => "平话",
        }
    }
}

impl Dialect {
    /// 方言显示名称
    pub fn display_name(&self) -> &'static str {
        match self {
            Dialect::Mandarin => "普通话",
            Dialect::Cantonese => "粤语",
            Dialect::SouthernMin => "闽南语",
            Dialect::Wu => "吴语",
            Dialect::Hakka => "客家话",
            Dialect::Xiang => "湘语",
            Dialect::Gan => "赣语",
            Dialect::Jin => "晋语",
            Dialect::NorthernMin => "闽北语",
            Dialect::EasternMin => "闽东语",
            Dialect::Puxian => "莆仙语",
            Dialect::CentralMin => "闽中语",
            Dialect::Huizhou => "徽语",
            Dialect::Pinghua => "平话",
        }
    }

    /// 方言代码（ISO 693-3 兼容）
    pub fn code(&self) -> &'static str {
        match self {
            Dialect::Mandarin => "cmn",
            Dialect::Cantonese => "yue",
            Dialect::SouthernMin => "nan",
            Dialect::Wu => "wuu",
            Dialect::Hakka => "hak",
            Dialect::Xiang => "hsn",
            Dialect::Gan => "gan",
            Dialect::Jin => "cjy",
            Dialect::NorthernMin => "mnp",
            Dialect::EasternMin => "cdo",
            Dialect::Puxian => "cpx",
            Dialect::CentralMin => "czo",
            Dialect::Huizhou => "czh",
            Dialect::Pinghua => "pinghua",
        }
    }

    /// 所属语族
    pub fn family(&self) -> DialectFamily {
        match self {
            Dialect::Mandarin | Dialect::Jin => DialectFamily::Mandarin,
            Dialect::Cantonese => DialectFamily::Yue,
            Dialect::SouthernMin | Dialect::NorthernMin
                | Dialect::EasternMin | Dialect::Puxian
                | Dialect::CentralMin => DialectFamily::Min,
            Dialect::Wu => DialectFamily::Wu,
            Dialect::Hakka => DialectFamily::Hakka,
            Dialect::Xiang => DialectFamily::Xiang,
            Dialect::Gan => DialectFamily::Gan,
            Dialect::Huizhou => DialectFamily::Huizhou,
            Dialect::Pinghua => DialectFamily::Pinghua,
        }
    }

    /// 从代码解析方言
    pub fn from_code(code: &str) -> Option<Self> {
        match code.to_lowercase().as_str() {
            "cmn" | "mandarin" | "zh" | "普通话" => Some(Dialect::Mandarin),
            "yue" | "cantonese" | "粤语" | "广东话" => Some(Dialect::Cantonese),
            "nan" | "southernmin" | "闽南语" | "台语" => Some(Dialect::SouthernMin),
            "wuu" | "wu" | "吴语" | "上海话" => Some(Dialect::Wu),
            "hak" | "hakka" | "客家话" | "客语" => Some(Dialect::Hakka),
            "hsn" | "xiang" | "湘语" | "湖南话" => Some(Dialect::Xiang),
            "gan" | "赣语" | "江西话" => Some(Dialect::Gan),
            "cjy" | "jin" | "晋语" | "山西话" => Some(Dialect::Jin),
            "mnp" | "northernmin" | "闽北语" => Some(Dialect::NorthernMin),
            "cdo" | "easternmin" | "闽东语" | "福州话" => Some(Dialect::EasternMin),
            "cpx" | "puxian" | "莆仙语" | "莆田话" => Some(Dialect::Puxian),
            "czo" | "centralmin" | "闽中语" => Some(Dialect::CentralMin),
            "czh" | "huizhou" | "徽语" => Some(Dialect::Huizhou),
            "pinghua" | "平话" => Some(Dialect::Pinghua),
            _ => None,
        }
    }

    /// 列出所有方言
    pub fn all() -> Vec<Dialect> {
        vec![
            Dialect::Mandarin,
            Dialect::Cantonese,
            Dialect::SouthernMin,
            Dialect::Wu,
            Dialect::Hakka,
            Dialect::Xiang,
            Dialect::Gan,
            Dialect::Jin,
            Dialect::NorthernMin,
            Dialect::EasternMin,
            Dialect::Puxian,
            Dialect::CentralMin,
            Dialect::Huizhou,
            Dialect::Pinghua,
        ]
    }

    /// 某个语族下的所有方言
    pub fn by_family(family: DialectFamily) -> Vec<Dialect> {
        Dialect::all().into_iter().filter(|d| d.family() == family).collect()
    }

    /// 是否同属一个语族（用于跨方言合成兼容性判断）
    pub fn same_family_as(&self, other: &Dialect) -> bool {
        self.family() == other.family()
    }
}

// ============================================================
// 跨方言合成
// ============================================================

/// 方言合成方向
///
/// 用于跨方言合成：将源方言的文本风格，用目标方言朗读。
///
/// 例如：从普通话 -> 粤语，"我好开心"用粤语发音朗读。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DialectConversion {
    /// 源方言（文本内容的方言风格）
    pub source: Dialect,
    /// 目标方言（朗读发音的方言）
    pub target: Dialect,
}

impl DialectConversion {
    /// 创建合成方向
    ///
    /// # Panics
    /// 如果 source == target，可直接用单一方言，不需要转换。
    pub fn new(source: Dialect, target: Dialect) -> Self {
        Self { source, target }
    }

    /// 是否跨方言（source != target）
    pub fn is_cross(&self) -> bool {
        self.source != self.target
    }

    /// 预计转换难度
    pub fn difficulty(&self) -> ConversionDifficulty {
        if self.source == self.target {
            return ConversionDifficulty::None;
        }
        if self.source.same_family_as(&self.target) {
            return ConversionDifficulty::Easy;
        }
        ConversionDifficulty::Hard
    }
}

/// 跨方言合成难度等级
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversionDifficulty {
    /// 同方言（不需要转换）
    None,
    /// 同语族转换（较容易，如闽南语→闽东语）
    Easy,
    /// 跨语族转换（较困难，如普通话→粤语）
    Hard,
}

impl ConversionDifficulty {
    pub fn display_name(&self) -> &'static str {
        match self {
            ConversionDifficulty::None => "无需转换",
            ConversionDifficulty::Easy => "同语族转换",
            ConversionDifficulty::Hard => "跨语族转换",
        }
    }
}

// ============================================================
// 方言支持等级
// ============================================================

/// 方言支持等级
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DialectQuality {
    /// 母语级：发音地道，符合该方言语音体系
    Native,
    /// 可接受级：能正确朗读，但可能带有口音
    Acceptable,
}

impl DialectQuality {
    pub fn display_name(&self) -> &'static str {
        match self {
            DialectQuality::Native => "母语级",
            DialectQuality::Acceptable => "可接受",
        }
    }
}

/// 方言支持信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialectSupport {
    pub dialect: Dialect,
    pub quality: DialectQuality,
    /// 该方言推荐使用的音色 ID
    pub recommended_voice: Option<String>,
    /// 是否支持从普通话跨方言合成到此方言
    #[serde(default)]
    pub cross_synthesis_supported: bool,
}

impl DialectSupport {
    pub fn new(dialect: Dialect, quality: DialectQuality) -> Self {
        Self {
            dialect,
            quality,
            recommended_voice: None,
            cross_synthesis_supported: false,
        }
    }

    pub fn with_voice(mut self, voice: &str) -> Self {
        self.recommended_voice = Some(voice.to_string());
        self
    }

    /// 标记支持跨方言合成
    pub fn with_cross_synthesis(mut self) -> Self {
        self.cross_synthesis_supported = true;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dialect_列表14种() {
        assert_eq!(Dialect::all().len(), 14);
    }

    #[test]
    fn test_dialect_显示名称() {
        assert_eq!(Dialect::Mandarin.display_name(), "普通话");
        assert_eq!(Dialect::Cantonese.display_name(), "粤语");
        assert_eq!(Dialect::Xiang.display_name(), "湘语");
        assert_eq!(Dialect::Pinghua.display_name(), "平话");
    }

    #[test]
    fn test_dialect_代码解析() {
        assert_eq!(Dialect::from_code("yue"), Some(Dialect::Cantonese));
        assert_eq!(Dialect::from_code("hsn"), Some(Dialect::Xiang));
        assert_eq!(Dialect::from_code("cjy"), Some(Dialect::Jin));
        assert_eq!(Dialect::from_code("湖南话"), Some(Dialect::Xiang));
        assert_eq!(Dialect::from_code("unknown"), None);
    }

    #[test]
    fn test_dialect_语族分组() {
        assert_eq!(Dialect::Mandarin.family(), DialectFamily::Mandarin);
        assert_eq!(Dialect::Jin.family(), DialectFamily::Mandarin);
        assert_eq!(Dialect::SouthernMin.family(), DialectFamily::Min);
        assert_eq!(Dialect::Puxian.family(), DialectFamily::Min);
    }

    #[test]
    fn test_dialect_by_family() {
        let min_dialects = Dialect::by_family(DialectFamily::Min);
        assert_eq!(min_dialects.len(), 5); // 闽南、闽北、闽东、莆仙、闽中
        assert!(min_dialects.contains(&Dialect::SouthernMin));
    }

    #[test]
    fn test_dialect_所有方言不重复() {
        let all = Dialect::all();
        let len = all.len();
        let unique: std::collections::HashSet<_> = all.into_iter().collect();
        assert_eq!(unique.len(), len);
    }

    #[test]
    fn test_cross_dialect_同方言无需转换() {
        let conv = DialectConversion::new(Dialect::Mandarin, Dialect::Mandarin);
        assert!(!conv.is_cross());
        assert_eq!(conv.difficulty(), ConversionDifficulty::None);
    }

    #[test]
    fn test_cross_dialect_同语族容易() {
        let conv = DialectConversion::new(Dialect::SouthernMin, Dialect::EasternMin);
        assert!(conv.is_cross());
        assert_eq!(conv.difficulty(), ConversionDifficulty::Easy);
    }

    #[test]
    fn test_cross_dialect_跨语族困难() {
        let conv = DialectConversion::new(Dialect::Mandarin, Dialect::Cantonese);
        assert!(conv.is_cross());
        assert_eq!(conv.difficulty(), ConversionDifficulty::Hard);
    }

    #[test]
    fn test_dialect_support_构建() {
        let support = DialectSupport::new(Dialect::Cantonese, DialectQuality::Native)
            .with_voice("cantonese_male")
            .with_cross_synthesis();
        assert_eq!(support.dialect, Dialect::Cantonese);
        assert_eq!(support.quality, DialectQuality::Native);
        assert_eq!(support.recommended_voice, Some("cantonese_male".to_string()));
        assert!(support.cross_synthesis_supported);
    }
}
