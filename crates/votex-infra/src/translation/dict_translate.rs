//! 离线词典翻译提供者
//!
//! 使用内置中英词典进行关键词翻译，适用于短视频标签、文件名等简短文本。
//! 作为 LLM 翻译的离线降级方案。

use std::collections::HashMap;
use votex_domain::error::TranslationError;
use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;

/// 基础中英词典（高频词汇）
fn build_dict() -> HashMap<&'static str, &'static str> {
    let mut dict = HashMap::new();

    // 常用场景词汇
    dict.insert("日落", "sunset");
    dict.insert("海滩", "beach");
    dict.insert("日出", "sunrise");
    dict.insert("山脉", "mountain");
    dict.insert("城市", "city");
    dict.insert("夜景", "night view");
    dict.insert("美食", "food");
    dict.insert("旅行", "travel");
    dict.insert("风景", "scenery");
    dict.insert("自然", "nature");
    dict.insert("动物", "animal");
    dict.insert("植物", "plant");
    dict.insert("建筑", "architecture");
    dict.insert("历史", "history");
    dict.insert("文化", "culture");
    dict.insert("艺术", "art");
    dict.insert("音乐", "music");
    dict.insert("舞蹈", "dance");
    dict.insert("运动", "sports");
    dict.insert("科技", "technology");
    dict.insert("时尚", "fashion");
    dict.insert("生活", "life");
    dict.insert("爱情", "love");
    dict.insert("家庭", "family");
    dict.insert("朋友", "friend");
    dict.insert("工作", "work");
    dict.insert("学习", "study");
    dict.insert("健康", "health");
    dict.insert("快乐", "happiness");
    dict.insert("梦想", "dream");
    dict.insert("未来", "future");
    dict.insert("成功", "success");
    dict.insert("美丽", "beautiful");
    dict.insert("温馨", "warm");
    dict.insert("浪漫", "romantic");
    dict.insert("经典", "classic");
    dict.insert("现代", "modern");
    dict.insert("传统", "traditional");
    dict.insert("中国", "China");
    dict.insert("日本", "Japan");
    dict.insert("韩国", "Korea");
    dict.insert("法国", "France");
    dict.insert("意大利", "Italy");
    dict.insert("美国", "America");
    dict.insert("英国", "UK");

    // 天气/季节
    dict.insert("春天", "spring");
    dict.insert("夏天", "summer");
    dict.insert("秋天", "autumn");
    dict.insert("冬天", "winter");
    dict.insert("阳光", "sunshine");
    dict.insert("雨", "rain");
    dict.insert("雪", "snow");
    dict.insert("风", "wind");
    dict.insert("云", "cloud");

    // 颜色
    dict.insert("红", "red");
    dict.insert("蓝", "blue");
    dict.insert("绿", "green");
    dict.insert("白", "white");
    dict.insert("黑", "black");
    dict.insert("金", "gold");
    dict.insert("银", "silver");

    // 视频/媒体相关
    dict.insert("视频", "video");
    dict.insert("音频", "audio");
    dict.insert("字幕", "subtitle");
    dict.insert("配音", "dubbing");
    dict.insert("文案", "script");
    dict.insert("剪辑", "editing");
    dict.insert("特效", "effects");
    dict.insert("背景", "background");
    dict.insert("模板", "template");
    dict.insert("教程", "tutorial");
    dict.insert("Vlog", "vlog");
    dict.insert("短片", "short film");

    dict
}

/// 词典翻译提供者
pub struct DictTranslationProvider {
    dict: HashMap<&'static str, &'static str>,
}

impl DictTranslationProvider {
    pub fn new() -> Self {
        Self {
            dict: build_dict(),
        }
    }
}

impl TranslationProvider for DictTranslationProvider {
    fn name(&self) -> &str {
        "dict"
    }

    fn supported_pairs(&self) -> Vec<(String, String)> {
        // 内置词典是中→英单向的
        vec![("zh".to_string(), "en".to_string())]
    }

    fn max_input_chars(&self) -> usize {
        // 词典逐词匹配，单句即可
        200
    }

    fn translate(&self, text: &str, direction: TranslationDirection) -> Result<String, TranslationError> {
        if text.is_empty() {
            return Err(TranslationError::EmptyText);
        }

        match direction {
            TranslationDirection::ZhToEn
            | TranslationDirection::ByLanguagePair { source: _, target: _ } => {
                // 尝试整个短语匹配
                if let Some(&translation) = self.dict.get(text) {
                    return Ok(translation.to_string());
                }

                // 逐词翻译
                let words: Vec<&str> = text.split_whitespace().collect();
                let mut translated_words = Vec::new();
                for word in words {
                    match self.dict.get(word) {
                        Some(&t) => translated_words.push(t.to_string()),
                        None => translated_words.push(word.to_string()),
                    }
                }

                if translated_words.is_empty() {
                    return Err(TranslationError::UnsupportedDirection);
                }

                Ok(translated_words.join(" "))
            }
            TranslationDirection::EnToZh | TranslationDirection::Auto => {
                // 词典是单向的，反向翻译不完整
                Err(TranslationError::UnsupportedDirection)
            }
        }
    }
}
