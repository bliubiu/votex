//! 关键词翻译服务
//!
//! 将中文关键词翻译为英文，用于视频素材搜索（Pexels/Pixabay 需要英文关键词）。
//!
//! 三级翻译流水线，自动降级：
//! 1. 离线词典（最快，零依赖）
//! 2. LLM 翻译（在线，需 DeepSeek API Key）
//! 3. 全部失败 → 返回原始关键词

use std::collections::HashMap;

/// 检测是否包含中文字符
fn has_chinese(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(c,
            '\u{4E00}'..='\u{9FFF}' |
            '\u{3400}'..='\u{4DBF}' |
            '\u{F900}'..='\u{FAFF}'
        )
    })
}

/// 内置中文→英文离线词典
fn build_dict() -> HashMap<&'static str, &'static str> {
    let mut d = HashMap::new();
    // 自然风景
    d.insert("日落", "sunset");
    d.insert("日出", "sunrise");
    d.insert("海滩", "beach");
    d.insert("大海", "ocean");
    d.insert("海浪", "wave");
    d.insert("山", "mountain");
    d.insert("山脉", "mountains");
    d.insert("森林", "forest");
    d.insert("河流", "river");
    d.insert("湖泊", "lake");
    d.insert("瀑布", "waterfall");
    d.insert("花", "flower");
    d.insert("樱花", "cherry blossom");
    d.insert("天空", "sky");
    d.insert("云", "cloud");
    d.insert("星空", "starry sky");
    d.insert("月亮", "moon");
    d.insert("雪", "snow");
    d.insert("雨", "rain");
    d.insert("彩虹", "rainbow");
    // 城市建筑
    d.insert("城市", "city");
    d.insert("街道", "street");
    d.insert("建筑", "building");
    d.insert("桥梁", "bridge");
    d.insert("广场", "square");
    d.insert("公园", "park");
    d.insert("市场", "market");
    d.insert("寺庙", "temple");
    d.insert("教堂", "church");
    d.insert("城堡", "castle");
    d.insert("乡村", "village");
    // 人物生活
    d.insert("人", "people");
    d.insert("人群", "crowd");
    d.insert("家庭", "family");
    d.insert("孩子", "children");
    d.insert("婚礼", "wedding");
    d.insert("朋友", "friends");
    d.insert("学生", "student");
    d.insert("老师", "teacher");
    // 工作商务
    d.insert("工作", "work");
    d.insert("办公室", "office");
    d.insert("会议", "meeting");
    d.insert("演讲", "speech");
    d.insert("商务", "business");
    // 美食烹饪
    d.insert("美食", "food");
    d.insert("水果", "fruit");
    d.insert("咖啡", "coffee");
    d.insert("茶", "tea");
    d.insert("蛋糕", "cake");
    d.insert("面包", "bread");
    d.insert("饺子", "dumpling");
    d.insert("火锅", "hotpot");
    d.insert("烧烤", "barbecue");
    d.insert("厨房", "kitchen");
    // 旅行交通
    d.insert("旅行", "travel");
    d.insert("度假", "vacation");
    d.insert("徒步", "hiking");
    d.insert("露营", "camping");
    d.insert("骑行", "cycling");
    // 运动健身
    d.insert("运动", "sports");
    d.insert("跑步", "running");
    d.insert("健身", "fitness");
    d.insert("瑜伽", "yoga");
    d.insert("游泳", "swimming");
    d.insert("滑雪", "skiing");
    d.insert("篮球", "basketball");
    d.insert("足球", "football");
    // 动物
    d.insert("动物", "animal");
    d.insert("猫", "cat");
    d.insert("狗", "dog");
    d.insert("鸟", "bird");
    d.insert("鱼", "fish");
    d.insert("马", "horse");
    d.insert("蝴蝶", "butterfly");
    d.insert("熊猫", "panda");
    d.insert("海豚", "dolphin");
    // 科技
    d.insert("科技", "technology");
    d.insert("电脑", "computer");
    d.insert("机器人", "robot");
    d.insert("手机", "mobile phone");
    d.insert("互联网", "internet");
    d.insert("数据", "data");
    // 艺术创意
    d.insert("艺术", "art");
    d.insert("设计", "design");
    d.insert("绘画", "painting");
    d.insert("摄影", "photography");
    d.insert("音乐", "music");
    d.insert("舞蹈", "dance");
    d.insert("电影", "movie");
    d.insert("动画", "animation");
    // 视频制作
    d.insert("背景", "background");
    d.insert("抽象", "abstract");
    d.insert("自然", "nature");
    d.insert("慢动作", "slow motion");
    d.insert("航拍", "aerial");
    d.insert("延时", "time lapse");
    d.insert("剪影", "silhouette");
    d.insert("特写", "close up");
    d.insert("纹理", "texture");
    d.insert("粒子", "particle");
    d.insert("烟雾", "smoke");
    d.insert("火焰", "fire");
    d.insert("水", "water");
    // 季节节日
    d.insert("春天", "spring");
    d.insert("夏天", "summer");
    d.insert("秋天", "autumn");
    d.insert("冬天", "winter");
    d.insert("春节", "spring festival");
    d.insert("圣诞", "christmas");
    d.insert("情人节", "valentine's day");
    // 色彩
    d.insert("红色", "red");
    d.insert("蓝色", "blue");
    d.insert("绿色", "green");
    d.insert("白色", "white");
    d.insert("黑色", "black");
    d.insert("金色", "golden");
    d.insert("彩色", "colorful");
    d.insert("黑白", "black and white");
    // 情绪氛围
    d.insert("快乐", "happiness");
    d.insert("浪漫", "romantic");
    d.insert("宁静", "peaceful");
    d.insert("热闹", "lively");
    d.insert("庆祝", "celebration");
    d.insert("节日", "festival");
    d.insert("烟花", "fireworks");
    d
}

/// 关键词翻译服务
pub struct KeywordTranslator {
    dict: HashMap<&'static str, &'static str>,
    llm_engine: String,
}

impl KeywordTranslator {
    /// 创建关键词翻译服务
    pub fn new(llm_engine: Option<&str>) -> Self {
        Self {
            dict: build_dict(),
            llm_engine: llm_engine.unwrap_or("deepseek-chat").to_string(),
        }
    }

    /// 翻译关键词
    ///
    /// 三级流水线：
    /// 1. 不含中文 → 直接返回
    /// 2. 离线词典（最快）
    /// 3. LLM 翻译（在线，需 API Key）
    /// 4. 全部失败 → 返回原文
    pub fn translate(&self, keyword: &str) -> String {
        let keyword = keyword.trim();
        if keyword.is_empty() {
            return String::new();
        }

        // 不含中文 → 直接返回
        if !has_chinese(keyword) {
            return keyword.to_string();
        }

        // 1. 离线词典
        if let Some(result) = self.dict_translate(keyword) {
            return result;
        }

        // 2. LLM 翻译
        if let Some(result) = self.llm_translate(keyword) {
            return result;
        }

        // 3. 全部失败 → 返回原文
        keyword.to_string()
    }

    /// 离线词典翻译
    fn dict_translate(&self, keyword: &str) -> Option<String> {
        // 精确匹配
        if let Some(&translation) = self.dict.get(keyword) {
            return Some(translation.to_string());
        }

        // 贪心最长匹配分词
        let mut parts: Vec<&str> = Vec::new();
        let mut remaining = keyword;
        while !remaining.is_empty() {
            let mut matched = false;
            let max_len = std::cmp::min(4, remaining.chars().count());
            for len in (1..=max_len).rev() {
                let chunk: String = remaining.chars().take(len).collect();
                if self.dict.contains_key(chunk.as_str()) {
                    parts.push(self.dict[chunk.as_str()]);
                    remaining = &remaining[chunk.len()..];
                    matched = true;
                    break;
                }
            }
            if !matched {
                // 有不认识的词 → 交给 LLM
                return None;
            }
        }

        Some(parts.join(" "))
    }

    /// LLM 翻译
    fn llm_translate(&self, keyword: &str) -> Option<String> {
        let api_key = std::env::var("DEEPSEEK_API_KEY").ok()?;
        if api_key.is_empty() {
            return None;
        }

        let client = votex_infra::api::base::BaseApiClient::new(
            votex_infra::api::base::ApiConfig {
                api_key: Some(api_key),
                endpoint: Some("https://api.deepseek.com".to_string()),
                timeout: std::time::Duration::from_secs(30),
                ..Default::default()
            }
        );

        let body = serde_json::json!({
            "model": self.llm_engine,
            "messages": [
                {"role": "system", "content": "你是一个翻译助手。请将用户输入的中文关键词翻译成英文。只输出翻译结果，不要包含任何解释、标点符号或额外内容。如果输入是英文或专有名词，直接原样返回。"},
                {"role": "user", "content": keyword}
            ],
            "temperature": 0.1,
            "max_tokens": 128,
        });

        let headers = vec![
            ("Authorization".to_string(), format!("Bearer {}", std::env::var("DEEPSEEK_API_KEY").unwrap_or_default())),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];

        let resp = client.post_json(
            "https://api.deepseek.com/v1/chat/completions",
            Some(&headers),
            body,
        ).ok()?;

        let result: serde_json::Value = resp.json().ok()?;
        let text = result["choices"][0]["message"]["content"]
            .as_str()?
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();

        if text.is_empty() { None } else { Some(text) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dict_translate_exact() {
        let t = KeywordTranslator::new(None);
        assert_eq!(t.translate("日落"), "sunset");
        assert_eq!(t.translate("海滩"), "beach");
        assert_eq!(t.translate("星空"), "starry sky");
    }

    #[test]
    fn test_dict_translate_no_chinese() {
        let t = KeywordTranslator::new(None);
        assert_eq!(t.translate("sunset"), "sunset");
        assert_eq!(t.translate("hello world"), "hello world");
    }

    #[test]
    fn test_dict_translate_multi_word() {
        let t = KeywordTranslator::new(None);
        assert_eq!(t.translate("春天樱花"), "spring cherry blossom");
        assert_eq!(t.translate("蓝色天空"), "blue sky");
    }

    #[test]
    fn test_dict_translate_empty() {
        let t = KeywordTranslator::new(None);
        assert_eq!(t.translate(""), "");
        assert_eq!(t.translate("   "), "");
    }

    #[test]
    fn test_dict_translate_single_char() {
        let t = KeywordTranslator::new(None);
        assert_eq!(t.translate("猫"), "cat");
        assert_eq!(t.translate("狗"), "dog");
    }

    #[test]
    fn test_dict_translate_with_numbers() {
        let t = KeywordTranslator::new(None);
        // 含数字的关键词，数字被保留，中文部分被翻译
        let result = t.translate("2024春节");
        // "2024" 不在词典中，整句降级到 LLM（无 API Key 时返回原文）
        assert!(result.contains("2024"), "结果: {}", result);
    }

    #[test]
    fn test_has_chinese() {
        assert!(has_chinese("中文"));
        assert!(has_chinese("hello 世界"));
        assert!(!has_chinese("hello"));
        assert!(!has_chinese("12345"));
        assert!(!has_chinese(""));
    }

    #[test]
    fn test_unknown_keyword_fallback_to_original() {
        let t = KeywordTranslator::new(None);
        // 不在词典中的词 → 没有 API Key 时返回原文
        let result = t.translate("超新星爆发");
        assert_eq!(result, "超新星爆发");
    }

    #[test]
    fn test_dict_build_coverage() {
        let dict = build_dict();
        // 确保常见视频搜索关键词都有覆盖
        assert!(dict.contains_key("城市"), "缺少 '城市'");
        assert!(dict.contains_key("办公室"), "缺少 '办公室'");
        assert!(dict.contains_key("瑜伽"), "缺少 '瑜伽'");
        assert!(dict.contains_key("海豚"), "缺少 '海豚'");
        assert!(dict.contains_key("熊猫"), "缺少 '熊猫'");
        assert!(dict.contains_key("慢动作"), "缺少 '慢动作'");
        assert!(dict.contains_key("航拍"), "缺少 '航拍'");
        // 总数不少于 80 个
        assert!(dict.len() >= 80, "词典条目数: {}", dict.len());
    }
}
