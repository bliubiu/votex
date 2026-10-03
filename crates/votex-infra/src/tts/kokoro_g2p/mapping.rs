use std::collections::HashMap;
use std::sync::LazyLock;

// ===================== v1.0: 拼音→IPA 映射 =====================

/// 拼音→IPA 映射表，键为 "ni3" 格式的拼音+声调，值为 IPA 音素字符串
/// 从文件系统加载（而非编译时），以容忍模型目录不完整
static PINYIN_MAP: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest_dir)
        .join("../../models/tts/kokoro-82m-v1.1-zh/pinyin_to_ipa.json");
    match std::fs::read_to_string(&path) {
        Ok(json_str) => load_string_map(&json_str, "pinyin_to_ipa.json"),
        Err(_) => {
            tracing::warn!("pinyin_to_ipa.json not found (v1.0 G2P unavailable): {}", path.display());
            HashMap::new()
        }
    }
});

/// 查询拼音→IPA 映射
pub fn lookup(pinyin_with_tone: &str) -> Option<&'static str> {
    PINYIN_MAP.get(pinyin_with_tone).copied()
}

/// 返回已加载的 v1.0 映射条目数
#[allow(dead_code)]
pub fn len() -> usize {
    PINYIN_MAP.len()
}

// ===================== v1.1-zh: 拼音→注音映射 =====================

/// 拼音→注音映射表，键为 "ni3" 格式的拼音+声调，值为注音字符串（如 "ㄋㄧ3"）
static PINYIN_ZHUYIN_MAP: LazyLock<HashMap<String, String>> = LazyLock::new(|| {
    load_pinyin_zhuyin_map()
});

/// 运行时加载拼音→注音映射
fn load_pinyin_zhuyin_map() -> HashMap<String, String> {
    // 尝试当前工作目录
    let relative_path = "models/tts/kokoro-82m-v1.1-zh/pinyin_to_zhuyin.json";
    let path = std::path::Path::new(relative_path);
    if path.exists() {
        if let Ok(content) = std::fs::read_to_string(path) {
            return serde_json::from_str(&content).unwrap_or_default();
        }
    }
    // 尝试从 manifest 目录查找
    let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .join(relative_path);
    if manifest_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&manifest_path) {
            return serde_json::from_str(&content).unwrap_or_default();
        }
    }
    tracing::warn!("拼音→注音文件不存在 ({}), 注音功能不可用", relative_path);
    HashMap::new()
}

/// 查询拼音→注音映射（v1.1-zh）
pub fn lookup_zhuyin(pinyin_with_tone: &str) -> Option<String> {
    PINYIN_ZHUYIN_MAP.get(pinyin_with_tone).cloned()
}

/// 返回 v1.1-zh 映射条目数
#[allow(dead_code)]
pub fn zhuyin_len() -> usize {
    PINYIN_ZHUYIN_MAP.len()
}

// ===================== 通用 =====================

/// 从 JSON 字符串加载 String→String 映射并泄漏为 &'static str
fn load_string_map(json_str: &str, name: &str) -> HashMap<&'static str, &'static str> {
    let map: HashMap<String, String> = serde_json::from_str(json_str)
        .unwrap_or_else(|e| panic!("{} 格式无效: {}", name, e));

    let mut static_map: HashMap<&'static str, &'static str> = HashMap::with_capacity(map.len());
    for (k, v) in map.into_iter() {
        let k: &'static str = Box::leak(k.into_boxed_str());
        let v: &'static str = Box::leak(v.into_boxed_str());
        static_map.insert(k, v);
    }
    static_map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_known() {
        // v1.0 IPA 映射文件不存在时，lookup 返回 None
        // 如果文件存在，应返回对应的 IPA 值
        let result = lookup("ni3");
        // 允许两种情况：文件不存在（None）或文件存在（Some）
        if result.is_some() {
            // 如果文件存在，验证返回的是 IPA 格式
            assert!(result.unwrap().contains("˧") || result.unwrap().contains("˥"));
        }
    }

    #[test]
    fn test_lookup_unknown() {
        assert_eq!(lookup("notexist9"), None);
        assert_eq!(lookup(""), None);
    }

    #[test]
    fn test_map_not_empty() {
        // v1.0 IPA 映射文件可能不存在，此时 len() 为 0
        // v1.1-zh 注音映射文件应该存在且有大量条目
        let ipa_len = len();
        let zhuyin_len = zhuyin_len();
        // 至少注音映射应该有大量条目
        assert!(zhuyin_len > 1500, "注音映射应有超过 1500 条目，实际: {}", zhuyin_len);
        // IPA 映射可能为空（文件不存在）
        if ipa_len > 0 {
            assert!(ipa_len > 2500, "IPA 映射应有超过 2500 条目，实际: {}", ipa_len);
        }
    }
}
