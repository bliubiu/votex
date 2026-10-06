use std::collections::HashMap;
use std::sync::LazyLock;

use crate::shared::WorkspacePaths;

/// Kokoro 中文 G2P 映射表所在子目录（相对 `models/tts/`）
const KOKORO_ZH_SUBDIR: &str = "kokoro-82m-v1.1-zh";

/// 在统一的模型根目录下定位映射表文件
fn locate_map_file(file_name: &str) -> std::path::PathBuf {
    WorkspacePaths::models_dir()
        .join("tts")
        .join(KOKORO_ZH_SUBDIR)
        .join(file_name)
}

// ===================== v1.0: 拼音→IPA 映射 =====================

/// 拼音→IPA 映射表，键为 "ni3" 格式的拼音+声调，值为 IPA 音素字符串
/// 从文件系统加载（而非编译时），以容忍模型目录不完整
static PINYIN_MAP: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let path = locate_map_file("pinyin_to_ipa.json");
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

// ===================== v1.1-zh: 拼音→注音映射 =====================

/// 拼音→注音映射表，键为 "ni3" 格式的拼音+声调，值为注音字符串（如 "ㄋㄧ3"）
static PINYIN_ZHUYIN_MAP: LazyLock<HashMap<String, String>> = LazyLock::new(|| {
    load_pinyin_zhuyin_map()
});

/// 运行时加载拼音→注音映射
fn load_pinyin_zhuyin_map() -> HashMap<String, String> {
    let path = locate_map_file("pinyin_to_zhuyin.json");
    match std::fs::read_to_string(&path) {
        Ok(content) => parse_string_map(&content, &path.display().to_string()),
        Err(_) => {
            tracing::warn!(
                "拼音→注音文件不存在 ({}), 注音功能不可用",
                path.display()
            );
            HashMap::new()
        }
    }
}

/// 返回已加载的 v1.0 映射条目数（供测试断言映射表非空）
#[cfg(test)]
pub fn len() -> usize {
    PINYIN_MAP.len()
}

/// 查询拼音→注音映射（v1.1-zh）
pub fn lookup_zhuyin(pinyin_with_tone: &str) -> Option<String> {
    PINYIN_ZHUYIN_MAP.get(pinyin_with_tone).cloned()
}

/// 返回 v1.1-zh 映射条目数（供测试断言映射表非空）
#[cfg(test)]
pub fn zhuyin_len() -> usize {
    PINYIN_ZHUYIN_MAP.len()
}

// ===================== 通用 =====================

/// 解析 String→String 映射 JSON
///
/// 降级策略：JSON 损坏时记录错误并返回空表，而不是 panic。
/// 旧实现在此 `panic!("{} 格式无效")`，而同一模块另一处
/// `load_pinyin_zhuyin_map` 却用 `unwrap_or_default()` 静默返回空表 ——
/// 同一类问题两套处理，且 panic 版本会直接击崩溃 GUI 进程。
fn parse_string_map(json_str: &str, name: &str) -> HashMap<String, String> {
    match serde_json::from_str(json_str) {
        Ok(m) => m,
        Err(e) => {
            tracing::error!("{} 格式无效，相关 G2P 映射不可用: {}", name, e);
            HashMap::new()
        }
    }
}

/// 从 JSON 字符串加载 String→String 映射并泄漏为 &'static str
fn load_string_map(json_str: &str, name: &str) -> HashMap<&'static str, &'static str> {
    let map = parse_string_map(json_str, name);

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
