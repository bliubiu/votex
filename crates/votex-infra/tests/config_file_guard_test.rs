//! 配置文件守护测试
//!
//! 背景：曾因 `application.yml` 中一行注释 `#` 前缺少空格
//! （`progressive_quality: true# 资源治理…`），导致整份配置反序列化失败，
//! 又被调用方的 `.ok()` 静默吞掉，程序长期按硬编码默认值运行，
//! 用户修改的所有配置（含 DirectML 崩溃保护开关）全部失效且毫无提示。
//!
//! 本测试作为回归防线：配置文件一旦不可解析，CI 立即失败。

// 项目测试用例统一使用中文命名，允许非 snake_case 风格
#![allow(non_snake_case)]

use std::path::PathBuf;
use votex_domain::config::value_object::AppConfig;
use votex_infra::config::loader::ConfigLoader;

/// 定位工作区根目录的 application.yml
fn workspace_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("application.yml")
}

#[test]
fn 仓库根目录的_application_yml_必须能被解析() {
    let path = workspace_config_path();
    assert!(
        path.exists(),
        "找不到仓库根配置文件: {}",
        path.display()
    );

    let content = std::fs::read_to_string(&path).expect("读取 application.yml 失败");

    // 反序列化失败时给出明确的类型/行号信息，便于定位
    match serde_yml::from_str::<AppConfig>(&content) {
        Ok(_) => {}
        Err(e) => panic!(
            "application.yml 解析失败: {}\n\
             常见原因：注释 `#` 前缺少空格（YAML 要求 `#` 前必须有空白）、\n\
             缩进不一致、字段类型不匹配。\n\
             注意：单个字段解析失败会导致整份配置失效并静默回退默认值。",
            e
        ),
    }
}

#[test]
fn 配置文件中的执行提供器取值必须合法() {
    let path = workspace_config_path();
    let config = ConfigLoader::from_file(&path).expect("application.yml 应可解析");

    let ep = config.inference.execution_provider.as_str();
    assert!(
        matches!(ep, "cpu" | "directml" | "auto"),
        "execution_provider 取值非法: {}（应为 cpu / directml / auto）",
        ep
    );
}

#[test]
fn 配置文件的日志级别取值必须合法() {
    let path = workspace_config_path();
    let config = ConfigLoader::from_file(&path).expect("application.yml 应可解析");

    let level = config.log.level.as_str();
    assert!(
        matches!(level, "debug" | "info" | "error"),
        "log.level 取值非法: {}（应为 debug / info / error）",
        level
    );
}

#[test]
fn 解析失败时_load_optional_必须报错而非静默返回默认() {
    // 构造一份含典型语法错误的配置：注释 `#` 前缺空格
    let bad = tempfile::NamedTempFile::new().expect("创建临时文件失败");
    std::fs::write(
        bad.path(),
        "pipeline:\n  progressive_quality: true# 注释前缺空格\n",
    )
    .expect("写入临时文件失败");

    // 文件存在但解析失败 → 返回 None（程序继续用默认值），
    // 但必须已经向 stderr 打印了错误说明
    let result = ConfigLoader::load_optional(bad.path());
    assert!(
        result.is_none(),
        "解析失败时不应返回配置对象，否则等同于静默降级"
    );

    // 对照：`#` 前补上空格后，同一个值必须被解析为布尔 true 而非字符串
    let good: serde_yml::Value =
        serde_yml::from_str("pipeline:\n  progressive_quality: true # 注释前有空格\n")
            .expect("正确写法应可解析");
    assert_eq!(
        good["pipeline"]["progressive_quality"],
        serde_yml::Value::Bool(true),
        "注释前补空格后应解析为布尔 true"
    );

    // 而错误写法会被解析成字符串（这正是当初让整份配置失效的根因）
    let bad_value: serde_yml::Value =
        serde_yml::from_str("pipeline:\n  progressive_quality: true# 缺空格\n")
            .expect("错误写法在 Value 层面可解析，只是类型变成字符串");
    assert!(
        bad_value["pipeline"]["progressive_quality"].is_string(),
        "错误写法应被解析为字符串，从而与 bool 字段类型冲突"
    );
}

#[test]
fn 文件不存在时_load_optional_静默返回_None() {
    let missing = PathBuf::from("__不存在的配置文件__.yml");
    assert!(
        ConfigLoader::load_optional(&missing).is_none(),
        "文件不存在属正常场景，应静默返回 None"
    );
}
