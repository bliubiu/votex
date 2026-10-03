use anyhow::Result;
use std::path::{Path, PathBuf};
use votex_app::use_case::translation_use_case::{load_glossary_from_file, TranslationUseCase};
use votex_domain::translation::glossary::Glossary;
use votex_domain::translation::options::TranslationOptions;
use votex_domain::translation::script::ChineseScript;

/// 翻译命令的可选参数
pub struct TranslateArgs<'a> {
    pub text: &'a str,
    pub engine: &'a str,
    pub direction: &'a str,
    /// 术语表文件路径（可选）
    pub glossary: Option<&'a str>,
    /// 模型根目录（可选，默认 models/）
    pub models_dir: Option<&'a str>,
    /// 目标书写系统（可选）
    pub target_script: Option<&'a str>,
    /// 单段最大字符数（0 表示自动）
    pub max_segment_chars: usize,
    /// 是否关闭翻译缓存
    pub no_cache: bool,
}

/// 处理翻译命令
pub fn handle_translate(text: &str, engine: &str, direction: &str) -> Result<()> {
    handle_translate_with_args(TranslateArgs {
        text,
        engine,
        direction,
        glossary: None,
        models_dir: None,
        target_script: None,
        max_segment_chars: 0,
        no_cache: false,
    })
}

/// 处理翻译命令（完整参数）
pub fn handle_translate_with_args(args: TranslateArgs) -> Result<()> {
    let models_dir = PathBuf::from(args.models_dir.unwrap_or("models"));

    println!("翻译引擎: {}", args.engine);
    println!("翻译方向: {}", args.direction);
    if let Some(g) = args.glossary {
        println!("术语表: {}", g);
    }
    println!("原文: {}", args.text);

    let use_case = TranslationUseCase::with_models_dir(args.engine, &models_dir)?;

    let mut options = TranslationOptions::new(TranslationUseCase::parse_direction(args.direction)?);
    options.max_segment_chars = args.max_segment_chars;

    if args.no_cache {
        options.use_cache = false;
    }

    if let Some(path) = args.glossary {
        options.glossary = load_glossary_from_file(Path::new(path))?;
        println!("已加载术语: {} 条", options.glossary.len());
    }

    if let Some(script) = args.target_script {
        options.target_script = parse_script(script);
    }

    let outcome = use_case.translate_with_options(args.text, &options)?;

    println!("译文: {}", outcome.text);

    // 分段与缓存信息只在非平凡情况下输出，避免干扰默认输出
    if outcome.segments > 1 {
        println!("（长文本已分为 {} 段翻译）", outcome.segments);
    }
    if outcome.cached_segments > 0 {
        println!("（命中缓存 {} 段）", outcome.cached_segments);
    }

    Ok(())
}

/// 解析书写系统参数
fn parse_script(s: &str) -> ChineseScript {
    match s.to_lowercase().as_str() {
        "simplified" | "hans" | "zh-hans" => ChineseScript::Simplified,
        "traditional" | "hant" | "zh-hant" | "cht" => ChineseScript::Traditional,
        _ => ChineseScript::None,
    }
}

/// 批量翻译：从文件读取多行文本，逐行翻译后写入输出文件
pub fn handle_batch_translate(
    input: &str,
    output: &str,
    engine: &str,
    direction: &str,
    glossary: Option<&str>,
    models_dir: Option<&str>,
) -> Result<()> {
    let input_path = Path::new(input);
    if !input_path.exists() {
        anyhow::bail!("输入文件不存在: {}", input);
    }

    let content = std::fs::read_to_string(input_path)
        .map_err(|e| anyhow::anyhow!("读取输入文件失败: {}", e))?;
    let lines: Vec<String> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect();

    if lines.is_empty() {
        anyhow::bail!("输入文件没有有效内容: {}", input);
    }

    let dir = PathBuf::from(models_dir.unwrap_or("models"));
    let use_case = TranslationUseCase::with_models_dir(engine, &dir)?;

    let mut options = TranslationOptions::new(TranslationUseCase::parse_direction(direction)?);
    if let Some(path) = glossary {
        options.glossary = load_glossary_from_file(Path::new(path))?;
    }

    println!("批量翻译: {} 条文本，引擎 {}，方向 {}", lines.len(), engine, direction);

    let mut outputs = Vec::with_capacity(lines.len());
    let outcomes = use_case.pipeline().translate_batch_with_progress(
        &lines,
        &options,
        &mut |done, total| {
            println!("  进度: {}/{}", done, total);
        },
    )?;

    for outcome in outcomes {
        outputs.push(outcome.text);
    }

    if let Some(parent) = Path::new(output).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(output, outputs.join("\n"))?;

    println!("已写入: {}（{} 条）", output, outputs.len());
    println!("缓存命中率: {:.1}%", use_case.cache_hit_rate() * 100.0);
    Ok(())
}

/// 生成一个术语表模板文件，方便用户填写
pub fn write_glossary_template(path: &str) -> Result<()> {
    let template = Glossary::from_entries(vec![
        votex_domain::translation::glossary::GlossaryEntry::new("爱丽丝", "Alice")
            .with_info("女主角"),
        votex_domain::translation::glossary::GlossaryEntry::new("鲍勃", "Bob"),
    ]);
    let json = template.to_json()?;
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, json)?;
    println!("术语表模板已写入: {}", path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 书写系统参数解析() {
        assert_eq!(parse_script("traditional"), ChineseScript::Traditional);
        assert_eq!(parse_script("hant"), ChineseScript::Traditional);
        assert_eq!(parse_script("simplified"), ChineseScript::Simplified);
        assert_eq!(parse_script("hans"), ChineseScript::Simplified);
        assert_eq!(parse_script("none"), ChineseScript::None);
        assert_eq!(parse_script("随便什么"), ChineseScript::None);
    }

    #[test]
    fn 术语表模板可写可读回() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("glossary.json");
        let path_str = path.to_string_lossy().to_string();
        write_glossary_template(&path_str).unwrap();
        let loaded = load_glossary_from_file(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.entries()[0].dst, "Alice");
    }

    #[test]
    fn 批量翻译_输入文件不存在时报错() {
        let r = handle_batch_translate("/nonexistent/input.txt", "/tmp/out.txt", "dict", "zh-en", None, None);
        assert!(r.is_err());
    }
}
