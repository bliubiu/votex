//! `votex role` 子命令：小说角色扫描
//!
//! 分层约束（见 `votex-cli::lib` 模块文档）：本文件**只做参数解析与输出格式化**，
//! 扫描规则全部在 `votex_app::use_case::role_scan_use_case`。

use anyhow::Result;
use votex_app::use_case::role_scan_use_case::{
    RoleScanRequest, RoleScanResult, RoleScanUseCase,
};

/// 处理 role 子命令
pub fn handle(action: &crate::commands::root::RoleAction) -> Result<()> {
    use crate::commands::root::RoleAction;
    match action {
        RoleAction::Scan {
            input,
            out,
            top,
            assign_voices,
            json,
            models_dir,
        } => {
            let req = RoleScanRequest {
                input: input.clone(),
                top_n: *top,
                models_dir: models_dir.clone(),
                assign_voices: *assign_voices,
            };
            let result = RoleScanUseCase.execute(req)?;

            if *json {
                //机器可读输出：终端只出JSON，便于重定向与脚本消费
                let s = RoleScanUseCase.to_json(&result, &input)?;
                match out {
                    Some(p) => std::fs::write(&p, &s)
                        .map_err(|e| anyhow::anyhow!("写入 {} 失败: {}", p, e))?,
                    None => println!("{}", s),
                }
                if let Some(p) = out {
                    eprintln!("角色扫描结果已写入: {}", p);
                }
            } else {
                print_human(&result, &input, out.as_deref());
            }
        }
        RoleAction::Voices { models_dir } => {
            print_voices(&models_dir);
        }
    }
    Ok(())
}

/// 人类可读输出（终端表格 + 后续操作提示）
fn print_human(result: &RoleScanResult, source: &str, out: Option<&str>) {
    println!("角色扫描: {}", source);
    println!("耗时 {}ms", result.elapsed_ms);
    println!("音色池: {}", pool_summary(result));
    println!();

    if result.candidates.is_empty() {
        println!("未识别到角色。");
        println!(
            "提示：规则依赖「XX说：『台词』」形态，若原文多用无引号的间接引语，\
             可考虑先用 TTS 试听通用音色，或人工编写 --role-map JSON。"
        );
        return;
    }

    // 表头宽度按中文双宽字符对齐
    println!(
        "{:<12} {:>6}  {:<6}  {:<6}  {}",
        "角色", "台词数", "性别", "音色", "首段台词（试听用）"
    );
    println!("{}", "-".repeat(72));

    for c in &result.candidates {
        let gender = if c.gender_confirmed() {
            c.gender.display_cn().to_string()
        } else {
            // 未确认性别时如实说明，并把推测值作为提示给出
            match c.suggested_gender {
                Some(g) => format!("待确认?{}", g.display_cn()),
                None => "待确认".to_string(),
            }
        };
        let voice = result
            .role_map
            .as_ref()
            .and_then(|m| m.roles.get(&c.name))
            .map(String::as_str)
            .unwrap_or("-");
        let sample = c
            .sample_line
            .as_deref()
            .map(|s| {
                let t: String = s.chars().take(20).collect();
                if s.chars().count() > 20 {
                    format!("{}…", t)
                } else {
                    t
                }
            })
            .unwrap_or_else(|| "-".to_string());

        println!(
            "{:<12} {:>6}  {:<6}  {:<6}  {}",
            c.name, c.mentions, gender, voice, sample
        );
    }

    println!();
    let confirmed = result.candidates.iter().filter(|c| c.gender_confirmed()).count();
    println!(
        "共 {} 个角色（有称谓证据可定性别: {} 个，其余需人工确认）",
        result.candidates.len(),
        confirmed
    );

    if let Some(map) = &result.role_map {
        println!("旁白音色: {}", map.narrator.as_deref().unwrap_or("未分配"));
        println!(
            "台词兜底: {}",
            map.dialogue_default.as_deref().unwrap_or("未分配")
        );
    }

    if let Some(p) = out {
        println!();
        println!("保存角色表: {}", p);
        println!("用于合成: votex tts -i 书稿.txt -o 有声书.wav --role-map {}", p);
    } else {
        println!();
        println!("保存角色表: votex role scan {} --out roles.json --assign-voices", source);
        println!(
            "试听确认后用于合成: votex tts -i {} -o 有声书.wav --role-map roles.json",
            source
        );
    }
}

/// 打印可用音色池（供用户挑选/确认性别分布）
fn print_voices(models_dir: &str) {
    let voices = votex_app::platform::tts::list_engine_voices(std::path::Path::new(models_dir));
    if voices.is_empty() {
        println!("未找到音色（请确认已下载 Kokoro 模型: models/tts/kokoro-82m-v1.1-zh/）");
        return;
    }
    println!("可用音色（{}）:", voices.len());

    // 按性别+语种分组，避免 114 行平铺难以扫读
    for (label, want) in [
        ("中文女声", votex_domain::tts::value_object::VoiceGender::Female),
        ("中文男声", votex_domain::tts::value_object::VoiceGender::Male),
    ] {
        let group: Vec<_> = voices.iter().filter(|v| v.gender == want).collect();
        if group.is_empty() {
            continue;
        }
        println!("\n{} ({}):", label, group.len());
        for v in group {
            print!("  {}  ", v.voice_id);
        }
        println!();
    }

    let others: Vec<_> = voices
        .iter()
        .filter(|v| {
            !matches!(
                v.gender,
                votex_domain::tts::value_object::VoiceGender::Female
                    | votex_domain::tts::value_object::VoiceGender::Male
            )
        })
        .collect();
    if !others.is_empty() {
        println!("\n其他 ({}):", others.len());
        for v in others {
            print!("  {}  ", v.voice_id);
        }
        println!();
    }
}

/// 音色池摘要
fn pool_summary(result: &RoleScanResult) -> String {
    let f = |g: votex_domain::tts::value_object::VoiceGender| {
        result.voices.iter().filter(|v| v.gender == g).count()
    };
    if result.voices.is_empty() {
        return "空（未找到 Kokoro 音色，无法自动分配）".to_string();
    }
    format!(
        "{} 个（中文女声 {} / 中文男声 {} / 其它 {}）",
        result.voices.len(),
        f(votex_domain::tts::value_object::VoiceGender::Female),
        f(votex_domain::tts::value_object::VoiceGender::Male),
        result.voices.len()
            - f(votex_domain::tts::value_object::VoiceGender::Female)
            - f(votex_domain::tts::value_object::VoiceGender::Male),
    )
}