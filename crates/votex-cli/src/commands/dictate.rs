//! `votex dictate` —— 实时听写
//!
//! 麦克风 → 流式识别 → 终端增量打印；Ctrl+C 停止并把全文落盘。
//! 业务逻辑全部在 [`votex_app::use_case::dictation_use_case`]。

use anyhow::Result;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use votex_app::use_case::dictation_use_case::{DictationEvent, DictationUseCase};

/// 处理 dictate 子命令
///
/// 终端输出约定：
/// - 中间结果：单行覆盖刷新（`\r`），提示为「未定稿」
/// - 定稿文本：换行打印，保留在终端历史中
/// - 停止：打印统计与保存路径
pub fn handle(model: &str, output: Option<&str>) -> Result<()> {
    let use_case = DictationUseCase::shared();

    println!("实时听写开始");
    println!("  模型: {}", model);
    match output {
        Some(path) => println!("  转写保存: {}", path),
        None => println!("  转写保存: 未指定（仅终端输出，Ctrl+C 后打印全文）"),
    }
    println!("  说话开始录音，按 Ctrl+C 结束并保存...");
    println!("─────────────────────────────────────────────");

    // 事件回调：Partial 单行覆盖，Final 换行保留
    // 注意：回调必须 'static，output 以 owned 值捕获
    let output_owned = output.map(|s| s.to_string());
    let on_event: Arc<dyn Fn(DictationEvent) + Send + Sync> = Arc::new(move |event| match event {
        DictationEvent::Started => {
            println!("◉ 麦克风就绪，开始聆听...");
        }
        DictationEvent::Partial { text, elapsed_ms } => {
            // \r 回到行首覆盖上一条 partial；补空格清除残留
            let pad = " ".repeat(8);
            print!("\r[{:>5.1}s] ○ {}{}", elapsed_ms as f64 / 1000.0, text, pad);
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        DictationEvent::Final { text, elapsed_ms } => {
            println!("\r[{:>5.1}s] ● {}", elapsed_ms as f64 / 1000.0, text);
        }
        DictationEvent::Stopped { full_text, duration_ms: _, saved_path } => {
            println!("─────────────────────────────────────────────");
            let line_count = full_text.lines().count();
            println!("听写结束：{} 段定稿文本", line_count);
            match saved_path {
                Some(path) => println!("转写已保存: {:?}", path),
                // 未指定输出路径时，终端打印全文供复制
                None if !full_text.is_empty() => {
                    println!("────── 全文 ──────");
                    println!("{}", full_text);
                }
                None => {}
            }
        }
        DictationEvent::Error(msg) => {
            eprintln!("听写错误: {}", msg);
        }
    });

    let mut handle = use_case.start(model, output_owned.as_deref().map(std::path::Path::new), on_event)?;

    // Ctrl+C → 置停止令牌（捕获 owned 的 Arc<AtomicBool>，满足 'static），
    // 工作线程冲刷残余音频并落盘后自然退出
    let stop_flag = handle.stop_flag();
    if let Err(e) = ctrlc::set_handler(move || {
        stop_flag.store(true, Ordering::SeqCst);
        eprintln!();
        eprintln!("收到中断信号，正在停止听写...");
    }) {
        tracing::warn!("注册 Ctrl+C 处理器失败: {}", e);
    }

    // 主线程等待用户 Ctrl+C（停止令牌由信号处理器置位），随后等待收尾
    while !handle.is_stopping() {
        std::thread::sleep(Duration::from_millis(200));
    }
    handle.stop_and_join();

    Ok(())
}

#[cfg(test)]
mod tests {
    /// 事件打印逻辑的纯函数部分：行数统计
    #[test]
    fn dictate_全文行数统计() {
        let text = "第一段\n第二段\n第三段";
        assert_eq!(text.lines().count(), 3);
        assert_eq!("".lines().count(), 0);
    }
}
