/// 日志输出格式回归测试
///
/// 覆盖两个曾在 E2E 验证中暴露的问题：
/// 1. **事件间缺少换行**（F58）——`CustomFormatter` 漏写 `\n`，导致整个日志文件挤成一行，
///    `Select-String` 等按行工具完全失效，日志类用例无法执行。
/// 2. **中文可断言性**——验证中文日志能按 UTF-8 原样读回，
///    使 G0-10、SYS-02/03/04 等用例的中文断言可用。
///
/// 全局 tracing subscriber 只能安装一次，因此本文件仅一个用例，
/// 把换行、格式、中文、脱敏四类断言合并执行。
///
/// 运行方式:
/// ```bash
/// cargo test -p votex-infra --test logging_line_break_test -- --nocapture
/// ```
use std::path::Path;

/// 读取临时目录下的日志文件（日志名为 `votex-YYYYMMDD.log`）
fn read_log(dir: &Path) -> String {
    let entry = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("读取日志目录失败: {}", e))
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().is_some_and(|x| x == "log"))
        .unwrap_or_else(|| {
            panic!(
                "未生成日志文件，目录实况: {}",
                votex_infra::shared::ModelFileLocator::list_dir(dir)
            )
        });
    std::fs::read_to_string(entry.path())
        .unwrap_or_else(|e| panic!("日志文件非 UTF-8，无法按行分析: {}", e))
}

fn non_empty_lines(content: &str) -> Vec<&str> {
    content.lines().filter(|l| !l.trim().is_empty()).collect()
}

#[test]
fn 日志格式换行中文与脱敏() {
    let dir = tempfile::tempdir().expect("创建临时目录失败");
    let dir_path = dir.path().to_string_lossy().to_string();

    votex_infra::logging::init::init(&dir_path, "debug", true).expect("日志系统初始化失败");

    // 回归 F59：重复初始化不得 panic（全局 subscriber 是进程级单例）
    votex_infra::logging::init::init(&dir_path, "debug", true).expect("重复初始化日志系统失败");
    votex_infra::logging::init::init(&dir_path, "debug", true).expect("第三次初始化日志系统失败");

    // `init` 自身会写入一条初始化日志，此处作为基线行数
    let baseline = non_empty_lines(&read_log(dir.path())).len();

    // ─── 1. 三条事件（含中文 / 中英混排 / 不同级别） ───
    tracing::info!("第一行中文断言点");
    tracing::warn!("第二行 mixed 中英文 123");
    tracing::error!("第三行 error 级别");

    let content = read_log(dir.path());
    let lines = non_empty_lines(&content);

    assert_eq!(
        lines.len(),
        baseline + 3,
        "每条事件应独占一行（回归 F58：漏写换行会让全部日志挤成一行）\n实际行数: {}\n日志开头: {:?}",
        lines.len(),
        &content[..content.len().min(300)]
    );

    // ─── 2. 中文必须能原样读回（UTF-8） ───
    let new_lines = &lines[baseline..];
    assert!(
        new_lines[0].contains("第一行中文断言点"),
        "中文日志内容损坏: {:?}",
        new_lines[0]
    );
    assert!(
        new_lines[1].contains("mixed 中英文 123"),
        "中英混排内容损坏: {:?}",
        new_lines[1]
    );

    // ─── 3. 级别标记与格式前缀 ───
    assert!(new_lines[0].contains("[INFO]"), "级别标记缺失: {:?}", new_lines[0]);
    assert!(new_lines[1].contains("[WARN]"), "级别标记缺失: {:?}", new_lines[1]);
    assert!(new_lines[2].contains("[ERROR]"), "级别标记缺失: {:?}", new_lines[2]);
    for l in new_lines {
        assert!(
            l.starts_with('[') && l.contains("[ThreadId(") && l.contains("] - "),
            "日志格式应符合 [时间] [级别] [线程ID] [模块:行号] - 内容: {:?}",
            l
        );
    }

    // Windows 上应为 CRLF，保证记事本正常换行
    if cfg!(target_os = "windows") {
        assert!(
            content.contains("\r\n"),
            "Windows 下日志须使用 CRLF 换行"
        );
    }

    // ─── 4. 脱敏不得破坏行结构 ───
    let before = non_empty_lines(&content).len();
    tracing::info!("联系邮箱 user@example.com 与 IP 192.168.1.100");

    let content2 = read_log(dir.path());
    assert!(
        !content2.contains("user@example.com"),
        "邮箱未被脱敏: {:?}",
        content2
    );
    assert!(
        !content2.contains("192.168.1.100"),
        "IP 未被脱敏: {:?}",
        content2
    );
    assert!(
        content2.contains("u***@example.com"),
        "邮箱脱敏结果缺失: {:?}",
        content2.lines().last()
    );
    assert_eq!(
        non_empty_lines(&content2).len(),
        before + 1,
        "脱敏后仍须独占一行"
    );
}
