use anyhow::Result;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use tracing_subscriber::fmt::format::{self, FormatFields};
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::fmt::{FmtContext, FormatEvent};
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::EnvFilter;

use super::desensitizer::Desensitizer;

/// 日志保留天数
const MAX_LOG_DAYS: i64 = 32;

/// 当前日期字符串 YYYYMMDD
fn today_str() -> String {
    chrono::Local::now().format("%Y%m%d").to_string()
}

/// 清理超过保留天数的日志文件
fn cleanup_expired_logs(log_dir: &PathBuf) {
    let cutoff = chrono::Local::now() - chrono::Duration::days(MAX_LOG_DAYS);
    let cutoff_str = cutoff.format("%Y%m%d").to_string();

    let Ok(entries) = std::fs::read_dir(log_dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if let Some(date_part) = filename
            .strip_prefix("votex-")
            .and_then(|s| s.strip_suffix(".log"))
        {
            if date_part < cutoff_str.as_str() {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}

/// 文件日志写入器（通过 MakeWriter 为每条日志事件创建）
struct FileMakeWriter {
    state: Mutex<FileState>,
}

struct FileState {
    log_dir: PathBuf,
    current_date: String,
    current_file: Option<std::fs::File>,
}

impl<'a> MakeWriter<'a> for FileMakeWriter {
    type Writer = FileLogWriter<'a>;

    fn make_writer(&'a self) -> Self::Writer {
        FileLogWriter {
            state: &self.state,
        }
    }
}

/// 单条日志写入器
struct FileLogWriter<'a> {
    state: &'a Mutex<FileState>,
}

impl Write for FileLogWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let today = today_str();

        // 日期变更或文件未打开时轮转
        if state.current_date != today || state.current_file.is_none() {
            if state.current_date != today {
                state.current_date = today.clone();
                cleanup_expired_logs(&state.log_dir);
            }
            let path = state.log_dir.join(format!("votex-{}.log", today));
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?;
            state.current_file = Some(file);
        }

        if let Some(ref mut f) = state.current_file {
            // 脱敏处理
            let input = String::from_utf8_lossy(buf);
            let desensitized = Desensitizer::desensitize(&input);

            // Windows 平台 CRLF 转换，确保记事本正常换行
            let output = if cfg!(target_os = "windows") {
                desensitized.replace('\n', "\r\n")
            } else {
                desensitized
            };

            // 将完整的（可能更长）处理结果写入文件
            let _ = f.write_all(output.as_bytes())?;

            // 每条日志事件后立即刷新，防止日志丢失
            let _ = f.flush();

            // 返回原始输入的长度（tracing write_all 用此偏移量切片 buf）
            Ok(buf.len())
        } else {
            Ok(0)
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(ref mut f) = state.current_file {
            f.flush()
        } else {
            Ok(())
        }
    }
}

/// 自定义时间格式：[YYYY-MM-DD HH:MM:SS.SSS]
struct CustomTime;

impl FormatTime for CustomTime {
    fn format_time(&self, w: &mut format::Writer<'_>) -> std::fmt::Result {
        let now = chrono::Local::now();
        write!(w, "{}", now.format("%Y-%m-%d %H:%M:%S%.3f"))
    }
}

/// 自定义日志格式：[时间] [级别] [线程ID] [模块:行号] - 内容
struct CustomFormatter;

impl<S, N> FormatEvent<S, N> for CustomFormatter
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    N: for<'writer> FormatFields<'writer> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        write!(writer, "[")?;
        CustomTime.format_time(&mut writer)?;
        write!(writer, "] [{}]", event.metadata().level())?;
        write!(writer, " [ThreadId({:?})]", std::thread::current().id())?;
        write!(writer, " [{}", event.metadata().target())?;
        if let Some(line) = event.metadata().line() {
            write!(writer, ":{}", line)?;
        }
        write!(writer, "] - ")?;
        // 用 by_ref 借出 writer，保留所有权以便收尾时补换行
        ctx.format_fields(writer.by_ref(), event)?;
        // 每条事件必须以换行收尾：标准 Format 实现会写 '\n'，自定义实现若遗漏，
        // 整个日志文件会挤成一行（无法按行分析、文件体积异常膨胀）。
        // 写出的 '\n' 随后由 FileLogWriter 在 Windows 上转成 "\r\n"。
        write!(writer, "\n")
    }
}

/// 脱敏 stderr 写入器：写入前经过 Desensitizer 处理
struct DesensitizingStderr;

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for DesensitizingStderr {
    type Writer = DesensitizingStderr;
    fn make_writer(&'a self) -> Self::Writer {
        DesensitizingStderr
    }
}

impl std::io::Write for DesensitizingStderr {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let input = String::from_utf8_lossy(buf);
        let desensitized = Desensitizer::desensitize(&input);
        let stderr = std::io::stderr();
        let mut handle = stderr.lock();
        // 写入完整处理后内容，但返回原始输入长度
        let _ = handle.write_all(desensitized.as_bytes())?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stderr().flush()
    }
}

/// 日志初始化状态（保证全局 subscriber 只安装一次）
///
/// 全局 subscriber 由 `tracing` 以进程级单例管理，重复安装会返回
/// `SetGlobalDefaultError`。原先直接调用 `.init()`（内部 `expect`），
/// 一旦初始化被触发两次就会 panic，把整个程序打断（F59）。
static INIT_STATE: OnceLock<std::result::Result<(), String>> = OnceLock::new();

/// 初始化日志系统（幂等，可重复调用）
///
/// - 首次调用：安装全局 subscriber 并写入一条初始化日志
/// - 重复调用：直接返回成功，不重复安装
/// - 已有其他全局 subscriber（如第三方库先行安装）：降级为 stderr 输出，
///   不视为错误——此时日志仍会被记录，不应因此中断程序
pub fn init(log_dir: &str, level: &str, file_enabled: bool) -> Result<()> {
    let outcome = INIT_STATE.get_or_init(|| install(log_dir, level, file_enabled));
    match outcome {
        Ok(()) => Ok(()),
        Err(e) => Err(anyhow::anyhow!("{}", e)),
    }
}

/// 实际执行 subscriber 安装
fn install(log_dir: &str, level: &str, file_enabled: bool) -> std::result::Result<(), String> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level));

    if file_enabled {
        if let Err(e) = std::fs::create_dir_all(log_dir) {
            return Err(format!("创建日志目录失败: {}", e));
        }

        let log_dir_buf = PathBuf::from(log_dir);
        let writer = FileMakeWriter {
            state: Mutex::new(FileState {
                log_dir: log_dir_buf,
                current_date: String::new(),
                current_file: None,
            }),
        };

        let res = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(writer)
            .event_format(CustomFormatter)
            .try_init();

        return finish_install(res, level);
    }

    let res = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(DesensitizingStderr)
        .event_format(CustomFormatter)
        .try_init();

    finish_install(res, level)
}

/// 收尾：安装成功写初始化日志；失败则降级到 stderr 而非 panic
fn finish_install(
    res: std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>,
    level: &str,
) -> std::result::Result<(), String> {
    match res {
        Ok(()) => {
            tracing::info!("日志系统初始化完成，级别: {}", level);
            Ok(())
        }
        Err(e) => {
            // 已有全局 subscriber（日志不会丢失），仅提示本次文件日志未生效
            eprintln!(
                "警告: 全局日志订阅器已存在（{}），本次文件日志未启用，事件将沿用既有订阅器输出。",
                e
            );
            Ok(())
        }
    }
}
