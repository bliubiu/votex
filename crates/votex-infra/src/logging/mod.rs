//! 日志基础设施：按日轮转的文件输出与脱敏。
//!
//! 格式：`[YYYY-MM-DD HH:MM:SS.SSS] [级别] [线程ID] [模块:行号] - 内容`
//! 级别：DEBUG / INFO / ERROR（默认 INFO）
//! 保留 32 天，过期自动清理。
//! 所有内容经 `Desensitizer::desensitize` 处理后才落盘。

pub mod init;
pub mod desensitizer;
