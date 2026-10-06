//! 仓储层错误治理辅助
//!
//! ## 背景
//!
//! 各仓储历史上普遍使用如下写法静默吞掉数据库错误：
//!
//! ```text
//! let conn = match self.conn.lock() { Ok(c) => c, Err(_) => return Vec::new() };
//! let mut stmt = match conn.prepare(...) { Ok(s) => s, Err(_) => return Vec::new() };
//! rows.filter_map(|r| r.ok()).collect()
//! ```
//!
//! 后果是调用方**无法区分「没有数据」与「数据库坏了」**。最危险的是
//! `find_by_id()` 用 `.ok()`：一次瞬时 IO 错误会让调用方认为任务不存在，
//! 从而重复创建任务、重复下载 GB 级模型；`find_incomplete()` 返回空 Vec
//! 会让启动恢复逻辑静默跳过所有待办任务。
//!
//! ## 治理方式
//!
//! 1. 锁中毒不再 panic —— 一次局部 panic 不应让整个仓储永久不可用
//! 2. 无法返回 `Result` 的查询在降级前必须 `tracing::error!` 记录，
//!    保证错误可观测
//! 3. `find_by_id` 区分「查不到行」（正常）与「查询出错」（异常），
//!    后者必须记日志

use rusqlite::Connection;
use std::sync::{Arc, Mutex, MutexGuard};

/// 获取互斥锁
///
/// 与直接 `lock().unwrap()` 的区别：锁中毒时**恢复使用**而非 panic。
/// `std::sync::Mutex` 在任一持有者 panic 后会变为 poisoned，
/// 若此处 panic，一次局部崩溃会横向传染到所有无关的仓储操作。
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// 获取数据库连接（[`lock`] 在 `Arc<Mutex<Connection>>` 上的便捷封装）
pub fn lock_conn(conn: &Arc<Mutex<Connection>>) -> MutexGuard<'_, Connection> {
    lock(conn)
}

/// 记录仓储操作失败日志
///
/// 用于签名不便返回 `Result` 的查询方法：降级返回空结果前必须留下痕迹，
/// 避免调用方把"数据库错误"误判为"没有数据"。
pub fn log_err(op: &str, e: &dyn std::fmt::Display) {
    tracing::error!("数据库操作失败 [{}]: {}", op, e);
}
