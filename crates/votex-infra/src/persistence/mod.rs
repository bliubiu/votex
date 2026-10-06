//! 持久化基础设施：SQLite 仓储实现。
//!
//! 约定：
//! - 所有查询用参数化 SQL，无字符串拼接
//! - 查询失败返回 `Result`，**禁止**静默返回空集合
//! - 启用 `PRAGMA foreign_keys`，否则 `ON DELETE CASCADE` 形同虚设
//! - 加锁统一走 `guard::lock_conn()`，锁中毒时恢复而非 panic

pub mod config_repo;
pub mod db;
pub mod guard;
pub mod model_repo;
pub mod tts_task_repo;
pub mod asr_task_repo;
pub mod task_repo;
pub mod pipeline_repo;
pub mod ocr_repo;
pub mod download_repo;
pub mod translation_repo;
pub mod playback_repo;
