//! 共享数据库连接管理
//!
//! 统一管理 SQLite 连接生命周期、PRAGMA 优化和表结构初始化。
//! 所有 Repository 共享同一个 `Arc<Mutex<Connection>>`，避免多连接锁争用。

use rusqlite::Connection;
use std::path::Path;
use std::sync::{Arc, Mutex};
use votex_domain::error::DomainError;

/// 打开（或创建）共享数据库文件
pub fn open_database(db_path: &Path) -> Result<Arc<Mutex<Connection>>, DomainError> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            DomainError::Config(crate::config_err(&format!("创建目录失败: {}", e)))
        })?;
    }

    let conn = Connection::open(db_path).map_err(|e| {
        DomainError::Config(crate::config_err(&format!("打开数据库失败: {}", e)))
    })?;

    init_schema(&conn)?;

    tracing::info!("数据库已打开: {:?}", db_path);
    Ok(Arc::new(Mutex::new(conn)))
}

/// 打开内存数据库（用于测试）
pub fn open_in_memory_database() -> Result<Arc<Mutex<Connection>>, DomainError> {
    let conn = Connection::open_in_memory().map_err(|e| {
        DomainError::Config(crate::config_err(&format!("打开内存数据库失败: {}", e)))
    })?;

    init_schema(&conn)?;

    Ok(Arc::new(Mutex::new(conn)))
}

/// 初始化 PRAGMA 优化 + 全部表结构
fn init_schema(conn: &Connection) -> Result<(), DomainError> {
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA busy_timeout=5000;
         PRAGMA cache_size=-8000;
         PRAGMA temp_store=MEMORY;

         CREATE TABLE IF NOT EXISTS tts_tasks (
            id          TEXT PRIMARY KEY,
            data_json   TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'Queued',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tts_tasks_status ON tts_tasks(status);

        CREATE TABLE IF NOT EXISTS asr_tasks (
            id          TEXT PRIMARY KEY,
            data_json   TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'Queued',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_asr_tasks_status ON asr_tasks(status);

        CREATE TABLE IF NOT EXISTS pipelines (
            id          TEXT PRIMARY KEY,
            data_json   TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'Idle',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_pipelines_status ON pipelines(status);

        CREATE TABLE IF NOT EXISTS ocr_tasks (
            id          TEXT PRIMARY KEY,
            data_json   TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'Queued',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ocr_pages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id     TEXT NOT NULL REFERENCES ocr_tasks(id) ON DELETE CASCADE,
            page_index  INTEGER NOT NULL,
            status      TEXT NOT NULL DEFAULT 'Pending',
            blocks_json TEXT,
            confidence  REAL,
            retry_count INTEGER NOT NULL DEFAULT 0,
            error_msg   TEXT,
            UNIQUE(task_id, page_index)
        );
        CREATE INDEX IF NOT EXISTS idx_ocr_pages_task_id ON ocr_pages(task_id);
        CREATE INDEX IF NOT EXISTS idx_ocr_tasks_status ON ocr_tasks(status);

        CREATE TABLE IF NOT EXISTS models (
            id          TEXT PRIMARY KEY,
            data_json   TEXT NOT NULL,
            kind        TEXT NOT NULL,
            status      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_models_kind ON models(kind);
        CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);

        CREATE TABLE IF NOT EXISTS downloads (
            model_id        TEXT NOT NULL,
            file_name       TEXT NOT NULL,
            url             TEXT NOT NULL,
            bytes_downloaded INTEGER NOT NULL DEFAULT 0,
            total_bytes     INTEGER NOT NULL DEFAULT 0,
            status          TEXT NOT NULL DEFAULT 'Pending',
            error_message   TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            PRIMARY KEY (model_id, file_name)
        );
        CREATE INDEX IF NOT EXISTS idx_downloads_status ON downloads(status);
        CREATE INDEX IF NOT EXISTS idx_downloads_model ON downloads(model_id);

        CREATE TABLE IF NOT EXISTS translation_tasks (
            id           TEXT PRIMARY KEY,
            data_json    TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'Queued',
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_translation_tasks_status ON translation_tasks(status);

        CREATE TABLE IF NOT EXISTS translation_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            engine       TEXT NOT NULL,
            direction    TEXT NOT NULL,
            source_text  TEXT NOT NULL,
            target_text  TEXT NOT NULL,
            glossary_fp  INTEGER NOT NULL DEFAULT 0,
            created_at   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_translation_history_lookup
            ON translation_history(engine, direction, glossary_fp);

        CREATE TABLE IF NOT EXISTS playback_progress (
            file_path    TEXT PRIMARY KEY,
            position_sec REAL NOT NULL DEFAULT 0,
            duration_sec REAL NOT NULL DEFAULT 0,
            updated_at   TEXT NOT NULL
        );",
    )
    .map_err(|e| DomainError::Config(crate::config_err(&format!("初始化数据库失败: {}", e))))?;

    // 迁移：为旧版 ocr_tasks 添加 status 列（如已存在则静默跳过）
    let _ = conn.execute_batch(
        "ALTER TABLE ocr_tasks ADD COLUMN status TEXT NOT NULL DEFAULT 'Queued';",
    );

    Ok(())
}
