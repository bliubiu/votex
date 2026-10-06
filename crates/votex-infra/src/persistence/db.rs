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

    // 启用外键约束。
    // SQLite 默认关闭外键，若不显式开启，schema 里声明的
    // `ON DELETE CASCADE`（如 ocr_pages.task_id）完全不生效，
    // 删除主表记录后会留下孤儿数据。该 PRAGMA 是 per-connection 的，
    // 因此每个新建连接都必须执行。
    conn.execute_batch("PRAGMA foreign_keys = ON;")
        .map_err(|e| DomainError::Config(crate::config_err(&format!("启用外键约束失败: {}", e))))?;

    // 校验开关是否真正生效（编译期若未启用 SQLITE_ENABLE_FOREIGN_KEYS 会静默失败）
    let fk_on: bool = conn
        .query_row("PRAGMA foreign_keys;", [], |row| row.get(0))
        .unwrap_or(false);
    if !fk_on {
        tracing::warn!("SQLite 外键约束未能启用，级联删除将不生效");
    }

    run_migrations(conn)?;

    Ok(())
}

/// 按版本顺序执行数据库迁移
///
/// 旧实现只有一条 `let _ = conn.execute_batch("ALTER TABLE ...")`：
/// 既没有版本记录无法判断当前库版本，又用 `let _ =` 吞掉包括
/// SQLITE_BUSY、磁盘满在内的真实失败。这里改为显式的版本化迁移列表，
/// 通过 `PRAGMA user_version` 记录进度，失败必须向上传播。
fn run_migrations(conn: &Connection) -> Result<(), DomainError> {
    let current_version: u32 = conn
        .query_row("PRAGMA user_version;", [], |row| row.get(0))
        .unwrap_or(0);

    // (目标版本号, 迁移 SQL)。新增迁移时追加一项，版本号必须单调递增
    const MIGRATIONS: &[(u32, &str)] = &[(
        1,
        "ALTER TABLE ocr_tasks ADD COLUMN status TEXT NOT NULL DEFAULT 'Queued';",
    )];

    for &(version, sql) in MIGRATIONS {
        if current_version >= version {
            continue;
        }
        match conn.execute_batch(sql) {
            Ok(()) => {
                // user_version 不能用参数绑定，只能拼接；取值来自上面的常量表，无注入风险
                conn.execute_batch(&format!("PRAGMA user_version = {};", version))
                    .map_err(|e| {
                        DomainError::Config(crate::config_err(&format!(
                            "记录迁移版本 {} 失败: {}",
                            version, e
                        )))
                    })?;
                tracing::info!("数据库迁移完成: 版本 {} -> {}", current_version, version);
            }
            Err(e) => {
                // 「列已存在」属于可接受的重复执行；其余错误必须暴露
                let msg = e.to_string();
                if msg.contains("duplicate column name") || msg.contains("already exists") {
                    tracing::debug!("迁移 {} 已应用，跳过: {}", version, msg);
                    continue;
                }
                return Err(DomainError::Config(crate::config_err(&format!(
                    "执行迁移 {} 失败: {}",
                    version, e
                ))));
            }
        }
    }

    Ok(())
}
