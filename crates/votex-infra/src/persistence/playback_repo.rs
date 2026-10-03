//! 播放进度仓储（T6 最小听书播放器）
//!
//! 表 `playback_progress`：按音频文件路径记录上次播放位置，
//! GUI 打开同一文件时自动续播（断点续听）。

use rusqlite::params;
use std::sync::{Arc, Mutex};

/// 一条播放进度
#[derive(Debug, Clone, PartialEq)]
pub struct PlaybackProgress {
    /// 上次播放位置（秒）
    pub position_sec: f32,
    /// 文件总时长（秒，0 = 未知）
    pub duration_sec: f32,
    pub updated_at: String,
}

/// SQLite 播放进度仓储
pub struct SqlitePlaybackRepository {
    conn: Arc<Mutex<rusqlite::Connection>>,
}

impl SqlitePlaybackRepository {
    pub fn new(conn: Arc<Mutex<rusqlite::Connection>>) -> Self {
        Self { conn }
    }

    /// 读取某文件的播放进度（无记录返回 None）
    pub fn get(&self, file_path: &str) -> Result<Option<PlaybackProgress>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("获取数据库连接失败: {}", e))?;
        conn.query_row(
            "SELECT position_sec, duration_sec, updated_at
             FROM playback_progress WHERE file_path = ?1",
            params![file_path],
            |row| {
                Ok(PlaybackProgress {
                    position_sec: row.get(0)?,
                    duration_sec: row.get(1)?,
                    updated_at: row.get(2)?,
                })
            },
        )
        .map(Some)
        .or_else(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => Ok(None),
            other => Err(format!("查询播放进度失败: {}", other)),
        })
    }

    /// 保存（upsert）某文件的播放进度
    pub fn save(&self, file_path: &str, position_sec: f32, duration_sec: f32) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("获取数据库连接失败: {}", e))?;
        conn.execute(
            "INSERT INTO playback_progress (file_path, position_sec, duration_sec, updated_at)
             VALUES (?1, ?2, ?3, datetime('now','localtime'))
             ON CONFLICT(file_path) DO UPDATE SET
                position_sec = excluded.position_sec,
                duration_sec = excluded.duration_sec,
                updated_at = excluded.updated_at",
            params![file_path, position_sec, duration_sec],
        )
        .map_err(|e| format!("保存播放进度失败: {}", e))?;
        Ok(())
    }

    /// 清除某文件的播放进度
    pub fn clear(&self, file_path: &str) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("获取数据库连接失败: {}", e))?;
        conn.execute(
            "DELETE FROM playback_progress WHERE file_path = ?1",
            params![file_path],
        )
        .map_err(|e| format!("清除播放进度失败: {}", e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo() -> SqlitePlaybackRepository {
        let conn = crate::persistence::db::open_in_memory_database().unwrap();
        SqlitePlaybackRepository::new(conn)
    }

    #[test]
    fn 播放进度_保存_读取_清除() {
        let repo = repo();
        assert!(repo.get("a.wav").unwrap().is_none());
        repo.save("a.wav", 12.5, 300.0).unwrap();
        let p = repo.get("a.wav").unwrap().unwrap();
        assert!((p.position_sec - 12.5).abs() < 1e-6);
        assert!((p.duration_sec - 300.0).abs() < 1e-6);
        // upsert 覆盖
        repo.save("a.wav", 45.0, 300.0).unwrap();
        assert!((repo.get("a.wav").unwrap().unwrap().position_sec - 45.0).abs() < 1e-6);
        repo.clear("a.wav").unwrap();
        assert!(repo.get("a.wav").unwrap().is_none());
    }
}
