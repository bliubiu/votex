//! 播放器纯逻辑
//!
//! # 为什么要单独抽一层
//!
//! `audio_player.rs` 里混着三类东西：
//!
//! 1. **纯计算**（时间格式化、章节定位、续播位置夹取）——可测
//! 2. **状态变更**（`open_file` / `stop`）——不碰 egui 即可测
//! 3. **渲染 + 音频 IO**（`AudioPlayer::show`、`audio_thread`）——难测
//!
//! 直接在 `audio_player.rs` 里写测试会被 egui 类型与全局音频单例
//! （`AUDIO_TX` / `AUDIO_STATUS` 是 `OnceLock`，同一进程内只能初始化一次）
//! 拖住。本模块把 1、2 抽出来，测试时不触碰任何全局单例。
//!
//! 渲染层只负责「读状态 → 画控件 → 写回状态」，
//! 计算规则一律委托到这里，避免同一规则在两处漂移。

use std::path::Path;

/// 章节标记（来自合成时写出的 `{音频}.chapters.json`）
#[derive(Debug, Clone, PartialEq)]
pub struct ChapterMark {
    pub title: String,
    pub start_sec: f32,
    pub end_sec: f32,
}

// ===== 时间格式化 =====

/// 把秒数格式化为 `HH:MM:SS`
///
/// 负数按 0 处理（浮点误差或seek 越界时不应显示 `-1`）。
/// 超过 99 小时不会截断——`{:02}` 只是最小宽度。
pub fn format_secs(s: f32) -> String {
    let total = s.max(0.0) as u64;
    format!(
        "{:02}:{:02}:{:02}",
        total / 3600,
        (total % 3600) / 60,
        total % 60
    )
}

// ===== 章节定位 =====

/// 按播放位置定位当前章节
///
/// 采用「左闭右开」区间 `[start, end)`，与 `chapters.json` 的语义一致。
/// 位置落在所有章节之外（开头留白 / 末尾空隙）时返回 `None`，
/// 由调用方决定保持原选中项还是清空。
pub fn chapter_at(chapters: &[ChapterMark], position: f32) -> Option<usize> {
    chapters
        .iter()
        .position(|c| position >= c.start_sec && position < c.end_sec)
}

/// 取播放总时长
///
/// 优先用最后一章的结束时间——章节表是合成时写出的权威时长，
/// 比解码器上报的 duration 更准确（后者受 VBR 编码影响）。
/// 章节表为空时返回 `0.0`。
pub fn total_duration(chapters: &[ChapterMark]) -> f32 {
    chapters.last().map(|c| c.end_sec).unwrap_or(0.0)
}

// ===== 续播位置 =====

/// 夹取续播位置
///
/// # 为什么需要
///
/// 断点续听读出的位置可能越界：
/// - 上次听到 500s，但用户后来把文件截短成 300s → 应从 0 开始
/// - 章节表缺失导致 `known_duration` 为 0 → 此时不做上界限制
///
/// 原实现写成 `position.min(known_duration.max(position))`，
/// 恒等于 `position`，夹取逻辑实际从未生效。
pub fn clamp_resume_position(
    saved_position: f32,
    known_duration: f32,
) -> f32 {
    if saved_position <= 0.0 {
        return 0.0;
    }
    if known_duration <= 0.0 {
        // 时长未知，不做上界限制
        return saved_position;
    }
    saved_position.min(known_duration)
}

// ===== 章节表加载 =====

/// 加载 `{音频}.chapters.json`
///
/// 文件不存在、JSON 非法、字段缺失一律返回空列表——
/// 章节表是锦上添花的功能，解析失败不应阻止播放。
pub fn load_chapters(audio_path: &str) -> Vec<ChapterMark> {
    let json_path = Path::new(audio_path).with_extension("chapters.json");
    let Ok(content) = std::fs::read_to_string(&json_path) else {
        return Vec::new();
    };
    #[derive(serde::Deserialize)]
    struct Entry {
        title: String,
        start_ms: u64,
        end_ms: u64,
    }
    serde_json::from_str::<Vec<Entry>>(&content)
        .map(|entries| {
            entries
                .into_iter()
                .map(|e| ChapterMark {
                    title: e.title,
                    start_sec: e.start_ms as f32 / 1000.0,
                    end_sec: e.end_ms as f32 / 1000.0,
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ch(title: &str, start: f32, end: f32) -> ChapterMark {
        ChapterMark {
            title: title.to_string(),
            start_sec: start,
            end_sec: end,
        }
    }

    // ===== format_secs =====

    #[test]
    fn format_secs_补零到两位() {
        assert_eq!(format_secs(0.0), "00:00:00");
        assert_eq!(format_secs(5.0), "00:00:05");
        assert_eq!(format_secs(65.0), "00:01:05");
        assert_eq!(format_secs(3600.0), "01:00:00");
        assert_eq!(format_secs(3661.0), "01:01:01");
    }

    #[test]
    fn format_secs_负数按零处理() {
        // seek 越界或浮点误差可能产生负值，不应显示 "-1:59:59"
        assert_eq!(format_secs(-1.0), "00:00:00");
        assert_eq!(format_secs(-0.5), "00:00:00");
    }

    #[test]
    fn format_secs_小数截断而非四舍五入() {
        // 59.9s 显示 59s：进度条每秒刷新，四舍五入会造成视觉抖动
        assert_eq!(format_secs(59.9), "00:00:59");
        assert_eq!(format_secs(1.999), "00:00:01");
    }

    // ===== chapter_at =====

    #[test]
    fn chapter_at_命中区间() {
        let chapters = vec![ch("一", 0.0, 10.0), ch("二", 10.0, 20.0), ch("三", 20.0, 30.0)];
        assert_eq!(chapter_at(&chapters, 0.0), Some(0));
        assert_eq!(chapter_at(&chapters, 5.0), Some(0));
        assert_eq!(chapter_at(&chapters, 10.0), Some(1));
        assert_eq!(chapter_at(&chapters, 25.0), Some(2));
    }

    #[test]
    fn chapter_at_边界为左闭右开() {
        let chapters = vec![ch("一", 0.0, 10.0), ch("二", 10.0, 20.0)];
        // 章节切换点应归属后一章，避免第一章被重复高亮
        assert_eq!(chapter_at(&chapters, 10.0), Some(1));
        // 最后一章的右边界不属于任何章节
        assert_eq!(chapter_at(&chapters, 20.0), None);
    }

    #[test]
    fn chapter_at_空章节表返回无() {
        assert_eq!(chapter_at(&[], 5.0), None);
    }

    #[test]
    fn chapter_at_间隙位置返回无() {
        // 章节表条目之间可能有留白（如片头曲与正片之间）
        let chapters = vec![ch("片头", 0.0, 10.0), ch("正片", 30.0, 100.0)];
        assert_eq!(chapter_at(&chapters, 20.0), None);
    }

    #[test]
    fn chapter_at_未排序章节表按首个命中返回() {
        // 容忍脏数据：不去要求调用方预先排序
        let chapters = vec![ch("后段", 50.0, 60.0), ch("前段", 0.0, 10.0)];
        assert_eq!(chapter_at(&chapters, 5.0), Some(1));
    }

    // ===== total_duration =====

    #[test]
    fn total_duration_取最后章节结束时间() {
        let chapters = vec![ch("一", 0.0, 10.5), ch("二", 10.5, 25.25)];
        assert_eq!(total_duration(&chapters), 25.25);
    }

    #[test]
    fn total_duration_空表为零() {
        assert_eq!(total_duration(&[]), 0.0);
    }

    // ===== clamp_resume_position =====

    #[test]
    fn clamp_正常位置原样返回() {
        assert_eq!(clamp_resume_position(30.0, 100.0), 30.0);
    }

    #[test]
    fn clamp_超出时长夹到末尾() {
        // 上次听到 500s，文件后来被截短到 300s
        assert_eq!(clamp_resume_position(500.0, 300.0), 300.0);
    }

    #[test]
    fn clamp_时长未知时不限制上界() {
        // 章节表缺失 → known_duration = 0，此时不应把位置压成 0
        assert_eq!(clamp_resume_position(500.0, 0.0), 500.0);
    }

    #[test]
    fn clamp_负位置归零() {
        assert_eq!(clamp_resume_position(-10.0, 100.0), 0.0);
    }

    #[test]
    fn clamp_恰好等于末尾不越界() {
        assert_eq!(clamp_resume_position(100.0, 100.0), 100.0);
    }

    // ===== load_chapters =====

    #[test]
    fn load_chapters_文件不存在返回空() {
        let dir = std::env::temp_dir().join("votex_gui_test_no_such");
        let _ = std::fs::remove_dir_all(&dir);
        let audio = dir.join("nonexistent.wav");
        assert!(load_chapters(&audio.to_string_lossy()).is_empty());
    }

    #[test]
    fn load_chapters_解析毫秒转秒() {
        let dir = std::env::temp_dir().join("votex_gui_test_chapters");
        std::fs::create_dir_all(&dir).unwrap();
        let audio = dir.join("book.wav");
        std::fs::write(
            audio.with_extension("chapters.json"),
            r#"[{"title":"序章","start_ms":0,"end_ms":1500},
                {"title":"第一章","start_ms":1500,"end_ms":12500}]"#,
        )
        .unwrap();

        let chapters = load_chapters(&audio.to_string_lossy());
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[0].title, "序章");
        assert_eq!(chapters[0].start_sec, 0.0);
        assert_eq!(chapters[0].end_sec, 1.5);
        assert_eq!(chapters[1].end_sec, 12.5);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_chapters_非法JSON返回空而非报错() {
        // 章节表是可选功能，解析失败不应阻止播放
        let dir = std::env::temp_dir().join("votex_gui_test_badjson");
        std::fs::create_dir_all(&dir).unwrap();
        let audio = dir.join("broken.wav");
        std::fs::write(audio.with_extension("chapters.json"), "{ not json").unwrap();

        assert!(load_chapters(&audio.to_string_lossy()).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_chapters_缺字段返回空() {
        let dir = std::env::temp_dir().join("votex_gui_test_missing_field");
        std::fs::create_dir_all(&dir).unwrap();
        let audio = dir.join("partial.wav");
        std::fs::write(
            audio.with_extension("chapters.json"),
            r#"[{"title":"只有标题"}]"#,
        )
        .unwrap();

        assert!(load_chapters(&audio.to_string_lossy()).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_chapters_空数组返回空列表() {
        let dir = std::env::temp_dir().join("votex_gui_test_empty_arr");
        std::fs::create_dir_all(&dir).unwrap();
        let audio = dir.join("none.wav");
        std::fs::write(audio.with_extension("chapters.json"), "[]").unwrap();

        assert!(load_chapters(&audio.to_string_lossy()).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
