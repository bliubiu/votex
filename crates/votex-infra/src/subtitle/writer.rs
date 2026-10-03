use anyhow::Result;
use votex_domain::asr::value_object::{SubtitleEntry, SubtitleFormat};

/// 字幕文件写入器
pub struct SubtitleWriter;

impl SubtitleWriter {
    /// 将字幕条目写入指定格式的文件
    pub fn write(entries: &[SubtitleEntry], format: SubtitleFormat, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = match format {
            SubtitleFormat::Srt => Self::to_srt(entries),
            SubtitleFormat::Lrc => Self::to_lrc(entries),
            SubtitleFormat::Txt => Self::to_txt(entries),
        };

        let tmp_path = path.with_extension("tmp");
        std::fs::write(&tmp_path, &content)?;
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// 生成 SRT 格式内容
    pub fn to_srt(entries: &[SubtitleEntry]) -> String {
        entries
            .iter()
            .map(|e| {
                format!(
                    "{}\n{} --> {}\n{}\n",
                    e.index,
                    e.start_time.to_srt_format(),
                    e.end_time.to_srt_format(),
                    e.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 生成 LRC 格式内容
    pub fn to_lrc(entries: &[SubtitleEntry]) -> String {
        entries
            .iter()
            .map(|e| format!("{}{}", e.start_time.to_lrc_format(), e.text))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 生成纯文本内容
    pub fn to_txt(entries: &[SubtitleEntry]) -> String {
        entries
            .iter()
            .map(|e| e.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use votex_domain::asr::value_object::Timestamp;

    fn sample_entries() -> Vec<SubtitleEntry> {
        vec![
            SubtitleEntry {
                index: 1,
                start_time: Timestamp::from_millis(0),
                end_time: Timestamp::from_millis(3000),
                text: "你好世界".to_string(),
            },
            SubtitleEntry {
                index: 2,
                start_time: Timestamp::from_millis(3000),
                end_time: Timestamp::from_millis(6500),
                text: "这是声阅".to_string(),
            },
        ]
    }

    #[test]
    fn to_srt_格式正确() {
        let srt = SubtitleWriter::to_srt(&sample_entries());
        assert!(srt.contains("1\n"));
        assert!(srt.contains("00:00:00,000 --> 00:00:03,000"));
        assert!(srt.contains("你好世界"));
        assert!(srt.contains("2\n"));
        assert!(srt.contains("00:00:03,000 --> 00:00:06,500"));
        assert!(srt.contains("这是声阅"));
    }

    #[test]
    fn to_lrc_格式正确() {
        let lrc = SubtitleWriter::to_lrc(&sample_entries());
        assert!(lrc.contains("[00:00.00]你好世界"));
        assert!(lrc.contains("[00:03.00]这是声阅"));
    }

    #[test]
    fn to_txt_纯文本() {
        let txt = SubtitleWriter::to_txt(&sample_entries());
        assert!(txt.contains("你好世界"));
        assert!(txt.contains("这是声阅"));
        assert!(!txt.contains("-->"));
    }

    #[test]
    fn write_写入srt文件() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.srt");
        SubtitleWriter::write(&sample_entries(), SubtitleFormat::Srt, &path).unwrap();
        assert!(path.exists());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("你好世界"));
    }
}
