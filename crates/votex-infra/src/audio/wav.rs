use anyhow::Result;
use std::io::Write;
use votex_domain::shared::value_object::AudioData;

/// WAV 文件读取器
pub fn read_wav(path: &std::path::Path) -> Result<AudioData> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => reader.samples::<i16>().map(|s| s.unwrap_or(0) as f32 / 32768.0).collect(),
            32 => reader.samples::<i32>().map(|s| s.unwrap_or(0) as f32 / 2147483648.0).collect(),
            _ => reader
                .samples::<i32>()
                .map(|s| {
                    let raw = s.unwrap_or(0) as f32;
                    let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
                    (raw / max_val).clamp(-1.0, 1.0)
                })
                .collect(),
        },
    };

    Ok(AudioData {
        samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels as u16,
    })
}

/// WAV 文件写入器
pub struct WavWriter;

impl WavWriter {
    /// 将 AudioData 写入 WAV 文件
    pub fn write(audio: &AudioData, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let num_channels = audio.channels as u16;
        let sample_rate = audio.sample_rate;
        let bits_per_sample: u16 = 16;
        let byte_rate = sample_rate as u32 * num_channels as u32 * (bits_per_sample / 8) as u32;
        let block_align = num_channels * (bits_per_sample / 8);

        // PCM f32 → i16
        let samples_i16: Vec<i16> = audio
            .samples
            .iter()
            .map(|&s| {
                let clamped = s.clamp(-1.0, 1.0);
                (clamped * 32767.0) as i16
            })
            .collect();

        let data_size = samples_i16.len() as u32 * 2;
        let file_size = 36 + data_size;

        let tmp_path = path.with_extension("wav.tmp");
        let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp_path)?);

        // RIFF 头
        file.write_all(b"RIFF")?;
        file.write_all(&file_size.to_le_bytes())?;
        file.write_all(b"WAVE")?;

        // fmt 子块
        file.write_all(b"fmt ")?;
        file.write_all(&16u32.to_le_bytes())?; // 子块大小
        file.write_all(&1u16.to_le_bytes())?; // PCM 格式
        file.write_all(&num_channels.to_le_bytes())?;
        file.write_all(&sample_rate.to_le_bytes())?;
        file.write_all(&byte_rate.to_le_bytes())?;
        file.write_all(&block_align.to_le_bytes())?;
        file.write_all(&bits_per_sample.to_le_bytes())?;

        // data 子块
        file.write_all(b"data")?;
        file.write_all(&data_size.to_le_bytes())?;
        for sample in &samples_i16 {
            file.write_all(&sample.to_le_bytes())?;
        }

        file.flush()?;
        drop(file);
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// 将多个 AudioData 拼接后写入 WAV 文件
    pub fn write_concat(audio_segments: &[AudioData], path: &std::path::Path) -> Result<()> {
        if audio_segments.is_empty() {
            anyhow::bail!("音频段列表为空");
        }

        let sample_rate = audio_segments[0].sample_rate;
        let channels = audio_segments[0].channels;

        let mut all_samples = Vec::new();
        for seg in audio_segments {
            if seg.sample_rate != sample_rate || seg.channels != channels {
                anyhow::bail!("音频段采样率或声道数不一致");
            }
            all_samples.extend_from_slice(&seg.samples);
        }

        let combined = AudioData {
            samples: all_samples,
            sample_rate,
            channels,
        };

        Self::write(&combined, path)
    }
}

/// 流式 WAV 写入器
///
/// 边合成边追加写盘，避免长文本全量内存驻留（10 小时音频不再需要 GB 级缓冲）。
/// RIFF 头先写占位值，finish 时 seek 回填真实大小。
pub struct StreamingWav {
    file: std::io::BufWriter<std::fs::File>,
    sample_rate: u32,
    channels: u16,
    data_bytes: u64,
    path: std::path::PathBuf,
}

impl StreamingWav {
    /// 创建流式写入器并写入 44 字节占位头
    pub fn create(path: &std::path::Path, sample_rate: u32, channels: u16) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(path)?;
        let mut writer = Self {
            file: std::io::BufWriter::new(file),
            sample_rate,
            channels,
            data_bytes: 0,
            path: path.to_path_buf(),
        };
        writer.write_header(0)?;
        Ok(writer)
    }

    fn write_header(&mut self, data_size: u32) -> Result<()> {
        let bits_per_sample: u16 = 16;
        let byte_rate = self.sample_rate * self.channels as u32 * (bits_per_sample / 8) as u32;
        let block_align = self.channels * (bits_per_sample / 8);
        let file_size = 36u32.checked_add(data_size).ok_or_else(|| anyhow::anyhow!("WAV 文件过大"))?;

        let w = &mut self.file;
        w.write_all(b"RIFF")?;
        w.write_all(&file_size.to_le_bytes())?;
        w.write_all(b"WAVE")?;
        w.write_all(b"fmt ")?;
        w.write_all(&16u32.to_le_bytes())?;
        w.write_all(&1u16.to_le_bytes())?; // PCM
        w.write_all(&self.channels.to_le_bytes())?;
        w.write_all(&self.sample_rate.to_le_bytes())?;
        w.write_all(&byte_rate.to_le_bytes())?;
        w.write_all(&block_align.to_le_bytes())?;
        w.write_all(&bits_per_sample.to_le_bytes())?;
        w.write_all(b"data")?;
        w.write_all(&data_size.to_le_bytes())?;
        Ok(())
    }

    /// 追加 f32 采样（自动转 i16）
    pub fn append_f32(&mut self, samples: &[f32]) -> Result<()> {
        // 分块转换，控制瞬时内存
        const CHUNK: usize = 64 * 1024;
        let mut buf = Vec::with_capacity(CHUNK);
        for chunk in samples.chunks(CHUNK) {
            buf.clear();
            for &s in chunk {
                let clamped = s.clamp(-1.0, 1.0);
                buf.push((clamped * 32767.0) as i16);
            }
            for v in &buf {
                self.file.write_all(&v.to_le_bytes())?;
            }
            self.data_bytes += (buf.len() * 2) as u64;
        }
        Ok(())
    }

    /// 追加 N 毫秒静音
    pub fn append_silence_ms(&mut self, duration_ms: u32) -> Result<()> {
        let num_samples = (self.sample_rate as u64 * duration_ms as u64 / 1000
            * self.channels as u64) as usize;
        const CHUNK: usize = 64 * 1024;
        let zero_chunk = vec![0i16; CHUNK.min(num_samples.max(1))];
        let mut remaining = num_samples;
        while remaining > 0 {
            let n = remaining.min(CHUNK);
            for v in &zero_chunk[..n] {
                self.file.write_all(&v.to_le_bytes())?;
            }
            self.data_bytes += (n * 2) as u64;
            remaining -= n;
        }
        Ok(())
    }

    /// 已写入的时长（毫秒）
    pub fn duration_ms(&self) -> u64 {
        let bytes_per_frame = self.channels as u64 * 2;
        if bytes_per_frame == 0 {
            return 0;
        }
        self.data_bytes / bytes_per_frame * 1000 / self.sample_rate as u64
    }

    /// 完成写入：回填 RIFF 头并落盘
    pub fn finish(mut self) -> Result<()> {
        use std::io::{Seek, SeekFrom};
        let data_size = u32::try_from(self.data_bytes)
            .map_err(|_| anyhow::anyhow!("WAV data 超过 4GB 上限"))?;
        self.file.flush()?;
        let file = self.file.get_mut();
        file.seek(SeekFrom::Start(4))?;
        file.write_all(&(36u32.wrapping_add(data_size)).to_le_bytes())?;
        file.seek(SeekFrom::Start(40))?;
        file.write_all(&data_size.to_le_bytes())?;
        file.flush()?;
        Ok(())
    }

    /// 输出路径
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

/// 将多个 WAV 文件流式拼接为一个 WAV 文件（段间可插入静音）
///
/// 用于断点续转：session 目录中已合成的段文件逐个追加到输出，
/// 全程只占用单段级别的内存。
pub fn concat_wav_files(
    wav_paths: &[std::path::PathBuf],
    output_path: &std::path::Path,
    gap_silence_ms: u32,
) -> Result<()> {
    if wav_paths.is_empty() {
        anyhow::bail!("音频段文件列表为空");
    }

    // 用第一段确定参数
    let first = read_wav(&wav_paths[0])?;
    let sample_rate = first.sample_rate;
    let channels = first.channels;
    drop(first);

    let mut writer = StreamingWav::create(output_path, sample_rate, channels)?;
    for (i, p) in wav_paths.iter().enumerate() {
        if i > 0 && gap_silence_ms > 0 {
            writer.append_silence_ms(gap_silence_ms)?;
        }
        let seg = read_wav(p)?;
        if seg.sample_rate != sample_rate || seg.channels != channels {
            anyhow::bail!("音频段采样率或声道数不一致: {:?}", p);
        }
        writer.append_f32(&seg.samples)?;
    }
    writer.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_write_写入静音音频() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wav");
        let audio = AudioData::silence(24000, 1000);

        WavWriter::write(&audio, &path).unwrap();
        assert!(path.exists());

        // 验证文件大小：44 字节头 + 24000 采样 * 2 字节
        let file_size = std::fs::metadata(&path).unwrap().len();
        assert_eq!(file_size, 44 + 24000 * 2);
    }

    #[test]
    fn wav_write_写入非零音频() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wav");
        let audio = AudioData {
            samples: vec![0.5f32, -0.5, 0.25, -0.25],
            sample_rate: 16000,
            channels: 1,
        };

        WavWriter::write(&audio, &path).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn wav_write_concat_拼接音频() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("concat.wav");
        let seg1 = AudioData::silence(24000, 500);
        let seg2 = AudioData::silence(24000, 500);

        WavWriter::write_concat(&[seg1, seg2], &path).unwrap();
        assert!(path.exists());

        let file_size = std::fs::metadata(&path).unwrap().len();
        assert_eq!(file_size, 44 + 24000 * 2); // 1 秒总时长
    }

    #[test]
    fn wav_write_concat_空列表应报错() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.wav");
        assert!(WavWriter::write_concat(&[], &path).is_err());
    }

    #[test]
    fn streaming_wav_流式写入与读取一致() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stream.wav");

        let mut w = StreamingWav::create(&path, 24000, 1).unwrap();
        let samples: Vec<f32> = (0..24000).map(|i| ((i % 100) as f32 / 100.0) - 0.5).collect();
        w.append_f32(&samples).unwrap();
        w.append_silence_ms(500).unwrap();
        w.finish().unwrap();

        // 24000 采样 + 500ms 静音(12000 采样) = 36000 采样
        let file_size = std::fs::metadata(&path).unwrap().len();
        assert_eq!(file_size, 44 + 36000 * 2);

        let audio = read_wav(&path).unwrap();
        assert_eq!(audio.samples.len(), 36000);
        assert_eq!(audio.sample_rate, 24000);
        // 静音段为 0
        assert!(audio.samples[24000..].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn streaming_wav_多段追加时长正确() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.wav");
        let mut w = StreamingWav::create(&path, 16000, 1).unwrap();
        w.append_f32(&vec![0.1f32; 16000]).unwrap(); // 1s
        w.append_f32(&vec![0.2f32; 16000]).unwrap(); // 1s
        assert_eq!(w.duration_ms(), 2000);
        w.finish().unwrap();
        let audio = read_wav(&path).unwrap();
        assert_eq!(audio.duration_ms(), 2000);
    }

    #[test]
    fn concat_wav_files_流式拼接文件() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("a.wav");
        let p2 = dir.path().join("b.wav");
        WavWriter::write(&AudioData::silence(24000, 500), &p1).unwrap();
        WavWriter::write(&AudioData::silence(24000, 500), &p2).unwrap();

        let out = dir.path().join("joined.wav");
        concat_wav_files(&[p1, p2], &out, 200).unwrap();

        // 500+500+200ms
        let audio = read_wav(&out).unwrap();
        assert_eq!(audio.duration_ms(), 1200);
    }
}
