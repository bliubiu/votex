//! 文件管理服务
//!
//! 提供输出目录管理、文件命名、临时文件清理、文件哈希计算等功能。

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// 文件管理服务
pub struct FileManagementService {
    output_dir: PathBuf,
    temp_files: HashSet<PathBuf>,
}

impl FileManagementService {
    /// 创建文件管理服务
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        let dir: PathBuf = output_dir.into();
        std::fs::create_dir_all(&dir).ok();
        Self {
            output_dir: dir,
            temp_files: HashSet::new(),
        }
    }

    /// 获取输出目录
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    /// 生成输出文件名
    ///
    /// * `prefix` - 文件名前缀
    /// * `suffix` - 文件扩展名（如 ".wav"）
    /// * `use_timestamp` - 是否添加时间戳
    pub fn generate_filename(&self, prefix: &str, suffix: &str, use_timestamp: bool) -> String {
        if use_timestamp {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            format!("{}_{}{}", prefix, ts, suffix)
        } else {
            format!("{}{}", prefix, suffix)
        }
    }

    /// 获取完整输出路径
    ///
    /// * `prefix` - 文件名前缀
    /// * `suffix` - 文件扩展名
    /// * `subdir` - 可选子目录
    pub fn get_output_path(&self, prefix: &str, suffix: &str, subdir: Option<&str>) -> PathBuf {
        let target_dir = match subdir {
            Some(dir) => {
                let d = self.output_dir.join(dir);
                std::fs::create_dir_all(&d).ok();
                d
            }
            None => self.output_dir.clone(),
        };
        let filename = self.generate_filename(prefix, suffix, true);
        target_dir.join(filename)
    }

    /// 注册临时文件
    pub fn register_temp_file(&mut self, path: PathBuf) {
        self.temp_files.insert(path);
    }

    /// 清理所有临时文件
    ///
    /// 返回清理的文件数量
    pub fn cleanup_temp_files(&mut self) -> usize {
        let mut count = 0;
        let paths: Vec<PathBuf> = self.temp_files.drain().collect();
        for path in paths {
            if path.exists() {
                if std::fs::remove_file(&path).is_ok() {
                    count += 1;
                }
            }
        }
        count
    }

    /// 计算文件的 SHA256 哈希
    pub fn compute_file_hash(path: &Path) -> Result<String, String> {
        if !path.exists() {
            return Err(format!("文件不存在: {}", path.display()));
        }
        use sha2::Digest;
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        let hash = sha2::Sha256::digest(&data);
        Ok(format!("{:x}", hash))
    }

    /// 确保输出目录存在
    pub fn ensure_dir(path: &Path) -> PathBuf {
        std::fs::create_dir_all(path).ok();
        path.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_filename_with_timestamp() {
        let svc = FileManagementService::new("_test_output");
        let name = svc.generate_filename("audio", ".wav", true);
        assert!(name.starts_with("audio_"));
        assert!(name.ends_with(".wav"));
        // 时间戳是数字
        let ts_part = &name["audio_".len()..name.len() - ".wav".len()];
        assert!(ts_part.parse::<u64>().is_ok(), "时间戳不是数字: {}", ts_part);
    }

    #[test]
    fn test_generate_filename_without_timestamp() {
        let svc = FileManagementService::new("_test_output");
        let name = svc.generate_filename("audio", ".wav", false);
        assert_eq!(name, "audio.wav");
    }

    #[test]
    fn test_get_output_path_with_subdir() {
        let svc = FileManagementService::new("_test_root");
        let path = svc.get_output_path("test", ".txt", Some("sub"));
        assert!(path.starts_with("_test_root\\sub\\") || path.starts_with("_test_root/sub/"));
        assert!(path.to_string_lossy().contains("test_"));
        assert!(path.to_string_lossy().ends_with(".txt"));
        // 清理
        let _ = std::fs::remove_dir_all("_test_root");
    }

    #[test]
    fn test_temp_file_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let mut svc = FileManagementService::new(dir.path());

        let tmp1 = dir.path().join("tmp1.txt");
        let tmp2 = dir.path().join("tmp2.txt");
        std::fs::write(&tmp1, "data1").unwrap();
        std::fs::write(&tmp2, "data2").unwrap();

        svc.register_temp_file(tmp1.clone());
        svc.register_temp_file(tmp2.clone());
        assert!(tmp1.exists());
        assert!(tmp2.exists());

        let cleaned = svc.cleanup_temp_files();
        assert_eq!(cleaned, 2);
        assert!(!tmp1.exists());
        assert!(!tmp2.exists());
    }

    #[test]
    fn test_temp_file_cleanup_empty() {
        let dir = tempfile::tempdir().unwrap();
        let mut svc = FileManagementService::new(dir.path());
        let cleaned = svc.cleanup_temp_files();
        assert_eq!(cleaned, 0);
    }

    #[test]
    fn test_file_hash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        std::fs::write(&path, b"hello world").unwrap();
        let hash = FileManagementService::compute_file_hash(&path).unwrap();
        assert_eq!(hash, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
    }

    #[test]
    fn test_file_hash_not_found() {
        let result = FileManagementService::compute_file_hash(Path::new("_nonexistent_file_xyz"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("不存在"));
    }

    #[test]
    fn test_ensure_dir() {
        let dir = Path::new("_test_ensure_dir_xyz");
        assert!(!dir.exists());
        let result = FileManagementService::ensure_dir(dir);
        assert!(dir.exists());
        assert_eq!(result, dir.to_path_buf());
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    fn test_output_dir() {
        let svc = FileManagementService::new("_test_od");
        assert_eq!(svc.output_dir(), Path::new("_test_od"));
        let _ = std::fs::remove_dir("_test_od");
    }
}
