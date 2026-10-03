use anyhow::Result;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::download::mirror_resolver::DownloadFile;
use crate::download::verifier::Verifier;

/// 下载进度回调
pub type ProgressFn = Box<dyn Fn(u64, u64, &str, &str) + Send>;

/// 多源下载器
pub struct Downloader;

impl Downloader {
    /// 按优先级尝试多个 URL 下载单个文件
    pub fn download(
        urls: &[String],
        dest: &Path,
        on_progress: Option<ProgressFn>,
    ) -> Result<()> {
        if urls.is_empty() {
            anyhow::bail!("下载源列表为空");
        }

        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut last_error = None;
        for url in urls {
            let source_name = Self::extract_source_name(url);
            match Self::download_single(url, dest, on_progress.as_deref(), &source_name, "") {
                Ok(()) => {
                    tracing::info!("下载成功: {}", url);
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!("下载失败 {}: {}", url, e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("所有下载源均失败")))
    }

    /// 下载多个文件
    pub fn download_files(
        files: &[DownloadFile],
        base_dir: &Path,
        on_progress: Option<ProgressFn>,
    ) -> Result<Vec<PathBuf>> {
        let mut downloaded_files = Vec::new();
        let mut processed_dests = std::collections::HashSet::new();

        for file in files {
            let dest = base_dir.join(&file.dest);

            // 跳过已处理的目标路径（同一文件多个源只下载一次）
            if processed_dests.contains(&file.dest) {
                continue;
            }

            // 文件已存在时校验大小，匹配则跳过，不匹配则删除重新下载
            if dest.exists() {
                let local_size = dest.metadata().ok().map(|m| m.len()).unwrap_or(0);
                let expected_size = file.expected_size;
                match expected_size {
                    Some(expected) if local_size == expected => {
                        tracing::debug!("文件已存在（大小匹配），跳过: {:?}", dest);
                        downloaded_files.push(dest);
                        processed_dests.insert(file.dest.clone());
                        continue;
                    }
                    Some(expected) => {
                        tracing::info!("文件不完整，删除重新下载: {:?} (本地={}, 预期={})", dest, local_size, expected);
                        let _ = std::fs::remove_file(&dest);
                    }
                    None => {
                        tracing::debug!("文件已存在（无大小校验跳过）: {:?}", dest);
                        downloaded_files.push(dest);
                        processed_dests.insert(file.dest.clone());
                        continue;
                    }
                }
            }

            let file_name = file.dest.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            let mut last_error = None;
            let mut success = false;

            // 尝试同一文件的多个源
            let same_dest_files: Vec<&DownloadFile> = files
                .iter()
                .filter(|f| f.dest == file.dest)
                .collect();

            // 获取预期的 SHA256（从同一目标文件的任意条目中取）
            let expected_sha256 = same_dest_files.iter()
                .find_map(|f| f.expected_sha256.as_deref());

            for f in same_dest_files {
                let src_name = Self::extract_source_name(&f.url);
                match Self::download_single(&f.url, &dest, on_progress.as_deref(), &src_name, file_name) {
                    Ok(()) => {
                        // 下载后大小校验
                        if let Some(expected) = f.expected_size {
                            let actual = std::fs::metadata(&dest)
                                .map(|m| m.len())
                                .unwrap_or(0);
                            if actual != expected {
                                tracing::warn!("下载后大小不匹配，删除文件: {:?} (实际={}, 预期={})", dest, actual, expected);
                                let _ = std::fs::remove_file(&dest);
                                last_error = Some(anyhow::anyhow!("大小校验不匹配: {}", file_name));
                                continue;
                            }
                            tracing::debug!("大小校验通过: {:?}", dest);
                        }

                        // SHA256 校验
                        if let Some(expected) = expected_sha256 {
                            match Verifier::verify_sha256(&dest, expected) {
                                Ok(true) => {
                                    tracing::info!("SHA256 校验通过: {:?}", dest);
                                }
                                Ok(false) => {
                                    tracing::warn!("SHA256 校验失败，删除文件: {:?}", dest);
                                    let _ = std::fs::remove_file(&dest);
                                    last_error = Some(anyhow::anyhow!("SHA256 校验不匹配: {}", file_name));
                                    continue;
                                }
                                Err(e) => {
                                    tracing::warn!("SHA256 校验出错: {}", e);
                                }
                            }
                        }
                        tracing::info!("下载成功: {}", f.url);
                        downloaded_files.push(dest.clone());
                        success = true;
                        break;
                    }
                    Err(e) => {
                        tracing::warn!("下载失败 {}: {}", f.url, e);
                        last_error = Some(e);
                    }
                }
            }

            processed_dests.insert(file.dest.clone());

            if !success {
                return Err(last_error.unwrap_or_else(|| anyhow::anyhow!("所有下载源均失败: {}", file_name)));
            }
        }

        Ok(downloaded_files)
    }

    /// 单源下载，支持断点续传和进度回调
    fn download_single(
        url: &str,
        dest: &Path,
        on_progress: Option<&(dyn Fn(u64, u64, &str, &str) + Send)>,
        source_name: &str,
        file_name: &str,
    ) -> Result<()> {
        let mut existing_len: u64 = 0;
        if dest.exists() {
            existing_len = dest.metadata()?.len();
        }

        let client = reqwest::blocking::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) votex/1.0")
            .https_only(true)
            .build()?;
        let mut request = client.get(url);
        if existing_len > 0 {
            request = request.header("Range", format!("bytes={}-", existing_len));
        }

        let mut response = request.send()?;
        let status = response.status();

        let (start_byte, total_size) = if status == reqwest::StatusCode::PARTIAL_CONTENT {
            let content_range = response
                .headers()
                .get("content-range")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            let total = content_range
                .split('/')
                .last()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);
            (existing_len, total)
        } else if status.is_success() {
            existing_len = 0;
            let total = response.content_length().unwrap_or(0);
            (0, total)
        } else {
            anyhow::bail!("HTTP 状态码: {}", status);
        };

        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = if existing_len > 0 {
            std::fs::OpenOptions::new().append(true).open(dest)?
        } else {
            std::fs::File::create(dest)?
        };

        let mut downloaded = start_byte;
        let mut buffer = [0u8; 65536];
        let mut last_reported = 0u64;

        loop {
            let n = response.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            file.write_all(&buffer[..n])?;
            downloaded += n as u64;

            if let Some(ref cb) = on_progress {
                if downloaded - last_reported >= 1_048_576 || n < 65536 {
                    cb(downloaded, total_size, source_name, file_name);
                    last_reported = downloaded;
                }
            }
        }

        if let Some(ref cb) = on_progress {
            cb(downloaded, total_size, source_name, file_name);
        }

        Ok(())
    }

    /// 从 URL 提取简短的源名称
    fn extract_source_name(url: &str) -> String {
        if url.contains("hf-mirror.com") {
            "hf-mirror".to_string()
        } else if url.contains("modelscope.cn") {
            "modelscope".to_string()
        } else if url.contains("huggingface.co") {
            "huggingface".to_string()
        } else if url.contains("github.com") {
            "github".to_string()
        } else {
            "unknown".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_source_name_识别镜像源() {
        assert_eq!(
            Downloader::extract_source_name("https://hf-mirror.com/test/model.zip"),
            "hf-mirror"
        );
        assert_eq!(
            Downloader::extract_source_name("https://huggingface.co/test/model.zip"),
            "huggingface"
        );
        assert_eq!(
            Downloader::extract_source_name("https://github.com/test/model.zip"),
            "github"
        );
        assert_eq!(
            Downloader::extract_source_name("https://modelscope.cn/test/model.zip"),
            "modelscope"
        );
    }

    #[test]
    fn download_空源列表应报错() {
        let result = Downloader::download(&[], Path::new("test.bin"), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("下载源列表为空"));
    }
}
