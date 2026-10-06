use anyhow::Result;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use votex_domain::config::value_object::DownloadConfig;

use crate::download::mirror_resolver::DownloadFile;
use crate::download::verifier::Verifier;

/// 下载进度回调
pub type ProgressFn = Box<dyn Fn(u64, u64, &str, &str) + Send>;

/// 下载临时文件后缀
///
/// 下载过程一律先写 `<目标>.part`，校验（大小 / SHA256）通过后才 rename 为最终文件。
/// 旧实现直接写最终路径，一旦中断就会留下与完整文件无法区分的残片。
const PART_SUFFIX: &str = ".part";

/// 连接超时（秒）——防止服务器不响应时下载线程永久挂起
///
/// 这是防挂起的关键：TCP 连接阶段无响应时，`read_timeout` 尚未生效，
/// 只能靠 `connect_timeout` 中断。
const CONNECT_TIMEOUT_SECS: u64 = 30;

/// 单次下载总超时的兜底值（秒）
///
/// 仅在 `DownloadConfig.timeout` 为 0（配置缺失或显式置 0）时使用。
/// 需兼顾 GB 级模型文件（耗时可能数十分钟）与"卡死服务器"保护。
const FALLBACK_DOWNLOAD_TIMEOUT_SECS: u64 = 3600;

/// 多源下载器
///
/// # 为什么是实例而不是静态方法
///
/// 旧实现是 `pub struct Downloader;` + 全静态方法，每次下载都新建一个
/// `reqwest::blocking::Client`。这有两个问题：
///
/// 1. `application.yml` 的 `models.download.timeout` 明明配了值，
///    却是**零消费者** —— 配置写了不生效，属于 AGENTS.md 明令禁止的
///    「配置项形同虚设」；
/// 2. 每次新建 Client 会重建连接池，批量下载时无法复用 TCP 连接。
///
/// 改为持有 `Client` 的实例后，配置有了消费方，且连接得以复用。
pub struct Downloader {
    client: reqwest::blocking::Client,
    /// 是否启用断点续传（对应 `models.download.resume`）
    resume: bool,
    /// 单次下载总超时（秒）
    timeout_secs: u64,
}

impl Downloader {
    /// 按默认配置构造下载器
    pub fn new() -> Result<Self> {
        Self::with_config(&DownloadConfig::default())
    }

    /// 按配置文件构造下载器
    ///
    /// # 参数
    /// - `config`: `models.download` 段配置
    ///
    /// # 行为
    /// - `timeout == 0` 时回落到 [`FALLBACK_DOWNLOAD_TIMEOUT_SECS`]
    /// - 强制 HTTPS：不接受 HTTP 明文下载（模型文件动辄数百 MB，
    ///   明文传输可被中间人替换，等于绕过 SHA256 校验）
    pub fn with_config(config: &DownloadConfig) -> Result<Self> {
        let timeout_secs = if config.timeout == 0 {
            tracing::warn!(
                "models.download.timeout 配置为 0，回退到 {} 秒兜底值",
                FALLBACK_DOWNLOAD_TIMEOUT_SECS
            );
            FALLBACK_DOWNLOAD_TIMEOUT_SECS
        } else {
            config.timeout
        };

        let client = reqwest::blocking::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) votex/1.0")
            .https_only(true)
            .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECS))
            .timeout(Duration::from_secs(timeout_secs))
            .build()?;

        tracing::debug!(
            "下载器已初始化: 超时={}s, 断点续传={}",
            timeout_secs,
            config.resume
        );

        Ok(Self {
            client,
            resume: config.resume,
            timeout_secs,
        })
    }

    /// 当前生效的单次下载超时（秒）
    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }

    /// 按优先级尝试多个 URL 下载单个文件
    pub fn download(
        &self,
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

        // 先写临时文件，全部成功后才落最终路径，避免残片污染
        let part = Self::part_path(dest);
        let mut last_error = None;
        for url in urls {
            let source_name = Self::extract_source_name(url);
            match self.download_single(url, &part, on_progress.as_deref(), &source_name, "") {
                Ok(()) => {
                    // 收尾失败必须上报，不能静默
                    if let Err(e) = std::fs::rename(&part, dest) {
                        last_error = Some(anyhow::anyhow!("下载完成但落盘失败: {}", e));
                        break;
                    }
                    tracing::info!("下载成功: {}", url);
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!("下载失败 {}: {}", url, e);
                    last_error = Some(e);
                }
            }
        }

        // 失败时清理临时文件，避免残留占用磁盘
        Self::remove_part(&part);

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("所有下载源均失败")))
    }

    /// 计算下载临时文件路径（`<目标路径>.part`）
    fn part_path(dest: &Path) -> PathBuf {
        let mut os = dest.as_os_str().to_owned();
        os.push(PART_SUFFIX);
        PathBuf::from(os)
    }

    /// 删除临时下载文件（失败仅告警，不阻断主流程）
    fn remove_part(part: &Path) {
        if part.exists() {
            if let Err(e) = std::fs::remove_file(part) {
                tracing::warn!("清理下载临时文件失败: {:?} - {}", part, e);
            }
        }
        // 同时清理续传元数据
        let etag = Self::etag_path(part);
        if etag.exists() {
            let _ = std::fs::remove_file(&etag);
        }
    }

    /// 续传元数据文件路径：记录上次下载的 ETag，用于校验服务端内容是否已变化
    fn etag_path(part: &Path) -> PathBuf {
        let mut os = part.as_os_str().to_owned();
        os.push(".etag");
        PathBuf::from(os)
    }

    /// 解压产物的临时文件路径（`<目标>.part` → `<目标>.part.extract`）
    fn extracted_path(part: &Path) -> PathBuf {
        let mut os = part.as_os_str().to_owned();
        os.push(".extract");
        PathBuf::from(os)
    }

    /// 下载多个文件
    pub fn download_files(&self,
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

            // 获取预期的 SHA256（从同一目标文件的任意条目中取）
            let expected_sha256: Option<&str> = files
                .iter()
                .filter(|f| f.dest == file.dest)
                .find_map(|f| f.expected_sha256.as_deref());

            // 文件已存在时先校验完整性，通过才跳过；否则删除重新下载
            if dest.exists() {
                let local_size = dest.metadata().ok().map(|m| m.len()).unwrap_or(0);
                let expected_size = file.expected_size;

                // 有 SHA256 时以哈希为准（大小可伪造，哈希才是完整性依据）
                let sha_ok = match expected_sha256 {
                    Some(expected) => match Verifier::verify_sha256(&dest, expected) {
                        Ok(true) => true,
                        Ok(false) => {
                            tracing::warn!("已存在文件 SHA256 校验失败，重新下载: {:?}", dest);
                            false
                        }
                        Err(e) => {
                            // 校验过程出错同样视为不可信，不允许当作"已完成"跳过
                            tracing::warn!("已存在文件 SHA256 校验出错，重新下载: {:?} - {}", dest, e);
                            false
                        }
                    },
                    None => true,
                };

                let size_ok = match expected_size {
                    Some(expected) => local_size == expected,
                    None => true,
                };

                if sha_ok && size_ok {
                    if expected_sha256.is_none() && expected_size.is_none() {
                        // 既无哈希也无大小：无法确认完整性，明确告警而不是静默跳过
                        tracing::warn!(
                            "文件已存在但缺少 SHA256 与大小校验信息，无法确认完整性，直接沿用: {:?}",
                            dest
                        );
                    } else {
                        tracing::debug!("文件已存在且校验通过，跳过: {:?}", dest);
                    }
                    downloaded_files.push(dest);
                    processed_dests.insert(file.dest.clone());
                    continue;
                }

                tracing::info!("文件不完整，删除重新下载: {:?} (本地={}, 预期={:?})", dest, local_size, expected_size);
                if let Err(e) = std::fs::remove_file(&dest) {
                    tracing::warn!("删除不完整文件失败: {:?} - {}", dest, e);
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

            // 下载统一写入临时文件，校验通过后才 rename 落最终路径
            let part = Self::part_path(&dest);
            // 压缩包场景：下载产物不是最终文件，解压提取到 extracted 后再校验它
            let extracted = Self::extracted_path(&part);

            for f in same_dest_files {
                let src_name = Self::extract_source_name(&f.url);
                match self.download_single(&f.url, &part, on_progress.as_deref(), &src_name, file_name) {
                    Ok(()) => {
                        // 压缩包：先提取成员文件，校验对象切换为解压产物
                        let work = match &f.archive {
                            Some(spec) => {
                                tracing::info!("解压提取成员 {} -> {:?}", spec.member, extracted);
                                match crate::download::archive::extract_member(&part, spec, &extracted)
                                {
                                    Ok(()) => {
                                        let _ = std::fs::remove_file(&part);
                                        extracted.as_path()
                                    }
                                    Err(e) => {
                                        tracing::warn!("解压提取失败 {}: {}", file_name, e);
                                        Self::remove_part(&part);
                                        let _ = std::fs::remove_file(&extracted);
                                        last_error =
                                            Some(anyhow::anyhow!("解压失败: {} - {}", file_name, e));
                                        continue;
                                    }
                                }
                            }
                            None => part.as_path(),
                        };

                        // 下载后大小校验
                        if let Some(expected) = f.expected_size {
                            let actual = std::fs::metadata(work)
                                .map(|m| m.len())
                                .unwrap_or(0);
                            if actual != expected {
                                tracing::warn!("下载后大小不匹配，删除文件: {:?} (实际={}, 预期={})", work, actual, expected);
                                Self::remove_part(&part);
                                let _ = std::fs::remove_file(&extracted);
                                last_error = Some(anyhow::anyhow!("大小校验不匹配: {}", file_name));
                                continue;
                            }
                            tracing::debug!("大小校验通过: {:?}", work);
                        }

                        // SHA256 校验
                        if let Some(expected) = expected_sha256 {
                            match Verifier::verify_sha256(work, expected) {
                                Ok(true) => {
                                    tracing::info!("SHA256 校验通过: {:?}", work);
                                }
                                Ok(false) => {
                                    tracing::warn!("SHA256 校验失败，删除文件: {:?}", work);
                                    Self::remove_part(&part);
                                    let _ = std::fs::remove_file(&extracted);
                                    last_error = Some(anyhow::anyhow!("SHA256 校验不匹配: {}", file_name));
                                    continue;
                                }
                                Err(e) => {
                                    // 修复要点：校验过程出错必须视为失败。
                                    // 旧实现此处只 warn 后 fall through 到"下载成功"分支，
                                    // 等于校验可被任意 IO 错误绕过
                                    tracing::warn!("SHA256 校验出错，视为失败并删除文件: {:?} - {}", work, e);
                                    Self::remove_part(&part);
                                    let _ = std::fs::remove_file(&extracted);
                                    last_error = Some(anyhow::anyhow!("SHA256 校验出错: {} - {}", file_name, e));
                                    continue;
                                }
                            }
                        }

                        // 校验全部通过后才落最终路径
                        if let Err(e) = std::fs::rename(work, &dest) {
                            last_error = Some(anyhow::anyhow!("下载完成但落盘失败: {}", e));
                            continue;
                        }
                        Self::cleanup_etag(&part);
                        let _ = std::fs::remove_file(&extracted);

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
                // 清理本次遗留的临时文件，避免残片在下次运行时被误当作可续传内容
                Self::remove_part(&part);
                let _ = std::fs::remove_file(&extracted);
                return Err(last_error.unwrap_or_else(|| anyhow::anyhow!("所有下载源均失败: {}", file_name)));
            }
        }

        Ok(downloaded_files)
    }

    /// 单源下载，支持断点续传和进度回调
    ///
    /// `dest` 传入的是临时文件路径（`<目标>.part`）；写入与校验都发生在临时文件上。
    /// 断点续传会携带上次记录的 ETag（`If-Range`），服务端内容已变化时自动从头重下，
    /// 避免把新内容的尾部续接到旧内容之后产生静默损坏的文件。
    fn download_single(
        &self,
        url: &str,
        dest: &Path,
        on_progress: Option<&(dyn Fn(u64, u64, &str, &str) + Send)>,
        source_name: &str,
        file_name: &str,
    ) -> Result<()> {
        // 断点续传开关：models.download.resume 为 false 时忽略已有残片，从头下载。
        // 旧实现硬编码续传，该配置项同样是零消费者。
        let mut existing_len: u64 = 0;
        if self.resume && dest.exists() {
            existing_len = dest.metadata()?.len();
        } else if dest.exists() && !self.resume {
            tracing::debug!("断点续传已禁用，忽略已有残片: {:?}", dest);
        }
        // 上次下载的 ETag，用于校验续传时服务端内容是否仍一致
        let saved_etag = Self::read_etag(dest);

        // 复用实例持有的 Client：连接池可跨文件共享，且超时来自配置
        let client = &self.client;

        let mut request = client.get(url);
        if existing_len > 0 {
            request = request.header("Range", format!("bytes={}-", existing_len));
            // 仅在本地记录了 ETag 时校验；服务端不支持 If-Range 会返回 200 全量，
            // 下面会据此把 existing_len 归零重新下载
            if let Some(ref etag) = saved_etag {
                request = request.header("If-Range", etag.as_str());
            }
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
            // 服务端未返回 206（不支持续传，或 If-Range 校验失败内容已变），从头下载
            if existing_len > 0 {
                tracing::warn!(
                    "服务端未接受续传请求（状态码 {}），从头重新下载: {}",
                    status,
                    file_name
                );
            }
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
            // 从头下载时必须截断，否则会叠加在上次的残片之后
            std::fs::File::create(dest)?
        };

        // 记录本次的 ETag，供下次续传校验
        let current_etag = response
            .headers()
            .get(reqwest::header::ETAG)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

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

        // 写入 ETag 供下次续传校验（失败不影响本次下载结果）
        if let Some(etag) = current_etag {
            Self::write_etag(dest, &etag);
        }

        Ok(())
    }

    /// 读取上次下载的 ETag
    fn read_etag(part: &Path) -> Option<String> {
        std::fs::read_to_string(Self::etag_path(part))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// 写入本次下载的 ETag
    fn write_etag(part: &Path, etag: &str) {
        if let Err(e) = std::fs::write(Self::etag_path(part), etag) {
            tracing::debug!("写入下载 ETag 失败（不影响本次下载）: {}", e);
        }
    }

    /// 下载完成并落盘后清理 ETag 元数据
    fn cleanup_etag(part: &Path) {
        let etag = Self::etag_path(part);
        if etag.exists() {
            let _ = std::fs::remove_file(&etag);
        }
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

    /// 构造测试用下载器（默认配置）
    fn test_downloader() -> Downloader {
        Downloader::new().expect("构造下载器失败")
    }

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
        let d = test_downloader();
        let result = d.download(&[], Path::new("test.bin"), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("下载源列表为空"));
    }

    #[test]
    fn 超时配置应被注入() {
        // 这是本次修复的核心：models.download.timeout 曾是零消费者
        let cfg = DownloadConfig {
            timeout: 42,
            ..DownloadConfig::default()
        };
        let d = Downloader::with_config(&cfg).expect("构造下载器失败");
        assert_eq!(d.timeout_secs(), 42, "配置的超时未生效");
    }

    #[test]
    fn 超时为零应回退兜底值() {
        let cfg = DownloadConfig {
            timeout: 0,
            ..DownloadConfig::default()
        };
        let d = Downloader::with_config(&cfg).expect("构造下载器失败");
        assert_eq!(
            d.timeout_secs(),
            FALLBACK_DOWNLOAD_TIMEOUT_SECS,
            "timeout=0 应回退兜底值而非禁用超时"
        );
    }

    #[test]
    fn 默认配置应可用() {
        let d = test_downloader();
        assert!(d.timeout_secs() > 0, "默认超时不应为 0");
        assert!(d.resume, "默认应启用断点续传");
    }

    #[test]
    fn 临时文件后缀正确() {
        let part = Downloader::part_path(Path::new("/tmp/model.onnx"));
        assert!(part.to_string_lossy().ends_with(".part"));
    }

    #[test]
    fn 清理临时文件应幂等() {
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        let part = dir.path().join("a.bin.part");
        // 两次调用都不应报错（幂等）
        Downloader::remove_part(&part);
        Downloader::remove_part(&part);
        assert!(!part.exists());
    }

    #[test]
    fn 清理etag文件应幂等() {
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        let part = dir.path().join("a.bin.part");
        std::fs::write(&part, b"partial").expect("写临时文件失败");
        Downloader::cleanup_etag(&part);
        assert!(!Downloader::etag_path(&part).exists(), "etag 文件应被清理");
    }

    #[test]
    fn etag往返读写() {
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        let part = dir.path().join("b.bin.part");
        assert!(Downloader::read_etag(&part).is_none(), "无 etag 文件时应返回 None");
        std::fs::write(Downloader::etag_path(&part), b"\"abc123\"").expect("写 etag 失败");
        assert_eq!(Downloader::read_etag(&part).as_deref(), Some("\"abc123\""));
    }
}
