//! AES-256-GCM 加密模块
//!
//! 提供配置敏感字段的加密/解密功能。
//! 支持 ENC(xxx) 标记识别，避免循环加密。

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use rand::rngs::OsRng;
use rand::Rng;
use std::path::PathBuf;
use std::sync::Mutex;

use votex_domain::error::CryptoError;

const KEY_SIZE: usize = 32; // 256 bits
const NONCE_SIZE: usize = 12; // 96 bits
const KEY_FILENAME: &str = ".key";

pub struct CryptoEngine {
    key_dir: PathBuf,
    cipher: Mutex<Option<Aes256Gcm>>,
}

impl CryptoEngine {
    /// 创建加密引擎
    ///
    /// - `key_dir`：密钥存放目录。`None` 时走统一路径解析
    ///   （环境变量 → 可执行文件目录 → 工作目录向上回溯），
    ///   而非直接用进程当前目录 —— GUI 双击启动时 cwd 可能是任意目录，
    ///   直接用 cwd 会导致每次启动都找不到已有密钥进而重新生成，
    ///   旧密钥加密的配置全部无法解密。
    pub fn new(key_dir: Option<PathBuf>) -> Self {
        Self {
            key_dir: key_dir.unwrap_or_else(crate::shared::WorkspacePaths::workspace_root),
            cipher: Mutex::new(None),
        }
    }

    /// 初始化加密引擎（加载/生成密钥）
    pub fn initialize(&self) -> Result<(), CryptoError> {
        let key = self.load_or_generate_key()?;
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|_| CryptoError::InvalidKeyLength)?;
        *self.cipher.lock().unwrap_or_else(|e| e.into_inner()) = Some(cipher);
        Ok(())
    }

    /// 是否已初始化
    pub fn is_initialized(&self) -> bool {
        self.cipher.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    /// 加密文本为 ENC(base64) 格式
    pub fn encrypt(&self, plaintext: &str) -> Result<String, CryptoError> {
        let cipher = self.cipher.lock().unwrap_or_else(|e| e.into_inner());
        let cipher = cipher.as_ref().ok_or(CryptoError::NotInitialized)?;

        // 空串走完整加密流程，不提前返回裸空串。
        // 原因：调用方普遍用 `is_encrypted()` 判断字段是否已加密，
        // 若 `encrypt("")` 返回 ""，空值会被误判为明文而跳过解密，
        // 造成「加密存了却读不回来」的不一致。
        let nonce_bytes: [u8; NONCE_SIZE] = OsRng.gen();
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, plaintext.as_bytes())
            .map_err(|e| CryptoError::EncryptFailed(e.to_string()))?;

        // 存储格式: nonce(base64) + ciphertext(base64)
        let mut payload = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
        payload.extend_from_slice(&nonce_bytes);
        payload.extend_from_slice(&ciphertext);
        let encoded = BASE64.encode(&payload);

        Ok(format!("ENC({})", encoded))
    }

    /// 解密 ENC(base64) 格式的数据
    pub fn decrypt(&self, enc_data: &str) -> Result<String, CryptoError> {
        let cipher = self.cipher.lock().unwrap_or_else(|e| e.into_inner());
        let cipher = cipher.as_ref().ok_or(CryptoError::NotInitialized)?;

        let payload_b64 = enc_data
            .trim()
            .strip_prefix("ENC(")
            .and_then(|s| s.strip_suffix(')'))
            .ok_or(CryptoError::InvalidFormat)?;

        let decoded = BASE64
            .decode(payload_b64)
            .map_err(|_| CryptoError::InvalidFormat)?;

        if decoded.len() < NONCE_SIZE {
            return Err(CryptoError::InvalidFormat);
        }

        let (nonce_bytes, ciphertext) = decoded.split_at(NONCE_SIZE);
        let nonce = Nonce::from_slice(nonce_bytes);

        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| CryptoError::DecryptFailed(e.to_string()))?;

        Ok(String::from_utf8(plaintext)
            .map_err(|_| CryptoError::DecryptFailed("UTF-8 解码失败".to_string()))?)
    }

    /// 检查值是否已加密
    /// 判断字符串是否为加密格式
    ///
    /// 要求 `ENC(...)` 内有非空载荷：空标记 `ENC()` 不是任何合法密文，
    /// 判为未加密可避免调用方把损坏数据送进 `decrypt`。
    pub fn is_encrypted(value: &str) -> bool {
        let trimmed = value.trim();
        match trimmed.strip_prefix("ENC(").and_then(|s| s.strip_suffix(')')) {
            Some(payload) => !payload.is_empty(),
            None => false,
        }
    }

    /// 从文件加载密钥，文件不存在则生成新密钥
    fn load_or_generate_key(&self) -> Result<Vec<u8>, CryptoError> {
        let key_path = self.key_dir.join(KEY_FILENAME);

        if key_path.exists() {
            let key = std::fs::read(&key_path)
                .map_err(|e| CryptoError::KeyFileNotFound(e.to_string()))?;
            if key.len() == KEY_SIZE {
                // 存量密钥补做权限收紧。
                // 早期版本只对**新生成**的密钥收紧权限，已存在的文件
                // 可能仍是 `-rw-r--r--`（本仓库实测即如此），
                // 同机其他用户可直接读到 AES 密钥。
                // 这里先检测再改：权限已正确时不重复 chmod，
                // 避免每次启动都改文件元数据触发安全软件告警。
                if !is_key_permission_restricted(&key_path) {
                    match restrict_key_permissions(&key_path) {
                        Ok(()) => tracing::info!(
                            "已收紧存量密钥文件权限: {:?}（此前过宽）",
                            key_path
                        ),
                        Err(e) => tracing::warn!(
                            "收紧存量密钥文件权限失败: {:?}（{}），请手工确认仅当前用户可读",
                            key_path,
                            e
                        ),
                    }
                }
                return Ok(key);
            }
            return Err(CryptoError::InvalidKeyLength);
        }

        // 生成新密钥
        let key_bytes: [u8; KEY_SIZE] = OsRng.gen();
        std::fs::write(&key_path, &key_bytes)
            .map_err(|e| CryptoError::EncryptFailed(format!("写入密钥文件失败: {}", e)))?;

        // 收紧权限：密钥文件必须仅当前用户可读写。
        // Unix 下用 mode 0o600；Windows 的 ACL 语义不同，改用「仅当前用户」标记。
        restrict_key_permissions(&key_path)?;

        tracing::info!("已生成新的加密密钥: {:?}（权限已收紧为仅当前用户可读写）", key_path);
        Ok(key_bytes.to_vec())
    }
}

/// 获取当前 Windows 用户名（`DOMAIN\user` 或 `user`）
///
/// 供 `icacls /grant:r` 使用。取不到时返回 `None`，
/// 调用方应跳过 ACL 收紧并告警，而不是退回到不安全的通配 SID。
#[cfg(windows)]
fn current_windows_user() -> Option<String> {
    // 1) USERNAME 环境变量最快，失败再走 whoami
    if let Ok(u) = std::env::var("USERNAME") {
        if !u.trim().is_empty() {
            // 拼接域：USERDOMAIN 存在时用 DOMAIN\user
            return Some(match std::env::var("USERDOMAIN") {
                Ok(d) if !d.trim().is_empty() => format!("{}\\{}", d, u),
                _ => u,
            });
        }
    }

    // 2) 回退到 whoami /user，从输出里解析 "domain\user"
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let out = std::process::Command::new("whoami")
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    // 输出形如 "domain\user"
    let first = text.lines().next()?.trim();
    if first.is_empty() {
        None
    } else {
        Some(first.to_string())
    }
}

/// 检测密钥文件权限是否已限制为「仅当前用户可读写」
///
/// Unix 下检查 mode 的组/其他位是否全为 0。
/// Windows 无法从 `std::fs` 读出 ACL，保守返回 `false`——
/// 宁可多调一次 `icacls`（失败仅告警），也不要漏掉过宽权限。
/// 其他平台同样返回 `false`，交由 [`restrict_key_permissions`] 打 warn。
fn is_key_permission_restricted(path: &std::path::Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        match std::fs::metadata(path) {
            Ok(md) => {
                let mode = md.permissions().mode();
                // 低 9 位为组/其他权限，全 0 才算收紧
                mode & 0o077 == 0
            }
            Err(_) => false,
        }
    }

    #[cfg(not(unix))]
    {
        let _ = path;
        false
    }
}

/// 收紧密钥文件权限为「仅当前用户可读写」
///
/// # 平台差异
///
/// - Unix（含 Linux/macOS）：`chmod 0o600`，去掉组与其他用户的全部权限。
///   **注意**：这是纵深防御而非唯一防线 —— 即使权限被改宽，
///   密钥仍只解开本机配置里的 API Key，不会外传。
/// - Windows：`std::fs` 无 POSIX 权限位，需借助 `icacls`。
///   失败时**不阻断启动**（Windows 上文件 ACL 通常已受用户目录保护），
///   但必须记 warn，让用户知情。
fn restrict_key_permissions(path: &std::path::Path) -> Result<(), CryptoError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).map_err(|e| {
            CryptoError::EncryptFailed(format!("设置密钥文件权限失败 {:?}: {}", path, e))
        })?;
        return Ok(());
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // 隐藏控制台窗口（CREATE_NO_WINDOW = 0x0800_0000）
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;

        // 必须用**当前用户名**而非通配 SID。
        // 原实现传 `*S-1-5-21-*-*-*-*:F`，Windows 的 icacls 不支持 SID 中缀通配，
        // 命令会失败但退出码仍可能为 0（旧实现只 warn 用户手工处理），
        // 结果是 ACL 原封不动 —— 实测本仓库 `.key` 在修复后仍是
        // `Authenticated Users:(M)` + `Users:(RX)`，同机其他用户可读密钥。
        let user = match current_windows_user() {
            Some(u) => u,
            None => {
                tracing::warn!(
                    "无法确定当前 Windows 用户名，跳过 ACL 收紧，请手工确认 {} 仅当前用户可访问",
                    path.display()
                );
                return Ok(());
            }
        };

        let grant = format!("{}:(F)", user);
        let status = std::process::Command::new("icacls")
            .arg(path)
            .args(["/inheritance:r", "/grant:r"])
            .arg(&grant)
            .creation_flags(CREATE_NO_WINDOW)
            .output();
        match status {
            Ok(out) if out.status.success() => {
                // icacls 可能部分失败（stdout 也含错误行），检查是否有失败标记
                let combined = format!(
                    "{}{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
                if combined.contains("失败") {
                    tracing::warn!(
                        "icacls 部分失败: {}，请手工确认 {} 的 ACL",
                        combined.trim(),
                        path.display()
                    );
                } else {
                    tracing::debug!("已收紧 {} 的 ACL 为仅 {} 可访问", path.display(), user);
                }
                Ok(())
            }
            Ok(out) => {
                tracing::warn!(
                    "收紧密钥文件 ACL 失败（Windows）: {}，请手工确认 {} 仅当前用户可访问",
                    format!(
                        "{}{}",
                        String::from_utf8_lossy(&out.stdout).trim(),
                        String::from_utf8_lossy(&out.stderr).trim()
                    ),
                    path.display()
                );
                Ok(())
            }
            Err(e) => {
                tracing::warn!(
                    "调用 icacls 失败（{}），请手工确认 {} 仅当前用户可访问",
                    e,
                    path.display()
                );
                Ok(())
            }
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        tracing::warn!("当前平台不支持设置密钥文件权限，请手工确认权限");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造隔离的加密引擎（独立临时目录，不触碰真实 .key）
    fn engine(dir: &std::path::Path) -> CryptoEngine {
        let e = CryptoEngine::new(Some(dir.to_path_buf()));
        e.initialize().expect("初始化加密引擎失败");
        e
    }

    #[test]
    fn 加密解密往返一致() {
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        let e = engine(dir.path());
        let plain = "sk-proj-abc123def456ghi789jkl";

        let enc = e.encrypt(plain).expect("加密失败");
        assert_ne!(enc, plain, "密文不应等于明文");
        assert!(CryptoEngine::is_encrypted(&enc), "密文应带 ENC 标记");
        assert!(!enc.contains("sk-proj"), "明文片段泄露: {enc}");

        let dec = e.decrypt(&enc).expect("解密失败");
        assert_eq!(dec, plain);
    }

    #[test]
    fn 每次加密密文不同() {
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        let e = engine(dir.path());

        let a = e.encrypt("same-plaintext").expect("加密失败");
        let b = e.encrypt("same-plaintext").expect("加密失败");
        // 每次都用新 nonce（AES-GCM 安全前提：nonce 绝不可复用）
        assert_ne!(a, b, "两次加密结果相同，nonce 可能被复用");
        assert_eq!(e.decrypt(&a).unwrap(), e.decrypt(&b).unwrap());
    }

    #[test]
    fn 密文被篡改应解密失败() {
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        let e = engine(dir.path());
        let enc = e.encrypt("sensitive").expect("加密失败");

        // 改动密文中间一个字符，GCM 认证标签应校验失败
        let mut chars: Vec<char> = enc.chars().collect();
        let mid = chars.len() / 2;
        chars[mid] = if chars[mid] == 'A' { 'B' } else { 'A' };
        let tampered: String = chars.into_iter().collect();

        assert!(
            e.decrypt(&tampered).is_err(),
            "被篡改的密文不应解密成功"
        );
    }

    #[test]
    fn 非加密值应被识别() {
        assert!(!CryptoEngine::is_encrypted("sk-plain-key"));
        assert!(!CryptoEngine::is_encrypted(""));
        assert!(!CryptoEngine::is_encrypted("ENC(未闭合"));
        assert!(CryptoEngine::is_encrypted("ENC(anything)"));
        // 空标记不是合法密文
        assert!(!CryptoEngine::is_encrypted("ENC()"));
        assert!(!CryptoEngine::is_encrypted("  ENC()  "));
    }

    #[test]
    fn 明文传入解密应报错() {
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        let e = engine(dir.path());
        // 未加 ENC 标记的值不是合法密文
        assert!(e.decrypt("plain-text").is_err());
    }

    #[test]
    fn 密钥长度非法应报错() {
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        // 写入长度错误的密钥
        std::fs::write(dir.path().join(KEY_FILENAME), b"too-short").expect("写密钥失败");

        let e = CryptoEngine::new(Some(dir.path().to_path_buf()));
        let err = e.initialize().expect_err("短密钥应被拒绝");
        assert!(
            matches!(err, CryptoError::InvalidKeyLength),
            "错误类型不符: {err:?}"
        );
    }

    #[test]
    fn 密钥文件应被创建且长度正确() {
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        let _e = engine(dir.path());

        let key_path = dir.path().join(KEY_FILENAME);
        assert!(key_path.exists(), "密钥文件应被创建");
        assert_eq!(
            std::fs::metadata(&key_path).unwrap().len() as usize,
            KEY_SIZE,
            "密钥长度应为 {KEY_SIZE}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn 密钥文件权限应为仅用户可读写() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        let _e = engine(dir.path());

        let mode = std::fs::metadata(dir.path().join(KEY_FILENAME))
            .expect("读取元数据失败")
            .permissions()
            .mode();
        // 期望 0o600：组与其他用户无任何权限
        assert_eq!(mode & 0o777, 0o600, "权限过宽: {:o}", mode & 0o777);
    }

    #[cfg(unix)]
    #[test]
    fn 存量密钥权限过宽应被自动收紧() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("创建临时目录失败");

        // 模拟历史版本留下的过宽密钥文件（实测本仓库曾为 -rw-r--r--）
        let key_path = dir.path().join(KEY_FILENAME);
        std::fs::write(&key_path, [7u8; KEY_SIZE]).expect("写入密钥失败");
        std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o644))
            .expect("设置宽松权限失败");

        assert!(
            !is_key_permission_restricted(&key_path),
            "0o644 应被判定为权限过宽"
        );

        // 构造引擎即触发补收紧（不需要重新生成密钥）
        let e = engine(dir.path());
        // 触发一次加解密，确保走过 load_or_generate_key 的存量分支
        let _ = e.encrypt("probe");

        let mode = std::fs::metadata(&key_path)
            .expect("读取元数据失败")
            .permissions()
            .mode();
        assert_eq!(
            mode & 0o777,
            0o600,
            "存量密钥权限应收紧为 0o600，实际 {:o}",
            mode & 0o777
        );
        assert!(is_key_permission_restricted(&key_path));
    }

    #[cfg(unix)]
    #[test]
    fn 权限已正确时不重复改动() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        let key_path = dir.path().join(KEY_FILENAME);
        std::fs::write(&key_path, [7u8; KEY_SIZE]).unwrap();
        std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o600)).unwrap();

        assert!(is_key_permission_restricted(&key_path));

        // 人为放大 mtime，用它判断是否被重复 chmod
        let before = std::fs::metadata(&key_path).unwrap().modified().unwrap();

        let e = engine(dir.path());
        let _ = e.encrypt("probe");

        let after = std::fs::metadata(&key_path).unwrap().modified().unwrap();
        assert_eq!(
            before, after,
            "权限已正确时不应重复写文件元数据（避免触发安全软件告警）"
        );
    }

    #[cfg(unix)]
    #[test]
    fn 收紧权限后仍能解密() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        let key_path = dir.path().join(KEY_FILENAME);
        std::fs::write(&key_path, [7u8; KEY_SIZE]).unwrap();
        std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o666)).unwrap();

        let e = engine(dir.path());
        let enc = e.encrypt("secret").unwrap();
        assert_eq!(e.decrypt(&enc).unwrap(), "secret");
    }

    /// 非 Unix 平台无法读出 POSIX mode，检测函数应保守返回 `false`
    ///（宁可多调一次收紧，也不漏掉过宽权限），
    /// 且整个加载流程不应因此出错。
    #[cfg(not(unix))]
    #[test]
    fn 非unix平台检测保守返回false且流程正常() {
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        let key_path = dir.path().join(KEY_FILENAME);
        std::fs::write(&key_path, [7u8; KEY_SIZE]).unwrap();

        assert!(
            !is_key_permission_restricted(&key_path),
            "非 Unix 平台应保守判定为未收紧，以触发补收紧流程"
        );

        // 走存量分支：读文件 + 尝试收紧（Windows 走 icacls，失败仅告警）
        let e = engine(dir.path());
        let enc = e.encrypt("probe").expect("加密不应因权限流程失败");
        assert_eq!(e.decrypt(&enc).unwrap(), "probe");
    }

    #[test]
    fn 权限检测对不存在的文件不panic() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_key_permission_restricted(&dir.path().join("nope")));
    }

    #[test]
    fn 复用已有密钥应能解密旧密文() {
        let dir = tempfile::tempdir().expect("创建临时目录失败");

        // 第一次运行：生成密钥并加密
        let first = engine(dir.path());
        let enc = first.encrypt("legacy-api-key").expect("加密失败");

        // 第二次运行：同一目录重建引擎（模拟重启）
        drop(first);
        let second = engine(dir.path());
        assert_eq!(
            second.decrypt(&enc).expect("重启后应能解密"),
            "legacy-api-key",
            "重启后无法解密，说明密钥未被复用"
        );
    }

    #[test]
    fn 空字符串可正常加解密() {
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        let e = engine(dir.path());
        let enc = e.encrypt("").expect("空串加密失败");
        assert_eq!(e.decrypt(&enc).expect("空串解密失败"), "");
    }

    #[test]
    fn 中文与特殊字符可正常加解密() {
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        let e = engine(dir.path());
        for plain in ["中文密钥值", "with spaces & symbols!@#", "换行\n制表\t"] {
            let enc = e.encrypt(plain).expect("加密失败");
            assert_eq!(e.decrypt(&enc).expect("解密失败"), plain);
        }
    }
}
