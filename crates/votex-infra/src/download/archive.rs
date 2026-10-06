//! 压缩包成员提取
//!
//! 部分官方发布物只提供压缩包（如 ONNX Runtime 仅提供 `.tgz` / `.zip`），
//! 必须解压并从中提取单个成员文件才能使用。本模块只做「提取一个成员」
//! 这一件事，不提供整包解压，避免把不受信任的压缩包内容大面积落盘。
//!
//! 安全约束：
//! - 只提取清单中显式声明的 `member`，不遍历落盘其他条目；
//! - 成员名若包含 `..` 或为绝对路径则拒绝，防止压缩包路径穿越写出目标目录；
//! - 目标路径始终由调用方给定的 `dest` 决定，不受压缩包内路径影响。

use anyhow::Result;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use votex_domain::model::registry::ArchiveSpec;

/// 校验成员路径安全，拒绝路径穿越与绝对路径
fn ensure_safe_member(member: &str) -> Result<()> {
    if member.is_empty() {
        anyhow::bail!("压缩包成员路径为空");
    }
    // 统一分隔符后检查
    let normalized = member.replace('\\', "/");
    if normalized.starts_with('/') {
        anyhow::bail!("压缩包成员路径不得为绝对路径: {}", member);
    }
    if normalized
        .split('/')
        .any(|seg| seg == ".." || seg == "~" || seg.contains(':'))
    {
        anyhow::bail!("压缩包成员路径含非法片段: {}", member);
    }
    Ok(())
}

/// 从压缩包中提取指定成员到 `dest`
///
/// `archive_type` 支持 `zip`、`targz` 与 `tarbz2`（不区分大小写）。
/// `dest` 的父目录会自动创建。
pub fn extract_member(archive_path: &Path, spec: &ArchiveSpec, dest: &Path) -> Result<()> {
    ensure_safe_member(&spec.member)?;

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let kind = spec.archive_type.to_ascii_lowercase();
    match kind.as_str() {
        "zip" => extract_from_zip(archive_path, &spec.member, dest),
        "targz" | "tar.gz" | "tgz" => extract_from_targz(archive_path, &spec.member, dest),
        "tarbz2" | "tar.bz2" | "tbz2" => extract_from_tarbz2(archive_path, &spec.member, dest),
        other => anyhow::bail!(
            "不支持的压缩包类型: {}（仅支持 zip / targz / tarbz2）",
            other
        ),
    }
}

/// ZIP 成员名匹配：容忍归档内以 `./` 开头或含顶层目录前缀
fn zip_name_matches(entry_name: &str, member: &str) -> bool {
    let a = entry_name.replace('\\', "/");
    let m = member.replace('\\', "/");
    let a = a.trim_start_matches("./");
    let m = m.trim_start_matches("./");
    a == m
}

fn extract_from_zip(archive_path: &Path, member: &str, dest: &Path) -> Result<()> {
    let file = File::open(archive_path)
        .map_err(|e| anyhow::anyhow!("打开 zip 失败 {:?}: {}", archive_path, e))?;
    let mut zip = zip::ZipArchive::new(BufReader::new(file))
        .map_err(|e| anyhow::anyhow!("读取 zip 结构失败: {}", e))?;

    // 优先精确匹配；否则退化为按文件名后缀唯一匹配（不同发布包顶层目录名不一致）
    let idx = (0..zip.len())
        .find(|i| {
            zip.by_index_raw(*i)
                .map(|f| zip_name_matches(f.name(), member))
                .unwrap_or(false)
        })
        .or_else(|| {
            let target = Path::new(member.replace('\\', "/").as_str())
                .file_name()
                .map(|s| s.to_string_lossy().to_string());
            match target {
                Some(t) => {
                    let mut found = None;
                    for i in 0..zip.len() {
                        if let Ok(f) = zip.by_index_raw(i) {
                            let n = f.name().replace('\\', "/");
                            if Path::new(&n).file_name().map(|s| s.to_string_lossy().to_string())
                                == Some(t.clone())
                            {
                                if found.is_some() {
                                    return None; // 同名多个，无法确定取哪个
                                }
                                found = Some(i);
                            }
                        }
                    }
                    found
                }
                None => None,
            }
        });

    let idx = idx.ok_or_else(|| anyhow::anyhow!("zip 中未找到成员: {}", member))?;
    let mut entry = zip
        .by_index(idx)
        .map_err(|e| anyhow::anyhow!("读取 zip 成员失败: {}", e))?;

    let mut out = File::create(dest)?;
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = entry.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
    }
    out.flush()?;
    Ok(())
}

fn extract_from_targz(archive_path: &Path, member: &str, dest: &Path) -> Result<()> {
    let target_name = Path::new(member.replace('\\', "/").as_str())
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .ok_or_else(|| anyhow::anyhow!("tgz 成员路径无效: {}", member))?;

    // 官方包内 lib/libonnxruntime.so 与 lib/libonnxruntime.so.1.30.0 是
    // 「同名符号链接 + 带版本实体文件」两个条目，tar 在部分平台无法还原符号链接
    // （Git Bash 下解出的符号链接是 0 字节文件）。因此匹配顺序为：
    // 精确路径 → 同名 → 以 .<版本> 结尾的实体文件。
    let chosen = scan_targz(archive_path, member, &target_name)?;

    // entries() 是一次性迭代，需重新打开归档后解包
    let file = File::open(archive_path)
        .map_err(|e| anyhow::anyhow!("打开 tgz 失败 {:?}: {}", archive_path, e))?;
    let gz = flate2::read::GzDecoder::new(BufReader::new(file));
    let mut tar = tar::Archive::new(gz);

    let mut extracted = false;
    for entry in tar.entries().map_err(|e| anyhow::anyhow!("读取 tgz 条目失败: {}", e))? {
        let mut entry = entry.map_err(|e| anyhow::anyhow!("读取 tgz 条目失败: {}", e))?;
        let path = entry.path().map_err(|e| anyhow::anyhow!("tgz 条目路径无效: {}", e))?;
        if path == chosen && entry.header().entry_type().is_file() {
            entry
                .unpack(dest)
                .map_err(|e| anyhow::anyhow!("解包 tgz 成员 {:?} 失败: {}", chosen, e))?;
            extracted = true;
            break;
        }
    }

    if !extracted {
        anyhow::bail!("tgz 中未找到可解包的成员: {}", member);
    }
    Ok(())
}

/// 扫描 tgz 定位目标成员，返回其在归档内的完整路径
fn scan_targz(archive_path: &Path, member: &str, target_name: &str) -> Result<std::path::PathBuf> {
    let file = File::open(archive_path)
        .map_err(|e| anyhow::anyhow!("打开 tgz 失败 {:?}: {}", archive_path, e))?;
    let gz = flate2::read::GzDecoder::new(BufReader::new(file));
    let mut tar = tar::Archive::new(gz);

    let want = member.replace('\\', "/");
    let mut exact: Option<std::path::PathBuf> = None;
    let mut by_name: Option<std::path::PathBuf> = None;
    let mut versioned: Option<std::path::PathBuf> = None;

    for entry in tar.entries().map_err(|e| anyhow::anyhow!("读取 tgz 条目失败: {}", e))? {
        let entry = entry.map_err(|e| anyhow::anyhow!("读取 tgz 条目失败: {}", e))?;
        let path = entry.path().map_err(|e| anyhow::anyhow!("tgz 条目路径无效: {}", e))?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let n = path.to_string_lossy().replace('\\', "/");
        if n.trim_start_matches("./") == want {
            exact = Some(path.to_path_buf());
            break;
        }
        if Path::new(&n).file_name().map(|s| s.to_string_lossy().to_string())
            == Some(target_name.to_string())
        {
            by_name = Some(path.to_path_buf());
        } else if by_name.is_none()
            && versioned.is_none()
            && Path::new(&n)
                .file_name()
                .map(|s| {
                    let fname = s.to_string_lossy();
                    fname.starts_with(&format!("{}.", target_name))
                })
                .unwrap_or(false)
        {
            versioned = Some(path.to_path_buf());
        }
    }

    exact
        .or(by_name)
        .or(versioned)
        .ok_or_else(|| anyhow::anyhow!("tgz 中未找到成员: {}", member))
}

/// 从 tar.bz2 归档中提取指定成员到 `dest`
///
/// 典型来源：sherpa-onnx 官方发布包（如 `sherpa-onnx-qwen3-asr-0.6B-int8-*.tar.bz2`）。
/// 匹配顺序：精确路径 → 同名文件（sherpa 包路径固定，无需 tgz 的版本号回退逻辑）。
fn extract_from_tarbz2(archive_path: &Path, member: &str, dest: &Path) -> Result<()> {
    let want = member.replace('\\', "/");
    let target_name = Path::new(&want)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .ok_or_else(|| anyhow::anyhow!("tar.bz2 成员路径无效: {}", member))?;

    let file = File::open(archive_path)
        .map_err(|e| anyhow::anyhow!("打开 tar.bz2 失败 {:?}: {}", archive_path, e))?;
    let bz = bzip2::read::BzDecoder::new(BufReader::new(file));
    let mut tar = tar::Archive::new(bz);

    let mut chosen: Option<std::path::PathBuf> = None;
    for entry in tar
        .entries()
        .map_err(|e| anyhow::anyhow!("读取 tar.bz2 条目失败: {}", e))?
    {
        let entry = entry.map_err(|e| anyhow::anyhow!("读取 tar.bz2 条目失败: {}", e))?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry
            .path()
            .map_err(|e| anyhow::anyhow!("tar.bz2 条目路径无效: {}", e))?;
        let n = path.to_string_lossy().replace('\\', "/");
        if n.trim_start_matches("./") == want {
            chosen = Some(path.to_path_buf());
            break;
        }
        if Path::new(&n)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            == Some(target_name.clone())
        {
            chosen = Some(path.to_path_buf());
        }
    }

    let chosen = chosen.ok_or_else(|| anyhow::anyhow!("tar.bz2 中未找到成员: {}", member))?;

    // entries() 是一次性迭代，需重新打开归档后解包
    let file = File::open(archive_path)
        .map_err(|e| anyhow::anyhow!("重新打开 tar.bz2 失败 {:?}: {}", archive_path, e))?;
    let bz = bzip2::read::BzDecoder::new(BufReader::new(file));
    let mut tar = tar::Archive::new(bz);

    let mut extracted = false;
    for entry in tar
        .entries()
        .map_err(|e| anyhow::anyhow!("读取 tar.bz2 条目失败: {}", e))?
    {
        let mut entry = entry.map_err(|e| anyhow::anyhow!("读取 tar.bz2 条目失败: {}", e))?;
        let path = entry
            .path()
            .map_err(|e| anyhow::anyhow!("tar.bz2 条目路径无效: {}", e))?;
        if path == chosen && entry.header().entry_type().is_file() {
            entry
                .unpack(dest)
                .map_err(|e| anyhow::anyhow!("解包 tar.bz2 成员 {:?} 失败: {}", chosen, e))?;
            extracted = true;
            break;
        }
    }

    if !extracted {
        anyhow::bail!("tar.bz2 中未找到可解包的成员: {}", member);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 拒绝路径穿越() {
        assert!(ensure_safe_member("../evil.so").is_err());
        assert!(ensure_safe_member("lib/../../evil.so").is_err());
        assert!(ensure_safe_member("/etc/passwd").is_err());
        assert!(ensure_safe_member("C:/evil.dll").is_err());
        assert!(ensure_safe_member("").is_err());
    }

    #[test]
    fn 接受正常成员路径() {
        assert!(ensure_safe_member("lib/libonnxruntime.so").is_ok());
        assert!(ensure_safe_member("./lib/onnxruntime.dll").is_ok());
    }

    #[test]
    fn zip_名称匹配容忍前缀() {
        // zip 包内无顶层目录，可精确匹配
        assert!(zip_name_matches("lib/onnxruntime.dll", "lib/onnxruntime.dll"));
        // 少数打包工具会加 ./ 前缀
        assert!(zip_name_matches("./lib/onnxruntime.dll", "lib/onnxruntime.dll"));
        // 顶层目录名不同的情况由 extract_from_zip 的文件名回退处理，
        // 此处仅保证不会误判为匹配
        assert!(!zip_name_matches(
            "onnxruntime-win-x64-1.30.0/lib/onnxruntime.dll",
            "lib/onnxruntime.dll"
        ));
    }

    #[test]
    fn 不支持的类型报错() {
        let spec = ArchiveSpec {
            archive_type: "rar".into(),
            member: "lib/x.so".into(),
        };
        let err = extract_member(Path::new("nonexistent.rar"), &spec, Path::new("out/x.so"))
            .unwrap_err();
        assert!(err.to_string().contains("不支持的压缩包类型"));
    }

    #[test]
    fn tarbz2_成员提取() {
        use bzip2::write::BzEncoder;
        use std::io::Write;

        // 现场构造一个带顶层目录的 tar.bz2（模拟 sherpa-onnx 官方发布包布局）
        let dir = tempfile::tempdir().unwrap();
        let archive_path = dir.path().join("pkg.tar.bz2");
        {
            let bz = BzEncoder::new(
                File::create(&archive_path).unwrap(),
                bzip2::Compression::default(),
            );
            let mut builder = tar::Builder::new(bz);
            // 成员 1：顶层目录下的文件
            let mut h = tar::Header::new_gnu();
            let data = b"conv frontend weights";
            h.set_size(data.len() as u64);
            h.set_mode(0o644);
            h.set_cksum();
            builder
                .append_data(&mut h, "pkg/conv_frontend.onnx", &data[..])
                .unwrap();
            // 成员 2：子目录文件（tokenizer/vocab.json）
            let mut h2 = tar::Header::new_gnu();
            let data2 = b"{\"vocab\": true}";
            h2.set_size(data2.len() as u64);
            h2.set_mode(0o644);
            h2.set_cksum();
            builder
                .append_data(&mut h2, "pkg/tokenizer/vocab.json", &data2[..])
                .unwrap();
            builder.into_inner().unwrap().finish().unwrap();
        }

        // 提取子目录成员到嵌套 dest 路径
        let dest = dir.path().join("out/tokenizer/vocab.json");
        let spec = ArchiveSpec {
            archive_type: "tarbz2".into(),
            member: "pkg/tokenizer/vocab.json".into(),
        };
        extract_member(&archive_path, &spec, &dest).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"{\"vocab\": true}");

        // 精确路径优先于同名回退
        let dest2 = dir.path().join("out/conv_frontend.onnx");
        let spec2 = ArchiveSpec {
            archive_type: "tar.bz2".into(),
            member: "pkg/conv_frontend.onnx".into(),
        };
        extract_member(&archive_path, &spec2, &dest2).unwrap();
        assert_eq!(std::fs::read(&dest2).unwrap(), b"conv frontend weights");
    }
}
