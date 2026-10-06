//! ONNX Runtime 压缩包解压的端到端验证
//!
//! 用真实的官方 tgz 包验证：下载产物 → 提取成员 → 校验 SHA256 全链路。
//! 依赖 `tmp/ort_pkg/` 下已存在的官方包；包不存在时跳过（不误报失败）。

use std::path::{Path, PathBuf};
use votex_domain::model::registry::ArchiveSpec;
use votex_infra::download::archive::extract_member;
use votex_infra::download::verifier::Verifier;

fn workspace_root() -> PathBuf {
    let mut dir = std::env::current_dir().expect("获取当前目录失败");
    for _ in 0..6 {
        if dir.join("models").join("registry").is_dir() {
            return dir;
        }
        if !dir.pop() {
            break;
        }
    }
    panic!("未找到工作区根目录");
}

#[test]
fn tgz_提取libonnxruntime并校验哈希() {
    let root = workspace_root();
    let pkg = root.join("tmp").join("ort_pkg").join("ort-linux-x64.tgz");
    if !pkg.exists() {
        eprintln!("跳过：未找到官方包 {:?}", pkg);
        return;
    }

    let out_dir = tempfile::tempdir().expect("创建临时目录失败");
    let dest = out_dir.path().join("libonnxruntime.so");

    // 与 models/registry/onnxruntime.yaml 中 Linux 条目保持一致
    let spec = ArchiveSpec {
        archive_type: "targz".into(),
        member: "lib/libonnxruntime.so.1.30.0".into(),
    };
    extract_member(&pkg, &spec, &dest).expect("提取 libonnxruntime.so 失败");

    let size = std::fs::metadata(&dest).expect("产物不存在").len();
    assert!(
        size > 10_000_000,
        "提取出的库过小（{} 字节），可能取到了符号链接占位文件",
        size
    );

    // 与清单登记的 SHA256 比对
    let expected = "245a6f8c38127551057a1cd1ffd59f0a186a227ade4f3492dea2494eb565542e";
    assert_eq!(
        Verifier::verify_sha256(&dest, expected).expect("校验执行失败"),
        true,
        "提取产物的 SHA256 与清单登记值不一致"
    );

    // 确认不是 ELF 之外的东西
    let head = std::fs::read(&dest).expect("读取失败");
    assert_eq!(&head[..4], b"\x7fELF", "应为 ELF 共享库");
}

#[test]
fn tgz_按同名匹配可回退到实体文件() {
    let root = workspace_root();
    let pkg = root.join("tmp").join("ort_pkg").join("ort-linux-x64.tgz");
    if !pkg.exists() {
        eprintln!("跳过：未找到官方包 {:?}", pkg);
        return;
    }

    // member 写符号链接名时，应回退到 .1.30.0 实体文件
    let out_dir = tempfile::tempdir().expect("创建临时目录失败");
    let dest = out_dir.path().join("fallback.so");
    let spec = ArchiveSpec {
        archive_type: "targz".into(),
        member: "lib/libonnxruntime.so".into(),
    };
    extract_member(&pkg, &spec, &dest).expect("回退提取失败");

    let size = std::fs::metadata(&dest).expect("产物不存在").len();
    assert!(
        size > 10_000_000,
        "回退提取到 {} 字节，疑似符号链接占位",
        size
    );
}

#[test]
fn 拒绝穿越路径的成员() {
    let root = workspace_root();
    let pkg = root.join("tmp").join("ort_pkg").join("ort-linux-x64.tgz");
    if !pkg.exists() {
        eprintln!("跳过：未找到官方包 {:?}", pkg);
        return;
    }
    let out_dir = tempfile::tempdir().expect("创建临时目录失败");
    let dest = out_dir.path().join("evil.so");
    let spec = ArchiveSpec {
        archive_type: "targz".into(),
        member: "../../evil.so".into(),
    };
    let err = extract_member(&pkg, &spec, &dest).unwrap_err();
    assert!(
        err.to_string().contains("非法片段"),
        "应拒绝路径穿越，实际错误: {}",
        err
    );
    assert!(!dest.exists(), "拒绝后不得创建任何文件");
    assert!(!Path::new("/tmp/evil.so").exists());
}
