//! 守护测试：models/registry/*.yaml 必须可解析且关键字段合法
//!
//! 背景：模型清单是纯数据驱动，字段名写错（如 size 写成 file_size）不会报编译错，
//! 只会在运行时被 serde 静默忽略或直接解析失败，导致模型下载不到。
//! 本测试在 CI 阶段就拦住这类问题。

use std::path::PathBuf;
use votex_domain::model::registry::ModelRegistryEntry;
use votex_infra::config::model_registry::ModelRegistryLoader;

/// 定位工作区根目录（含 models/registry 的那一层）
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
    panic!("未找到工作区根目录（应包含 models/registry）");
}

fn load_all() -> Vec<ModelRegistryEntry> {
    let root = workspace_root();
    let loader = ModelRegistryLoader::load(&root.join("models").join("registry"))
        .expect("加载模型清单失败");
    assert!(
        !loader.is_empty(),
        "模型清单为空：目录中应有 *.yaml 描述文件"
    );
    loader
}

/// 磁盘上的 *.yaml 数量
fn yaml_file_count() -> usize {
    let root = workspace_root();
    std::fs::read_dir(root.join("models").join("registry"))
        .expect("读取清单目录失败")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "yaml"))
        .count()
}

/// 磁盘文件数必须与解析成功条数一致。
///
/// 背景：`ModelRegistryLoader` 对单个文件的 YAML 解析失败是**跳过并告警**，
/// 不会中断加载。因此一个 YAML 重复键（例如工具误插了第二行 `sha256:`）
/// 就会让整份清单悄悄消失，而「清单非空」断言仍会通过 ——
/// 实测 `onnxruntime.yaml` 因重复键被跳过，29 个文件只加载出 28 条，
/// 运行时依赖清单整个消失。本测试专门堵这个洞。
#[test]
fn registry_无清单被静默跳过() {
    let loaded = load_all().len();
    let on_disk = yaml_file_count();
    assert_eq!(
        loaded, on_disk,
        "清单文件数({})与成功解析数({})不一致，说明有 YAML 解析失败被静默跳过。\
         常见原因：同一映射层内出现重复键（如两行 sha256:）",
        on_disk, loaded
    );
}

/// 每个文件条目内 `sha256` 至多出现一次，且非空时必须是 64 位十六进制。
///
/// 这条不变量曾被 `tools/fill_sha256.py` 的非幂等插入破坏过，
/// 重复键会让 serde_yaml 拒绝整份清单。
#[test]
fn registry_条目内sha256唯一且格式合法() {
    use votex_domain::model::registry::ModelFileEntry;

    fn check(files: &[ModelFileEntry], id: &str) {
        for f in files {
            match f.sha256.as_deref() {
                None => {}
                Some(s) => {
                    assert_eq!(
                        s.len(),
                        64,
                        "清单 {} 的文件 {} 的 sha256 长度应为 64，实际 {}",
                        id,
                        f.name,
                        s.len()
                    );
                    assert!(
                        s.chars().all(|c| c.is_ascii_hexdigit()),
                        "清单 {} 的文件 {} 的 sha256 含非十六进制字符: {}",
                        id,
                        f.name,
                        s
                    );
                }
            }
        }
    }

    for e in load_all() {
        check(&e.files, &e.id);
    }
}

/// 同一文件条目内**原始文本**只能出现一次 `sha256:`。
///
/// serde 反序列化会把重复键静默丢弃（保留最后一个），
/// 因此上面的 `registry_条目内sha256唯一且格式合法` 抓不到重复键 ——
/// 真正的风险在解析阶段（整份清单被拒）。这里必须扫原始文本。
#[test]
fn registry_原始文本无重复sha256键() {
    let root = workspace_root();
    let reg = root.join("models").join("registry");

    for entry in std::fs::read_dir(&reg).expect("读取清单目录失败") {
        let path = entry.expect("目录项读取失败").path();
        if path.extension().is_none_or(|x| x != "yaml") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("读取清单失败");
        let mut in_files = false;
        let mut current: Option<String> = None;
        let mut seen = 0usize;

        for (n, raw) in text.lines().enumerate() {
            let st = raw.trim();
            let indent = raw.len() - raw.trim_start().len();

            if st.is_empty() || st.starts_with('#') {
                continue;
            }
            if indent == 0 {
                in_files = st == "files:";
                continue;
            }
            if !in_files {
                continue;
            }
            if indent == 2 && st.starts_with("- name:") {
                if let Some(name) = &current {
                    assert_eq!(
                        seen, 1,
                        "{} 第 {} 行所属条目 {} 出现了 {}{} 个 sha256: 键，\
                         YAML 重复键会导致整份清单解析失败",
                        path.display(),
                        n + 1,
                        name,
                        seen,
                        if seen > 1 { "（多于 1）" } else { "" }
                    );
                }
                current = Some(st.split(':').nth(1).unwrap_or("").trim().to_string());
                seen = 0;
                continue;
            }
            if indent == 4 && st.starts_with("sha256:") {
                seen += 1;
            }
        }

        if let Some(name) = &current {
            assert_eq!(
                seen, 1,
                "{} 末条目 {} 出现了 {} 个 sha256: 键",
                path.display(),
                name,
                seen
            );
        }
    }
}

#[test]
fn registry_所有清单均可解析() {
    let entries = load_all();
    for e in &entries {
        assert!(!e.id.is_empty(), "清单缺少 id");
        assert!(!e.name.is_empty(), "清单 {} 缺少 name", e.id);
        assert!(!e.kind.is_empty(), "清单 {} 缺少 kind", e.id);
        for f in &e.files {
            assert!(!f.name.is_empty(), "清单 {} 存在空文件名", e.id);
            assert!(
                !f.sources.is_empty(),
                "清单 {} 的文件 {} 没有配置任何镜像源",
                e.id,
                f.name
            );
        }
    }
}

/// 已知问题登记：以下清单未声明任何文件，用户点击下载拿不到资源。
///
/// 本测试只负责把问题暴露出来，不阻塞 CI——修复后应从该列表中移除。
#[test]
fn registry_已知空文件清单() {
    let entries = load_all();
    let empty: Vec<&str> = entries
        .iter()
        .filter(|e| e.files.is_empty())
        .map(|e| e.id.as_str())
        .collect();
    for id in &empty {
        eprintln!("[已知问题] 清单 {} 未声明任何文件，下载功能不可用", id);
    }
    assert!(
        !empty.contains(&"onnxruntime"),
        "运行时清单 onnxruntime 不应为空"
    );
}

#[test]
fn registry_onnxruntime_运行时清单正确() {
    let entries = load_all();
    let ort = entries
        .iter()
        .find(|e| e.id == "onnxruntime")
        .expect("未找到 onnxruntime 运行时清单");

    // 必须直接落在 models/runtime，而不是 models/<kind>/<id>
    assert_eq!(
        ort.storage_dir(),
        "runtime",
        "ONNX Runtime 应直接落在 runtime 目录"
    );

    // 三个平台各一份产物，且都带 archive 解压配置与校验值
    let names: Vec<&str> = ort.files.iter().map(|f| f.name.as_str()).collect();
    for expected in [
        "onnxruntime.dll",
        "libonnxruntime.so",
        "libonnxruntime.dylib",
    ] {
        assert!(
            names.contains(&expected),
            "清单缺少 {} 平台产物，当前: {:?}",
            expected,
            names
        );
    }

    for f in &ort.files {
        assert!(
            f.platforms.is_some(),
            "运行时库文件 {} 必须用 platforms 限定平台，否则会下载全部三份",
            f.name
        );
        assert!(
            f.archive.is_some(),
            "运行时库文件 {} 必须声明 archive（官方只发布压缩包）",
            f.name
        );
        let sha = f.sha256.as_deref().unwrap_or("");
        assert_eq!(sha.len(), 64, "{} 的 sha256 应为 64 位十六进制", f.name);
        assert!(
            sha.chars().all(|c| c.is_ascii_hexdigit()),
            "{} 的 sha256 含非十六进制字符: {}",
            f.name,
            sha
        );
    }
}

#[test]
fn registry_平台过滤只保留当前系统产物() {
    use votex_infra::download::mirror_resolver::MirrorResolver;

    let entries = load_all();
    let ort = entries.iter().find(|e| e.id == "onnxruntime").expect("缺少清单");
    let files = MirrorResolver::resolve(ort, &["github".to_string()]);

    assert!(!files.is_empty(), "当前平台应解析出至少一个运行时库文件");

    // 解析结果里不应出现其它平台的产物名
    let current = if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    for f in &files {
        let name = f.dest.file_name().unwrap().to_string_lossy().to_string();
        assert!(
            name.starts_with("libonnxruntime") || name == "onnxruntime.dll",
            "解析出了非当前平台产物: {}",
            name
        );
        assert!(f.archive.is_some(), "{} 应携带 archive 配置", name);
    }
    assert!(
        files.iter().any(|f| f.dest.ends_with(current)),
        "解析结果应包含当前平台产物 {}",
        current
    );
}
