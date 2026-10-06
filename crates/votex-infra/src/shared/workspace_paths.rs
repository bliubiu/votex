//! 工作区路径统一解析
//!
//! # 为什么要集中在这里
//!
//! 项目里模型文件、清单、ONNX Runtime 动态库都放在「工作区根目录」下，
//! 但**工作区根目录并不等于进程当前目录**：
//!
//! | 场景 | 当前目录 | 直接 `current_dir()` 的后果 |
//! | :--- | :--- | :--- |
//! | `cargo test` | `crates/votex-infra` | 找不到 `models/`，测试误报模型缺失 |
//! | GUI 双击启动 | 用户最近访问的目录 | 找不到模型，用户以为模型丢了 |
//! | 从 `D:\` 启动 | `D:\` | 同上 |
//! | 单二进制分发 | 安装目录 | 用户数据目录与程序目录分离时失效 |
//!
//! 历史上有 9 处各自实现「向上回溯找 models/」的副本，且
//! `tokenizer/byte_level.rs`、`tts/kokoro_g2p/mapping.rs` 用
//! `env!("CARGO_MANIFEST_DIR")` 固化编译期路径 —— 编译机路径写进了
//! 发布二进制，换机器即失效。
//!
//! # 解析顺序
//!
//! 1. 环境变量 `VOTEX_MODELS_DIR` / `VOTEX_WORKSPACE_DIR`（显式覆盖，最高优先级）
//! 2. `application.yml` 中 `models.storage_path`（若为绝对路径）
//! 3. 可执行文件所在目录（GUI 双击、msi 安装场景）
//! 4. 当前目录及其各级父目录（`cargo run` / `cargo test` / 源码内运行）
//! 5. 兜底：当前目录
//!
//! 该模块是**唯一**允许调用 `current_dir()` 解析资源位置的地方，
//! 新增代码请勿再自行拼接 `models/`。

use std::path::{Path, PathBuf};

/// 资源根目录解析器
pub struct WorkspacePaths;
impl WorkspacePaths {
    /// 查找工作区根目录（含 `models/` 与 `application.yml` 的那一层）
    ///
    /// # 返回
    /// 第一个满足「含 `models/` 目录或 `application.yml` 文件」的候选路径；
    /// 全部候选都不满足时返回兜底值（环境变量 → 可执行文件目录 → 当前目录）。
    pub fn workspace_root() -> PathBuf {
        for candidate in Self::root_candidates() {
            if Self::is_workspace_root(&candidate) {
                return candidate;
            }
        }
        // 无候选命中：优先用显式配置，其次可执行文件目录，最后当前目录
        Self::explicit_root()
            .or_else(Self::exe_dir)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }

    /// 获取模型根目录（`models/`）
    pub fn models_dir() -> PathBuf {
        // storage_path 可以是绝对路径（用户自建模型库），优先尊重
        if let Some(custom) = Self::configured_models_dir() {
            return custom;
        }
        Self::workspace_root().join("models")
    }

    /// 获取模型清单目录（`models/registry/`）
    pub fn registry_dir() -> PathBuf {
        Self::models_dir().join("registry")
    }

    /// 获取 ONNX Runtime 动态库目录（`models/runtime/`）
    pub fn runtime_dir() -> PathBuf {
        Self::models_dir().join("runtime")
    }

    /// 解析工作区根目录的全部候选，按优先级排列
    fn root_candidates() -> Vec<PathBuf> {
        let mut candidates: Vec<PathBuf> = Vec::with_capacity(4);

        if let Some(explicit) = Self::explicit_root() {
            candidates.push(explicit);
        }
        if let Some(exe) = Self::exe_dir() {
            Self::push_ancestors(&mut candidates, exe);
        }
        if let Ok(cwd) = std::env::current_dir() {
            Self::push_ancestors(&mut candidates, cwd);
        }

        candidates.dedup();
        candidates
    }

    /// 显式指定的根目录（环境变量）
    fn explicit_root() -> Option<PathBuf> {
        for key in ["VOTEX_WORKSPACE_DIR", "VOTEX_MODELS_DIR"] {
            if let Some(dir) = std::env::var_os(key) {
                let path = PathBuf::from(dir);
                // VOTEX_MODELS_DIR 指向的是 models 本身，向上取一层作为根
                if key == "VOTEX_MODELS_DIR" {
                    return path.parent().map(Path::to_path_buf);
                }
                if path.is_dir() {
                    return Some(path);
                }
            }
        }
        None
    }

    /// 配置文件中指定的模型目录（仅接受绝对路径）
    ///
    /// 相对路径会随进程工作目录漂移，等于没配置，因此直接忽略。
    fn configured_models_dir() -> Option<PathBuf> {
        let config = Self::workspace_root().join("application.yml");
        let content = std::fs::read_to_string(config).ok()?;
        for line in content.lines() {
            let line = line.trim();
            let Some(rest) = line.strip_prefix("storage_path:") else {
                continue;
            };
            let value = rest.trim().trim_matches(['"', '\'']);
            if value.is_empty() {
                continue;
            }
            let path = PathBuf::from(value);
            if path.is_absolute() {
                return Some(path);
            }
        }
        None
    }

    /// 可执行文件所在目录
    fn exe_dir() -> Option<PathBuf> {
        std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(Path::to_path_buf))
    }

    /// 把 `dir` 及其最多 `depth` 级父目录按顺序追加进候选列表
    fn push_ancestors(out: &mut Vec<PathBuf>, dir: PathBuf) {
        let mut cur = dir;
        for _ in 0..6 {
            out.push(cur.clone());
            if !cur.pop() {
                break;
            }
        }
    }

    /// 判定某路径是否像工作区根目录
    fn is_workspace_root(path: &Path) -> bool {
        path.join("models").is_dir() || path.join("application.yml").is_file()
    }
}

/// 测试辅助：临时改写工作区环境变量
///
/// 单元测试默认多线程并行，共享进程环境变量会互相污染
/// （`cargo test` 早期版本里这类污染曾导致随机失败）。
/// 因此这里用全局互斥锁把「改环境变量 + 执行断言」整段串行化，
/// 并在 `Drop` 时恢复原值。
#[cfg(test)]
pub mod tests_support {
    use std::path::Path;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    /// 进程级互斥锁：确保任一时刻只有一个用例在改环境变量
    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// 环境变量守卫
    pub struct EnvGuard {
        _lock: MutexGuard<'static, ()>,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        /// 将 `VOTEX_WORKSPACE_DIR` 指向指定目录，并持有全局锁
        pub fn set_workspace(dir: &Path) -> Self {
            // 锁中毒时恢复使用，避免一次 panic 让后续所有用例失败
            let lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
            let previous = std::env::var_os("VOTEX_WORKSPACE_DIR");
            // SAFETY: 已持有进程级互斥锁，同一时刻无其他线程读写该变量
            unsafe { std::env::set_var("VOTEX_WORKSPACE_DIR", dir) };
            Self {
                _lock: lock,
                previous,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: 仍持有 `self._lock`（drop 顺序：先恢复值再释放锁）
            match self.previous.take() {
                Some(v) => unsafe { std::env::set_var("VOTEX_WORKSPACE_DIR", v) },
                None => unsafe { std::env::remove_var("VOTEX_WORKSPACE_DIR") },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一个临时工作区：`root/models/{tts,asr,registry,runtime}` + `application.yml`
    fn make_workspace() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("创建临时目录失败");
        for sub in ["models/tts", "models/asr", "models/registry", "models/runtime"] {
            std::fs::create_dir_all(dir.path().join(sub)).expect("创建目录失败");
        }
        std::fs::write(
            dir.path().join("application.yml"),
            "models:\n  storage_path: models\n",
        )
        .expect("写配置失败");
        dir
    }

    #[test]
    fn 环境变量指向的工作区应被识别() {
        let ws = make_workspace();
        let _guard = super::tests_support::EnvGuard::set_workspace(ws.path());

        let root = WorkspacePaths::workspace_root();
        let models = WorkspacePaths::models_dir();


        assert_eq!(root, ws.path());
        assert_eq!(models, ws.path().join("models"));
    }

    #[test]
    fn 从子目录启动应向上找到工作区根() {
        let ws = make_workspace();
        let nested = ws.path().join("crates").join("votex-infra");
        std::fs::create_dir_all(&nested).expect("创建子目录失败");

        let _guard = super::tests_support::EnvGuard::set_workspace(ws.path());

        // 模拟从 crates 子目录启动：候选列表应包含根目录
        let mut candidates = Vec::new();
        WorkspacePaths::push_ancestors(&mut candidates, nested);


        assert!(
            candidates.contains(&ws.path().to_path_buf()),
            "向上回溯未覆盖工作区根: {candidates:?}"
        );
        assert!(candidates.len() >= 3, "回溯层数不足: {}", candidates.len());
    }

    #[test]
    fn 相对storage_path应被忽略() {
        let ws = make_workspace();
        // 覆盖为相对路径（默认值）
        std::fs::write(
            ws.path().join("application.yml"),
            "models:\n  storage_path: models\n",
        )
        .expect("写配置失败");

        let _guard = super::tests_support::EnvGuard::set_workspace(ws.path());
        let configured = WorkspacePaths::configured_models_dir();

        assert!(
            configured.is_none(),
            "相对 storage_path 不应被当作自定义模型目录: {configured:?}"
        );
    }

    #[test]
    fn 绝对storage_path应被采纳() {
        let ws = make_workspace();
        let custom = ws.path().join("my-models");
        std::fs::create_dir_all(&custom).expect("创建自定义模型目录失败");
        std::fs::write(
            ws.path().join("application.yml"),
            format!("models:\n  storage_path: {}\n", custom.display()),
        )
        .expect("写配置失败");

        let _guard = super::tests_support::EnvGuard::set_workspace(ws.path());
        let configured = WorkspacePaths::configured_models_dir();

        assert_eq!(configured, Some(custom));
    }

    #[test]
    fn 子目录常量应正确拼接() {
        let ws = make_workspace();
        let _guard = super::tests_support::EnvGuard::set_workspace(ws.path());

        let registry = WorkspacePaths::registry_dir();
        let runtime = WorkspacePaths::runtime_dir();


        assert!(registry.ends_with("models/registry") || registry.ends_with("models\\registry"));
        assert!(runtime.ends_with("models/runtime") || runtime.ends_with("models\\runtime"));
    }

    #[test]
    fn 任意位置都应返回可用路径() {
        // 兜底行为：即使什么都没配置，也不能 panic
        let root = WorkspacePaths::workspace_root();
        assert!(!root.as_os_str().is_empty(), "兜底路径不应为空");
    }
}
