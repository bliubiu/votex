//! 基础设施门面（Facade）
//!
//! # 职责
//!
//! `votex-app` 是唯一允许直接依赖 `votex-infra` 的层。
//! 本模块把 infra 中「表现层确实需要」的能力收敛成稳定门面，
//! 使 `votex-cli` / `votex-gui`：
//!
//! - ✅ 只依赖 `votex-app` / `votex-domain`
//! - ❌ 不出现 `votex_infra::`、`rusqlite`、`ort` 任何符号
//!
//! # 分工
//!
//! | 子模块 | 门面内容 |
//! | :--- | :--- |
//! | `runtime` | ORT 动态库、日志初始化、执行提供器、推理配置、资源治理 |
//! | `config` | 配置文件读写、敏感字段加密 |
//! | `persistence` | 数据库连接与各 SQLite 仓储构造 |
//! | `registry` | 模型清单加载 |
//! | `paths` | 工作区与模型路径解析 |
//! | `text` | 编码探测与文档正文提取 |
//! | `tts` | 音色库、设备条件探测 |
//! | `download` | 下载进度回调类型别名 |
//!
//! # 边界原则
//!
//! 门面只做**转发与语义包装**，不含业务规则。
//! 一旦某个能力需要在门面里写 if/match 业务判断，
//! 说明它应该下沉到 `use_case` 或 `services`。

pub mod config;
pub mod download;
pub mod paths;
pub mod persistence;
pub mod registry;
pub mod runtime;
pub mod text;
pub mod tts;
