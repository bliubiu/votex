//! 声阅（votex）主入口
//!
//! 只做「装配 + 分发」：CLI 模式执行完命令即退出，
//! `--gui` 模式把已装配好的 [`votex_app::bootstrap::AppContext`] 交给 GUI。
//!
//! 之所以由主入口而非 `votex-cli` 直接调 `votex-gui`，
//! 是为了保持 crate 依赖单向：`votex-cli` 不依赖 `votex-gui`。

fn main() -> anyhow::Result<()> {
    match votex_cli::run()? {
        votex_cli::RunMode::Cli => {
            // CLI 命令已执行完毕
            Ok(())
        }
        votex_cli::RunMode::Gui(ctx) => {
            // 装配已完成（ORT / 配置 / 日志 / 仓储），GUI 直接消费
            votex_gui::run(*ctx)
        }
    }
}
