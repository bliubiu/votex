fn main() -> anyhow::Result<()> {
    match votex_cli::run()? {
        votex_cli::RunMode::Cli => {
            // CLI 命令已执行完毕
            Ok(())
        }
        votex_cli::RunMode::Gui { models_dir, pipeline_repo, download_repo } => {
            // 由主入口启动 GUI（CLI crate 不依赖 GUI crate）
            votex_gui::run(models_dir, pipeline_repo, download_repo)
        }
    }
}
