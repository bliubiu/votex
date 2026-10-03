//! LLM 适配器模块
//!
//! 提供 LLM 提供者的工厂函数和 DeepSeek API 适配器。

pub mod deepseek;

use votex_domain::llm::provider::LlmProvider;

/// 根据引擎名称创建 LLM 提供者
pub fn create_llm_provider(engine: &str) -> Result<Box<dyn LlmProvider>, Box<dyn std::error::Error>> {
    match engine {
        "deepseek-chat" | "deepseek-reasoner" | "deepseek" => {
            Ok(Box::new(deepseek::DeepSeekProvider::new(engine)?))
        }
        _ => Err(format!("不支持的 LLM 引擎: {}", engine).into()),
    }
}
