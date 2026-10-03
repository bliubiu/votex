use std::sync::Arc;

use crate::pipeline::entity::Stage;
use crate::pipeline::value_object::StageKind;
use crate::shared::value_object::ProgressEvent;

/// 流水线阶段处理器 trait
///
/// 每个阶段类型实现此 trait，使流水线执行器无需硬编码 match 分支。
/// 新增阶段类型时只需新增一个实现并注册到 StageRegistry 即可。
pub trait StageHandler: Send + Sync {
    /// 返回此处理器负责的阶段类型
    fn kind(&self) -> StageKind;

    /// 执行阶段逻辑
    ///
    /// - `stage`: 当前阶段实例（含序号、状态）
    /// - `params`: 运行参数（文本等上下文）
    /// - `progress`: 进度回调，用于实时报告进度
    fn execute(
        &self,
        stage: &Stage,
        params: StageParams,
        progress: &dyn Fn(ProgressEvent),
    ) -> Result<StageParams, Box<dyn std::error::Error + Send>>;
}

/// 阶段输入/输出参数 —— 在阶段之间传递
#[derive(Debug, Clone, Default)]
pub struct StageParams {
    /// 原始文本
    pub raw_text: String,
    /// 预处理后的文本
    pub processed_text: String,
    /// 分段后的文本段
    pub segments: Vec<String>,
    /// 基础文件名
    pub base_name: String,
}

/// 阶段注册表 trait
///
/// 将 StageKind 映射到对应的 StageHandler 实现。
pub trait StageRegistry: Send + Sync {
    /// 注册一个阶段处理器
    fn register(&mut self, handler: Arc<dyn StageHandler>);

    /// 获取指定阶段类型对应的处理器
    fn get(&self, kind: StageKind) -> Option<Arc<dyn StageHandler>>;

    /// 列出所有已注册的阶段类型
    fn registered_kinds(&self) -> Vec<StageKind>;
}
