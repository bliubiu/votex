use crate::error::PipelineError;
use crate::pipeline::entity::Pipeline;
use crate::pipeline::value_object::{PipelineStatus, QualityUpgrade, StageStatus};

/// 流水线执行领域服务
pub struct PipelineExecutor;

impl PipelineExecutor {
    /// 暂停流水线
    pub fn pause(pipeline: &mut Pipeline) -> Result<(), PipelineError> {
        match &pipeline.status {
            PipelineStatus::Running(stage) => {
                pipeline.status = PipelineStatus::Paused(*stage);
                Ok(())
            }
            _ => Err(PipelineError::InvalidStatus),
        }
    }

    /// 恢复流水线
    pub fn resume(pipeline: &mut Pipeline) -> Result<(), PipelineError> {
        match &pipeline.status {
            PipelineStatus::Paused(stage) => {
                pipeline.status = PipelineStatus::Running(*stage);
                Ok(())
            }
            _ => Err(PipelineError::InvalidStatus),
        }
    }

    /// 取消流水线
    pub fn cancel(pipeline: &mut Pipeline) -> Result<(), PipelineError> {
        match &pipeline.status {
            PipelineStatus::Running(_) | PipelineStatus::Paused(_) => {
                pipeline.status = PipelineStatus::Cancelled;
                Ok(())
            }
            _ => Err(PipelineError::InvalidStatus),
        }
    }

    /// 重试失败阶段
    pub fn retry_stage(pipeline: &mut Pipeline, stage_index: usize) -> Result<(), PipelineError> {
        if stage_index >= pipeline.stages.len() {
            return Err(PipelineError::StageFailed {
                stage: stage_index,
                reason: "阶段索引越界".to_string(),
            });
        }

        match &pipeline.stages[stage_index].status {
            StageStatus::Failed(_) => {
                pipeline.stages[stage_index].status = StageStatus::Waiting;
                pipeline.status = PipelineStatus::Running(stage_index);
                Ok(())
            }
            _ => Err(PipelineError::InvalidStatus),
        }
    }
}

/// 渐进式质量建议领域服务
pub struct ProgressiveQualityAdvisor;

impl ProgressiveQualityAdvisor {
    /// 根据当前引擎建议升级方案
    pub fn suggest_upgrade(pipeline: &Pipeline) -> Option<QualityUpgrade> {
        // 检查是否使用了轻量模型，建议升级
        let has_tts = pipeline.stages.iter().any(|s| {
            matches!(s.kind, crate::pipeline::value_object::StageKind::TtsSynthesize)
        });
        let has_asr = pipeline.stages.iter().any(|s| {
            matches!(s.kind, crate::pipeline::value_object::StageKind::AsrRecognize)
        });

        if has_tts {
            return Some(QualityUpgrade {
                current_engine: "Kokoro-82M".to_string(),
                suggested_engine: "IndexTTS2".to_string(),
                reason: "IndexTTS2 支持方言，语音更自然".to_string(),
            });
        }

        if has_asr {
            return Some(QualityUpgrade {
                current_engine: "Whisper base".to_string(),
                suggested_engine: "Whisper small".to_string(),
                reason: "Whisper small 识别准确度更高".to_string(),
            });
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::entity::{Pipeline, PipelineInput};
    use crate::pipeline::value_object::PipelineKind;

    #[test]
    fn pipeline_executor_暂停恢复() {
        let mut pipeline = Pipeline::new(
            PipelineKind::Audiobook,
            PipelineInput {
                text_file: None,
                media_file: None,
                tts_params: None,
                asr_params: None,
            },
        );
        pipeline.status = PipelineStatus::Running(0);

        assert!(PipelineExecutor::pause(&mut pipeline).is_ok());
        assert!(matches!(pipeline.status, PipelineStatus::Paused(0)));

        assert!(PipelineExecutor::resume(&mut pipeline).is_ok());
        assert!(matches!(pipeline.status, PipelineStatus::Running(0)));
    }

    #[test]
    fn pipeline_executor_取消() {
        let mut pipeline = Pipeline::new(
            PipelineKind::Audiobook,
            PipelineInput {
                text_file: None,
                media_file: None,
                tts_params: None,
                asr_params: None,
            },
        );
        pipeline.status = PipelineStatus::Running(0);
        assert!(PipelineExecutor::cancel(&mut pipeline).is_ok());
        assert!(matches!(pipeline.status, PipelineStatus::Cancelled));
    }

    #[test]
    fn pipeline_executor_重试失败阶段() {
        let mut pipeline = Pipeline::new(
            PipelineKind::Audiobook,
            PipelineInput {
                text_file: None,
                media_file: None,
                tts_params: None,
                asr_params: None,
            },
        );
        pipeline.stages[0].status = StageStatus::Failed("测试失败".to_string());
        pipeline.status = PipelineStatus::Failed(0, "测试失败".to_string());

        assert!(PipelineExecutor::retry_stage(&mut pipeline, 0).is_ok());
        assert!(matches!(pipeline.stages[0].status, StageStatus::Waiting));
    }
}
