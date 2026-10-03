use crate::asr::value_object::AsrParams;
use crate::pipeline::value_object::*;
use crate::shared::value_object::PipelineId;
use crate::tts::value_object::TtsParams;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 阶段
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    pub index: usize,
    pub kind: StageKind,
    pub status: StageStatus,
    pub artifact_path: Option<PathBuf>,
}

impl Stage {
    pub fn new(index: usize, kind: StageKind) -> Self {
        Self {
            index,
            kind,
            status: StageStatus::Waiting,
            artifact_path: None,
        }
    }
}

/// 流水线输入
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineInput {
    pub text_file: Option<PathBuf>,
    pub media_file: Option<PathBuf>,
    pub tts_params: Option<TtsParams>,
    pub asr_params: Option<AsrParams>,
}

/// 流水线输出
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineOutput {
    pub audio_path: Option<PathBuf>,
    pub subtitle_path: Option<PathBuf>,
}

/// 流水线（聚合根）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline {
    pub id: PipelineId,
    pub kind: PipelineKind,
    pub stages: Vec<Stage>,
    pub status: PipelineStatus,
    pub input: PipelineInput,
    pub output: Option<PipelineOutput>,
    pub created_at: String,
}

impl Pipeline {
    pub fn new(kind: PipelineKind, input: PipelineInput) -> Self {
        let stages = Self::build_stages(kind);
        Self {
            id: PipelineId::new(),
            kind,
            stages,
            status: PipelineStatus::Idle,
            input,
            output: None,
            created_at: chrono_now_string(),
        }
    }

    /// 根据流水线类型构建阶段列表
    fn build_stages(kind: PipelineKind) -> Vec<Stage> {
        match kind {
            PipelineKind::Audiobook => vec![
                Stage::new(0, StageKind::TextPreprocess),
                Stage::new(1, StageKind::TextSegment),
                Stage::new(2, StageKind::TtsSynthesize),
                Stage::new(3, StageKind::AudioConcat),
                Stage::new(4, StageKind::Export),
            ],
            PipelineKind::Subtitle => vec![
                Stage::new(0, StageKind::AudioExtract),
                Stage::new(1, StageKind::AudioSlice),
                Stage::new(2, StageKind::AsrRecognize),
                Stage::new(3, StageKind::SubtitleAlign),
                Stage::new(4, StageKind::Export),
            ],
            PipelineKind::AudiobookWithSubtitle => vec![
                Stage::new(0, StageKind::TextPreprocess),
                Stage::new(1, StageKind::TextSegment),
                Stage::new(2, StageKind::TtsSynthesize),
                Stage::new(3, StageKind::AudioConcat),
                Stage::new(4, StageKind::AsrRecognize),
                Stage::new(5, StageKind::SubtitleAlign),
                Stage::new(6, StageKind::Export),
            ],
        }
    }
}

fn chrono_now_string() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_有声书阶段构建() {
        let pipeline = Pipeline::new(
            PipelineKind::Audiobook,
            PipelineInput {
                text_file: None,
                media_file: None,
                tts_params: None,
                asr_params: None,
            },
        );
        assert_eq!(pipeline.stages.len(), 5);
        assert_eq!(pipeline.stages[0].kind, StageKind::TextPreprocess);
        assert_eq!(pipeline.stages[4].kind, StageKind::Export);
    }

    #[test]
    fn pipeline_字幕阶段构建() {
        let pipeline = Pipeline::new(
            PipelineKind::Subtitle,
            PipelineInput {
                text_file: None,
                media_file: None,
                tts_params: None,
                asr_params: None,
            },
        );
        assert_eq!(pipeline.stages.len(), 5);
        assert_eq!(pipeline.stages[0].kind, StageKind::AudioExtract);
    }

    #[test]
    fn pipeline_有声书加字幕阶段构建() {
        let pipeline = Pipeline::new(
            PipelineKind::AudiobookWithSubtitle,
            PipelineInput {
                text_file: None,
                media_file: None,
                tts_params: None,
                asr_params: None,
            },
        );
        assert_eq!(pipeline.stages.len(), 7);
    }
}
