//! ASR 引擎适配器：SenseVoice / Whisper / FireRed / WeNet / Qwen3-ASR 等。
//!
//! 每个适配器负责自己的分词、特征提取与解码，差异较大，暂不抽象公共基类。

pub mod whisper;
pub mod sensevoice;
pub mod paraformer;
pub mod qwen3_asr;
pub mod firered_asr;
pub mod wenet;
pub mod azure_speech;
pub mod aliyun;
