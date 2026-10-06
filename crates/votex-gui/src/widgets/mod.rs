//! GUI 复用组件：音频播放器、波形显示等。
//!
//! 组件按「是否可测」分两类：
//!
//! - [`player_logic`]：纯计算与状态变更，不依赖 egui / 音频设备，
//!   有完整单元测试。新增播放相关规则应写在这里。
//! - [`audio_player`] 等：渲染与设备 IO，通过委托调用 `player_logic`
//!   复用计算规则，避免同一规则在两处漂移。

pub mod audio_player;
pub mod file_drop;
pub mod player_logic;
pub mod subtitle_editor;
