use anyhow::Result;
use votex_app::platform::tts as voice_lib;

/// 处理 voice 子命令（克隆音色库）
pub fn handle(action: &crate::commands::root::VoiceAction) -> Result<()> {
    use crate::commands::root::VoiceAction;
    match action {
        VoiceAction::Add { name, reference, denoise } => {
            println!("添加克隆音色: {}（参考音频 {:?}，降噪: {}）", name, reference, denoise);
            let meta = voice_lib::add_voice(name, std::path::Path::new(reference), *denoise)?;
            println!("  已入库: {}/{}", voice_lib::voice_library_dir().display(), meta.ref_file);
            println!("  时长: {}ms, 采样率: {}Hz, 已降噪: {}", meta.duration_ms, meta.sample_rate, meta.denoised);
            println!("  使用方式: tts --engine indextts25 --voice {}", name);
        }
        VoiceAction::List => {
            let voices = voice_lib::list_voices();
            if voices.is_empty() {
                println!("音色库为空（使用 voice add 添加克隆音色）");
                return Ok(());
            }
            println!("克隆音色 ({}):", voices.len());
            for v in &voices {
                println!(
                    "  {}  时长 {}ms  采样率 {}Hz  降噪: {}  创建于 {}",
                    v.name, v.duration_ms, v.sample_rate, v.denoised, v.created_at
                );
            }
        }
        VoiceAction::Remove { name, confirm } => {
            let removed = voice_lib::remove_voice(name, *confirm)?;
            if removed {
                println!("已删除音色: {}", name);
            } else {
                println!("音色不存在: {}", name);
            }
        }
    }
    Ok(())
}
