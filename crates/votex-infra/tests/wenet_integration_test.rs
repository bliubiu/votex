/// WeNet Conformer CTC 集成测试
///
/// 使用模型自带的 test_wavs 中的音频文件进行语音识别验证。
///
/// 运行方式:
/// ```bash
/// cargo test -p votex-infra --test wenet_integration_test -- --nocapture
/// ```
use std::path::PathBuf;
use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, Language};
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{EngineKind, ModelId, ModelKind};

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

fn model_dir() -> PathBuf {
    let mut dir = workspace_root();
    dir.push("models/asr/wenet");
    dir
}

fn test_wavs_dir() -> PathBuf {
    let mut dir = model_dir();
    dir.push("test_wavs");
    dir
}

fn read_test_wav(path: &PathBuf) -> votex_domain::shared::value_object::AudioData {
    votex_infra::audio::wav::read_wav(path)
        .expect(&format!("WAV 读取失败: {:?}", path))
}

fn make_asr_params() -> AsrParams {
    AsrParams {
        model: ModelId::new("wenet"),
        language: Language::Zh,
        auto_punctuation: false,
        auto_slice: false,
        slice_length: votex_domain::asr::value_object::SliceLength::S30,
        denoise: false,
        denoise_level: votex_domain::asr::value_object::DenoiseLevel::Low,
        output_format: votex_domain::asr::value_object::SubtitleFormat::Txt,
    }
}

#[test]
fn test_wenet_load_and_recognize_all_wavs() {
    let mdir = model_dir();
    let wav_dir = test_wavs_dir();

    if !mdir.exists() {
        eprintln!("⚠ WeNet 模型目录不存在: {:?}，跳过测试", mdir);
        return;
    }
    if !wav_dir.exists() {
        eprintln!("⚠ 测试音频目录不存在: {:?}，跳过测试", wav_dir);
        return;
    }

    let mut wav_files: Vec<PathBuf> = std::fs::read_dir(&wav_dir)
        .expect("读取目录失败")
        .filter_map(|e| {
            let p = e.ok()?.path();
            if p.extension()?.to_str()? == "wav" { Some(p) } else { None }
        })
        .collect();
    wav_files.sort();

    if wav_files.is_empty() {
        eprintln!("⚠ 没有找到测试音频文件，跳过测试");
        return;
    }

    eprintln!("=== WeNet Conformer 集成测试 ===");
    eprintln!("模型目录: {:?}", mdir);
    eprintln!("找到 {} 个测试音频\n", wav_files.len());

    let mut provider = votex_infra::asr::wenet::WeNetProvider::new();
    let model = Model::new(
        ModelId::new("wenet"),
        &mdir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::WeNet,
    );

    provider.load(&model).expect("WeNet 模型加载失败");
    eprintln!("✓ 模型加载成功\n");

    let params = make_asr_params();
    let mut total_chars = 0;
    let mut success_count = 0;

    for wav_path in &wav_files {
        let filename = wav_path.file_name().unwrap().to_string_lossy();
        let audio_data = read_test_wav(wav_path);

        eprintln!("--- {:?} ({} samples, {} Hz) ---",
            filename, audio_data.samples.len(), audio_data.sample_rate);

        match provider.recognize(&audio_data, &params) {
            Ok(result) => {
                let text = result.text.trim();
                eprintln!("  识别结果: {}", if text.is_empty() { "(空)" } else { text });
                total_chars += text.chars().count();
                success_count += 1;
                assert!(!text.is_empty(),
                    "测试音频 {} 识别结果不应为空", filename);
            }
            Err(e) => {
                eprintln!("  ❌ 识别失败: {}", e);
            }
        }
    }

    provider.unload().ok();
    eprintln!("\n=== 测试结果 ===");
    eprintln!("成功: {}/{}", success_count, wav_files.len());
    eprintln!("总识别字数: {}", total_chars);
    assert!(success_count > 0, "至少应成功识别一个音频文件");
    assert!(total_chars > 0, "总识别字数应大于 0");
}

/// 使用 0.wav 验证具体识别内容
#[test]
fn test_wenet_recognize_0wav_content() {
    let mdir = model_dir();
    let wav_path = test_wavs_dir().join("0.wav");
    if !mdir.exists() || !wav_path.exists() {
        eprintln!("⚠ 模型或测试音频不存在，跳过测试");
        return;
    }

    let mut provider = votex_infra::asr::wenet::WeNetProvider::new();
    let model = Model::new(
        ModelId::new("wenet"),
        &mdir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::WeNet,
    );
    provider.load(&model).expect("模型加载失败");

    let audio_data = read_test_wav(&wav_path);
    let result = provider.recognize(&audio_data, &make_asr_params())
        .expect("识别失败");

    let text = result.text.trim();
    eprintln!("WeNet 0.wav 识别结果: '{}'", text);
    assert!(!text.is_empty(), "识别结果不应为空");
    assert!(text.contains("一") || text.contains("第") || text.contains("的") || text.contains("中"),
        "应识别出中文内容，得到: '{}'", text);

    provider.unload().ok();
}

/// 使用 8kHz 采样率音频测试兼容性
#[test]
fn test_wenet_recognize_8k_wav() {
    let mdir = model_dir();
    let wav_path = test_wavs_dir().join("8k.wav");
    if !mdir.exists() || !wav_path.exists() {
        eprintln!("⚠ 模型或测试音频不存在，跳过测试");
        return;
    }

    let mut provider = votex_infra::asr::wenet::WeNetProvider::new();
    let model = Model::new(
        ModelId::new("wenet"),
        &mdir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::WeNet,
    );
    provider.load(&model).expect("模型加载失败");

    let audio_data = read_test_wav(&wav_path);
    eprintln!("8k.wav: {} samples, {} Hz", audio_data.samples.len(), audio_data.sample_rate);

    let result = provider.recognize(&audio_data, &make_asr_params())
        .expect("识别失败");

    let text = result.text.trim();
    eprintln!("WeNet 8k.wav 识别结果: '{}'", text);
    assert!(!text.is_empty(), "8kHz 音频识别结果不应为空");
    // 即使采样率不同，也应识别出有效中文
    assert!(text.chars().any(|c| c as u32 > 0x4E00),
        "应识别出中文字符，得到: '{}'", text);

    provider.unload().ok();
}
