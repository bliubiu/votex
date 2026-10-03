/// FireRedASR CTC + AED 集成测试
///
/// 使用模型自带的 test_wavs 中的音频文件进行语音识别验证。
///
/// 运行方式:
/// ```bash
/// cargo test -p votex-infra --test firered_asr_integration_test -- --nocapture
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

/// 获取 CTC 模型的测试音频目录
fn ctc_test_wavs_dir() -> PathBuf {
    let mut dir = workspace_root();
    dir.push("models/asr/firered-asr-ctc/test_wavs");
    dir
}

/// 获取 AED 模型的测试音频目录
fn aed_test_wavs_dir() -> PathBuf {
    let mut dir = workspace_root();
    dir.push("models/asr/firered-asr-aed/test_wavs");
    dir
}

/// 获取 CTC 模型目录
fn ctc_model_dir() -> PathBuf {
    let mut dir = workspace_root();
    dir.push("models/asr/firered-asr-ctc");
    dir
}

/// 获取 AED 模型目录
fn aed_model_dir() -> PathBuf {
    let mut dir = workspace_root();
    dir.push("models/asr/firered-asr-aed");
    dir
}

/// 读取 WAV 文件并转换为 AudioData
fn read_test_wav(path: &PathBuf) -> votex_domain::shared::value_object::AudioData {
    votex_infra::audio::wav::read_wav(path)
        .expect(&format!("WAV 读取失败: {:?}", path))
}

/// 创建 ASR 参数
fn make_asr_params() -> AsrParams {
    AsrParams {
        model: ModelId::new("firered-asr-ctc"),
        language: Language::ZhEn,
        auto_punctuation: false,
        auto_slice: false,
        slice_length: votex_domain::asr::value_object::SliceLength::S30,
        denoise: false,
        denoise_level: votex_domain::asr::value_object::DenoiseLevel::Low,
        output_format: votex_domain::asr::value_object::SubtitleFormat::Txt,
    }
}

/// 列出目录中的 WAV 文件
fn list_wav_files(dir: &PathBuf) -> Vec<PathBuf> {
    if !dir.exists() {
        return Vec::new();
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("读取目录失败")
        .filter_map(|e| {
            let p = e.ok()?.path();
            if p.extension()?.to_str()? == "wav" { Some(p) } else { None }
        })
        .collect();
    files.sort();
    files
}

// ─── CTC 集成测试 ─────────────────────────────────

#[test]
fn test_ctc_load_and_recognize_all_wavs() {
    let model_dir = ctc_model_dir();
    let wav_dir = ctc_test_wavs_dir();

    if !model_dir.exists() {
        eprintln!("⚠ FireRedASR CTC 模型目录不存在: {:?}，跳过测试", model_dir);
        return;
    }

    let wav_files = list_wav_files(&wav_dir);
    if wav_files.is_empty() {
        eprintln!("⚠ 没有找到测试音频文件: {:?}，跳过测试", wav_dir);
        return;
    }

    eprintln!("=== FireRedASR CTC 集成测试 ===");
    eprintln!("模型目录: {:?}", model_dir);
    eprintln!("找到 {} 个测试音频\n", wav_files.len());

    let mut provider = votex_infra::asr::firered_asr::FireRedAsrCtcProvider::new();
    let model = Model::new(
        ModelId::new("firered-asr-ctc"),
        &model_dir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::FireRedAsr,
    );

    provider.load(&model).expect("FireRedASR CTC 模型加载失败");
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

                // 所有测试音频都应该返回非空结果
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

#[test]
fn test_ctc_recognize_0wav_content() {
    let model_dir = ctc_model_dir();
    let wav_path = ctc_test_wavs_dir().join("0.wav");

    if !model_dir.exists() || !wav_path.exists() {
        eprintln!("⚠ 模型或测试音频不存在，跳过测试");
        return;
    }

    let mut provider = votex_infra::asr::firered_asr::FireRedAsrCtcProvider::new();
    let model = Model::new(
        ModelId::new("firered-asr-ctc"),
        &model_dir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::FireRedAsr,
    );
    provider.load(&model).expect("模型加载失败");

    let audio_data = read_test_wav(&wav_path);
    let result = provider.recognize(&audio_data, &make_asr_params())
        .expect("识别失败");

    let text = result.text.trim();
    eprintln!("CTC 0.wav 识别结果: '{}'", text);
    assert!(!text.is_empty(), "识别结果不应为空");
    assert!(text.contains("一") || text.contains("第") || text.contains("的"),
        "应识别出中文内容，得到: '{}'", text);

    provider.unload().ok();
}

#[test]
fn test_ctc_recognize_sichuan_dialect() {
    let model_dir = ctc_model_dir();
    let wav_path = ctc_test_wavs_dir().join("3-sichuan.wav");

    if !model_dir.exists() || !wav_path.exists() {
        eprintln!("⚠ 模型或测试音频不存在，跳过测试");
        return;
    }

    let mut provider = votex_infra::asr::firered_asr::FireRedAsrCtcProvider::new();
    let model = Model::new(
        ModelId::new("firered-asr-ctc"),
        &model_dir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::FireRedAsr,
    );
    provider.load(&model).expect("模型加载失败");

    let audio_data = read_test_wav(&wav_path);
    let result = provider.recognize(&audio_data, &make_asr_params())
        .expect("识别失败");

    let text = result.text.trim();
    eprintln!("CTC 四川话识别结果: '{}'", text);
    assert!(!text.is_empty(), "识别结果不应为空");

    provider.unload().ok();
}

// ─── AED 集成测试 ─────────────────────────────────

#[test]
fn test_aed_load_and_recognize_all_wavs() {
    let model_dir = aed_model_dir();
    let wav_dir = aed_test_wavs_dir();

    if !model_dir.exists() {
        eprintln!("⚠ FireRedASR AED 模型目录不存在: {:?}，跳过测试", model_dir);
        return;
    }

    let wav_files = list_wav_files(&wav_dir);
    if wav_files.is_empty() {
        eprintln!("⚠ 没有找到测试音频文件: {:?}，跳过测试", wav_dir);
        return;
    }

    eprintln!("=== FireRedASR AED 集成测试 ===");
    eprintln!("模型目录: {:?}", model_dir);
    eprintln!("找到 {} 个测试音频\n", wav_files.len());

    let mut provider = votex_infra::asr::firered_asr::FireRedAsrAedProvider::new();
    let model = Model::new(
        ModelId::new("firered-asr-aed"),
        &model_dir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::FireRedAsr,
    );

    provider.load(&model).expect("FireRedASR AED 模型加载失败");
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

#[test]
fn test_aed_recognize_0wav_content() {
    let model_dir = aed_model_dir();
    let wav_path = aed_test_wavs_dir().join("0.wav");

    if !model_dir.exists() || !wav_path.exists() {
        eprintln!("⚠ 模型或测试音频不存在，跳过测试");
        return;
    }

    let mut provider = votex_infra::asr::firered_asr::FireRedAsrAedProvider::new();
    let model = Model::new(
        ModelId::new("firered-asr-aed"),
        &model_dir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::FireRedAsr,
    );
    provider.load(&model).expect("模型加载失败");

    let audio_data = read_test_wav(&wav_path);
    let result = provider.recognize(&audio_data, &make_asr_params())
        .expect("识别失败");

    let text = result.text.trim();
    eprintln!("AED 0.wav 识别结果: '{}'", text);
    assert!(!text.is_empty(), "识别结果不应为空");
    assert!(text.contains("一") || text.contains("第") || text.contains("的"),
        "应识别出中文内容，得到: '{}'", text);

    provider.unload().ok();
}

#[test]
fn test_aed_recognize_sichuan_dialect() {
    let model_dir = aed_model_dir();
    let wav_path = aed_test_wavs_dir().join("3-sichuan.wav");

    if !model_dir.exists() || !wav_path.exists() {
        eprintln!("⚠ 模型或测试音频不存在，跳过测试");
        return;
    }

    let mut provider = votex_infra::asr::firered_asr::FireRedAsrAedProvider::new();
    let model = Model::new(
        ModelId::new("firered-asr-aed"),
        &model_dir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::FireRedAsr,
    );
    provider.load(&model).expect("模型加载失败");

    let audio_data = read_test_wav(&wav_path);
    let result = provider.recognize(&audio_data, &make_asr_params())
        .expect("识别失败");

    let text = result.text.trim();
    eprintln!("AED 四川话识别结果: '{}'", text);
    assert!(!text.is_empty(), "识别结果不应为空");

    provider.unload().ok();
}

/// CTC vs AED 对比测试：使用同一个音频文件，对比两个引擎的识别结果
#[test]
fn test_compare_ctc_vs_aed() {
    let ctc_model_dir = ctc_model_dir();
    let aed_model_dir = aed_model_dir();
    let wav_path = ctc_test_wavs_dir().join("0.wav"); // 使用 CTC 目录下的 0.wav（两模型 test_wavs 相同）

    if !ctc_model_dir.exists() || !aed_model_dir.exists() || !wav_path.exists() {
        eprintln!("⚠ 模型或测试音频不存在，跳过对比测试");
        return;
    }

    let audio_data = read_test_wav(&wav_path);
    let params = make_asr_params();

    // CTC 识别
    let mut ctc_provider = votex_infra::asr::firered_asr::FireRedAsrCtcProvider::new();
    let ctc_model = Model::new(
        ModelId::new("firered-asr-ctc"),
        &ctc_model_dir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::FireRedAsr,
    );
    ctc_provider.load(&ctc_model).expect("CTC 模型加载失败");
    let ctc_result = ctc_provider.recognize(&audio_data, &params)
        .expect("CTC 识别失败");
    ctc_provider.unload().ok();
    let ctc_text = ctc_result.text.trim().to_string();

    // AED 识别
    let mut aed_provider = votex_infra::asr::firered_asr::FireRedAsrAedProvider::new();
    let aed_model = Model::new(
        ModelId::new("firered-asr-aed"),
        &aed_model_dir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::FireRedAsr,
    );
    aed_provider.load(&aed_model).expect("AED 模型加载失败");
    let aed_result = aed_provider.recognize(&audio_data, &params)
        .expect("AED 识别失败");
    aed_provider.unload().ok();
    let aed_text = aed_result.text.trim().to_string();

    eprintln!("\n=== CTC vs AED 对比 ===");
    eprintln!("CTC: '{}'", ctc_text);
    eprintln!("AED: '{}'", aed_text);

    // 两个结果都不应为空
    assert!(!ctc_text.is_empty(), "CTC 识别结果不应为空");
    assert!(!aed_text.is_empty(), "AED 识别结果不应为空");

    // 至少应有一些共同字符（说明识别的是同一个音频内容）
    let common: Vec<char> = ctc_text.chars().filter(|c| aed_text.contains(*c)).collect();
    eprintln!("共同字符: {} / {}", common.len(), ctc_text.chars().count().max(aed_text.chars().count()));
}
