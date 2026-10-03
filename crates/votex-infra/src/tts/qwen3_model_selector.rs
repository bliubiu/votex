//! Qwen3-TTS 模型变体智能检测与推荐选择器
//!
//! 根据设备条件（GPU 可用性、内存大小、模型文件存在性）
//! 自动推荐合适的模型变体（0.6B / 1.7B int4），
//! 同时支持用户手动覆盖选择。
//!
//! 优先级：手动选择 > 智能推荐 > 默认值（1.7B）

use std::path::Path;

use crate::shared::{ExecutionProvider, OrtSessionFactory};

/// 设备检测结果
#[derive(Debug, Clone)]
pub struct DeviceConditions {
    /// DirectML GPU 是否可用
    pub has_gpu: bool,
    /// 可用物理内存 (MB)
    pub available_ram_mb: u64,
    /// 0.6B 模型目录是否存在
    pub model_06_exists: bool,
    /// 1.7B 模型目录是否存在
    pub model_17_exists: bool,
}

/// 模型推荐结果
#[derive(Debug, Clone)]
pub struct ModelRecommendation {
    /// 推荐的模型 ID（如 "qwen3-tts-0.6b" / "qwen3-tts-1.7b"）
    pub recommended_id: String,
    /// 推荐理由（展示给用户）
    pub reason: String,
    /// 推荐变体标签（展示用）
    pub label: String,
    /// 设备条件（供 UI 展示）
    pub conditions: DeviceConditions,
}

/// 检测设备条件
///
/// 检查三项指标：
/// - GPU 可用性（通过全局 EP 配置 + feature flag）
/// - 可用物理内存（Windows GlobalMemoryStatusEx）
/// - 模型文件存在性（在 models/tts/ 下查找）
pub fn detect_device(models_base: &Path) -> DeviceConditions {
    // ---- GPU 检测 ----
    let ep = OrtSessionFactory::get_global_ep();
    let has_gpu = match ep {
        ExecutionProvider::DirectML => true,
        ExecutionProvider::Auto => {
            // Auto 模式：DML feature 启用且全局配置允许 DML
            // 不实际加载模型，信任 feature flag + 配置
            cfg!(feature = "directml")
        }
        ExecutionProvider::Cpu => false,
    };

    // ---- 内存检测 ----
    let available_ram_mb = get_available_physical_memory_mb();

    // ---- 模型文件存在性 ----
    let tts_dir = models_base.join("tts");
    let model_06_exists = tts_dir.join("qwen3-tts-0.6b").exists();
    let model_17_exists = tts_dir.join("qwen3-tts-1.7b").exists();

    tracing::info!(
        "Qwen3-TTS 设备检测: GPU={}, RAM={}MB, 0.6B={}, 1.7B={}",
        has_gpu, available_ram_mb, model_06_exists, model_17_exists
    );

    DeviceConditions {
        has_gpu,
        available_ram_mb,
        model_06_exists,
        model_17_exists,
    }
}

/// 智能推荐模型变体
///
/// 规则：
/// 1. GPU 可用 + 内存 >= 8GB + 1.7B 存在 → 推荐 1.7B（最佳音质）
/// 2. CPU 模式 + 0.6B 存在 → 推荐 0.6B（内存友好）
/// 3. 内存 < 8GB + 0.6B 存在 → 推荐 0.6B（防 OOM）
/// 4. 仅 1.7B 存在 → 推荐 1.7B（唯一选项）
/// 5. 都不可用 → 默认 1.7B（下载提示）
pub fn recommend(conditions: &DeviceConditions) -> ModelRecommendation {
    let (recommended_id, label, reason) = match conditions {
        c if c.has_gpu && c.available_ram_mb >= 8192 && c.model_17_exists => {
            (
                "qwen3-tts-1.7b".to_string(),
                "1.7B int4（推荐）".to_string(),
                format!(
                    "GPU 可用 + 内存充足（{:.0}GB），推荐使用 1.7B 获得最佳音质。\n\
                     支持 VoiceDesign 音色描述，6 种预设音色，合成质量更高。",
                    c.available_ram_mb as f64 / 1024.0
                ),
            )
        }
        c if c.model_06_exists && (c.available_ram_mb > 0 && c.available_ram_mb < 8192) => {
            (
                "qwen3-tts-0.6b".to_string(),
                "0.6B（推荐）".to_string(),
                format!(
                    "可用内存仅 {:.0}GB，推荐使用 0.6B 模型降低内存占用。\n\
                     加载更快，9 种预设音色（含方言），适合一般合成需求。",
                    c.available_ram_mb as f64 / 1024.0
                ),
            )
        }
        c if !c.has_gpu && c.model_06_exists => {
            (
                "qwen3-tts-0.6b".to_string(),
                "0.6B（推荐）".to_string(),
                "CPU 模式运行，推荐使用 0.6B 模型。\n\
                 参数量更小，推理速度更快，内存占用更低。"
                    .to_string(),
            )
        }
        c if c.model_17_exists => {
            (
                "qwen3-tts-1.7b".to_string(),
                "1.7B int4".to_string(),
                "仅检测到 1.7B 模型文件，将使用此变体。\n\
                 int4 量化版本体积更小，强制使用 CPU 推理。"
                    .to_string(),
            )
        }
        c if c.model_06_exists => {
            (
                "qwen3-tts-0.6b".to_string(),
                "0.6B".to_string(),
                "仅检测到 0.6B 模型文件，将使用此变体。".to_string(),
            )
        }
        _ => {
            (
                "qwen3-tts-1.7b".to_string(),
                "1.7B int4（默认）".to_string(),
                "未检测到模型文件，将使用默认 1.7B int4 变体。\n\
                 请先通过 `votex model download qwen3-tts-0.6b`（约 5.6GB）\
                 或 `votex model download qwen3-tts-1.7b`（约 6.2GB）下载模型。"
                    .to_string(),
            )
        }
    };

    ModelRecommendation {
        recommended_id,
        reason,
        label,
        conditions: conditions.clone(),
    }
}

/// 获取可用物理内存 (MB)
///
/// 使用 Windows API GlobalMemoryStatusEx
fn get_available_physical_memory_mb() -> u64 {
    #[cfg(target_os = "windows")]
    {
        use std::mem;
        type DWORD = u32;
        #[repr(C)]
        struct MEMORYSTATUSEX {
            dw_length: DWORD,
            dw_memory_load: DWORD,
            ull_total_phys: u64,
            ull_avail_phys: u64,
            ull_total_page_file: u64,
            ull_avail_page_file: u64,
            ull_total_virtual: u64,
            ull_avail_virtual: u64,
            ull_avail_extended_virtual: u64,
        }
        extern "system" {
            fn GlobalMemoryStatusEx(lp_buffer: *mut MEMORYSTATUSEX) -> i32;
        }
        let mut mem = MEMORYSTATUSEX {
            dw_length: mem::size_of::<MEMORYSTATUSEX>() as DWORD,
            dw_memory_load: 0,
            ull_total_phys: 0,
            ull_avail_phys: 0,
            ull_total_page_file: 0,
            ull_avail_page_file: 0,
            ull_total_virtual: 0,
            ull_avail_virtual: 0,
            ull_avail_extended_virtual: 0,
        };
        unsafe {
            if GlobalMemoryStatusEx(&mut mem) != 0 && mem.ull_avail_phys > 0 {
                mem.ull_avail_phys / (1024 * 1024)
            } else {
                4096 // 探测失败，保守假设 4GB
            }
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        4096 // 非 Windows 平台默认假设 4GB
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommend_gpu_ram_17() {
        let cond = DeviceConditions {
            has_gpu: true,
            available_ram_mb: 16384,
            model_06_exists: true,
            model_17_exists: true,
        };
        let rec = recommend(&cond);
        assert!(rec.recommended_id.contains("1.7b"));
        assert!(rec.label.contains("推荐"));
    }

    #[test]
    fn test_recommend_cpu_06() {
        let cond = DeviceConditions {
            has_gpu: false,
            available_ram_mb: 16384,
            model_06_exists: true,
            model_17_exists: true,
        };
        let rec = recommend(&cond);
        assert!(rec.recommended_id.contains("0.6b"), "CPU 模式应推荐 0.6B");
    }

    #[test]
    fn test_recommend_low_ram_06() {
        let cond = DeviceConditions {
            has_gpu: false,
            available_ram_mb: 4096,
            model_06_exists: true,
            model_17_exists: true,
        };
        let rec = recommend(&cond);
        assert!(rec.recommended_id.contains("0.6b"), "内存不足应推荐 0.6B");
    }

    #[test]
    fn test_recommend_only_17() {
        let cond = DeviceConditions {
            has_gpu: false,
            available_ram_mb: 16384,
            model_06_exists: false,
            model_17_exists: true,
        };
        let rec = recommend(&cond);
        assert!(rec.recommended_id.contains("1.7b"), "仅有 1.7B 时应推荐 1.7B");
    }

    #[test]
    fn test_recommend_default() {
        let cond = DeviceConditions {
            has_gpu: false,
            available_ram_mb: 16384,
            model_06_exists: false,
            model_17_exists: false,
        };
        let rec = recommend(&cond);
        assert!(!rec.recommended_id.is_empty(), "应返回默认推荐");
    }
}
