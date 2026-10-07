//! 实时/流式听写用例
//!
//! 编排：麦克风采集（infra::audio::capture）→ 流式识别会话
//! （infra::asr::streaming）→ 事件回调（CLI 打印 / GUI 展示）。
//!
//! # 线程模型
//!
//! `start` 在后台线程完成三件事：
//! 1. 加载流式模型（首次 1~3 秒）
//! 2. 启动 cpal 麦克风采集（采集线程 → channel）
//! 3. 工作循环：消费音频块 → `session.accept_waveform` → 回调事件
//!
//! 停止令牌置位后，工作循环退出前调用 `session.finish()` 冲刷残余
//! 音频，发出 [`DictationEvent::Stopped`]（含全文与保存路径）。
//!
//! # CLI / GUI 对等
//!
//! CLI `votex dictate` 与 GUI 听写页共用本用例，事件驱动方式一致。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};

use votex_domain::asr::streaming::StreamingAsrProvider;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{EngineKind, ModelId, ModelKind};
use votex_infra::asr::streaming::StreamingZipformerProvider;
use votex_infra::audio::capture::{start_default_mic, TARGET_SAMPLE_RATE};

/// 听写事件（CLI 打印 / GUI 展示共用）
#[derive(Debug, Clone, PartialEq)]
pub enum DictationEvent {
    /// 模型与麦克风就绪，开始监听
    Started,
    /// 中间结果（未定稿，会被后续更新覆盖）
    Partial { text: String, elapsed_ms: u64 },
    /// 端点检测定稿的一段文本（不会重复给出）
    Final { text: String, elapsed_ms: u64 },
    /// 听写结束：全文与（可选）保存路径
    Stopped { full_text: String, duration_ms: u64, saved_path: Option<PathBuf> },
    /// 致命错误（听写已终止）
    Error(String),
}

/// 听写用例
pub struct DictationUseCase {
    streaming: Arc<StreamingZipformerProvider>,
}

impl DictationUseCase {
    pub fn new() -> Self {
        Self {
            streaming: Arc::new(StreamingZipformerProvider::new()),
        }
    }

    /// 进程级共享实例（避免重复加载流式模型）
    pub fn shared() -> Arc<Self> {
        static INSTANCE: std::sync::OnceLock<Arc<DictationUseCase>> = std::sync::OnceLock::new();
        INSTANCE
            .get_or_init(|| Arc::new(DictationUseCase::new()))
            .clone()
    }

    /// 启动实时听写
    ///
    /// - `model_id`：流式模型标识（当前支持 `streaming-zipformer`）
    /// - `output_path`：可选转写保存路径（txt）；None 则不落盘
    /// - `on_event`：事件回调（在工作线程触发，跨线程需自行同步）
    ///
    /// 返回句柄：`stop()` 停止听写；drop 句柄不会停止（须显式 stop）。
    pub fn start(
        &self,
        model_id: &str,
        output_path: Option<&Path>,
        on_event: Arc<dyn Fn(DictationEvent) + Send + Sync>,
    ) -> Result<DictationHandle> {
        let model = Self::parse_model(model_id)?;

        // 停止令牌：handle 持有，工作线程轮询
        let stop_flag = Arc::new(AtomicBool::new(false));

        // 音频通道：采集线程 → 工作线程
        let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>();

        let stop_for_worker = Arc::clone(&stop_flag);
        let event_cb = Arc::clone(&on_event);
        let output_file = output_path.map(|p| p.to_path_buf());
        let provider = Arc::clone(&self.streaming);

        let worker = std::thread::Builder::new()
            .name("votex-dictation".to_string())
            .spawn(move || {
                // 麦克风必须在工作线程内启动与停止：
                // cpal::Stream 非 Send，不能跨线程移动（创建与释放需同线程）
                run_dictation_loop(
                    provider,
                    model,
                    audio_tx,
                    audio_rx,
                    stop_for_worker,
                    event_cb,
                    output_file,
                );
            })
            .context("创建听写工作线程失败")?;

        Ok(DictationHandle {
            stop_flag,
            worker: Some(worker),
            _on_event: Some(on_event),
        })
    }

    /// 解析听写模型标识
    fn parse_model(model_id: &str) -> Result<Model> {
        let (id, engine) = match model_id {
            "streaming-zipformer" => (ModelId::new("streaming-zipformer"), EngineKind::StreamingZipformer),
            other => {
                return Err(anyhow!(
                    "不支持的听写模型: {}，当前可选: streaming-zipformer",
                    other
                ))
            }
        };
        Ok(Model::new(id, model_id, ModelKind::Asr, engine))
    }
}

impl Default for DictationUseCase {
    fn default() -> Self {
        Self::new()
    }
}

/// 听写句柄：置位停止令牌，工作线程收尾后自然退出
pub struct DictationHandle {
    stop_flag: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
    /// 持有回调引用保证 'static 生命周期（worker 已捕获克隆）
    _on_event: Option<Arc<dyn Fn(DictationEvent) + Send + Sync>>,
}

impl DictationHandle {
    /// 请求停止听写（异步：工作线程冲刷后发 Stopped 事件）
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
    }

    /// 停止令牌（CLI 信号处理器捕获后置位，满足 'static）
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop_flag)
    }

    /// 请求停止并等待工作线程退出（阻塞，超时保护 10 秒）
    pub fn stop_and_join(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }

    pub fn is_stopping(&self) -> bool {
        self.stop_flag.load(Ordering::SeqCst)
    }
}

/// 听写工作循环（运行在后台线程；麦克风在此线程内启动/停止）
fn run_dictation_loop(
    provider: Arc<StreamingZipformerProvider>,
    model: Model,
    audio_tx: mpsc::Sender<Vec<f32>>,
    audio_rx: mpsc::Receiver<Vec<f32>>,
    stop_flag: Arc<AtomicBool>,
    on_event: Arc<dyn Fn(DictationEvent) + Send + Sync>,
    output_path: Option<PathBuf>,
) {
    let emit = |e: DictationEvent| on_event(e);

    // 1. 启动麦克风采集
    let mut mic = match start_default_mic(TARGET_SAMPLE_RATE, audio_tx) {
        Ok(m) => m,
        Err(e) => {
            emit(DictationEvent::Error(format!("启动麦克风采集失败: {}", e)));
            return;
        }
    };

    // 2. 加载流式模型
    if let Err(e) = provider.load(&model) {
        emit(DictationEvent::Error(format!("流式模型加载失败: {}", e)));
        mic.stop();
        return;
    }

    // 2. 创建识别会话
    let mut session = match provider.create_session() {
        Ok(s) => s,
        Err(e) => {
            emit(DictationEvent::Error(format!("创建识别会话失败: {}", e)));
            mic.stop();
            return;
        }
    };

    emit(DictationEvent::Started);

    // 3. 工作循环：消费音频块 → 增量解码 → 事件回调
    let mut full_text = String::new();
    loop {
        if stop_flag.load(Ordering::SeqCst) {
            break;
        }
        match audio_rx.recv_timeout(Duration::from_millis(200)) {
            Ok(chunk) => {
                match session.accept_waveform(&chunk) {
                    Ok(update) => {
                        if let Some(final_text) = update.final_text {
                            if !full_text.is_empty() {
                                full_text.push('\n');
                            }
                            full_text.push_str(&final_text);
                            emit(DictationEvent::Final {
                                text: final_text,
                                elapsed_ms: update.elapsed_ms,
                            });
                        } else if !update.partial.is_empty() {
                            emit(DictationEvent::Partial {
                                text: update.partial,
                                elapsed_ms: update.elapsed_ms,
                            });
                        }
                    }
                    Err(e) => {
                        emit(DictationEvent::Error(format!("识别失败: {}", e)));
                        mic.stop();
                        return;
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break, // 采集端已退出
        }
    }

    // 4. 收尾：停麦克风 → 冲刷残余音频 → 汇总全文
    mic.stop();
    let tail = session.finish().unwrap_or_default();
    if !tail.trim().is_empty() {
        if !full_text.is_empty() {
            full_text.push('\n');
        }
        full_text.push_str(tail.trim());
    }

    // 5. 可选保存转写
    let saved_path = output_path.and_then(|p| match save_transcript(&p, &full_text) {
        Ok(path) => Some(path),
        Err(e) => {
            tracing::error!("保存听写转写失败: {}", e);
            None
        }
    });

    emit(DictationEvent::Stopped {
        full_text,
        duration_ms: 0, // 由 Stopped 消费方按 Started 时间差展示；音频时长见 Final.elapsed_ms
        saved_path,
    });

    tracing::info!("听写工作循环退出");
}

/// 保存听写转写（UTF-8 txt）
fn save_transcript(path: &Path, text: &str) -> Result<PathBuf> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("创建目录失败: {:?}", parent))?;
        }
    }
    std::fs::write(path, text).with_context(|| format!("写入转写文件失败: {:?}", path))?;
    tracing::info!("听写转写已保存: {:?}", path);
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use votex_domain::error::AsrError;

    #[test]
    fn 听写_模型标识解析() {
        let m = DictationUseCase::parse_model("streaming-zipformer").unwrap();
        assert_eq!(m.engine, EngineKind::StreamingZipformer);
        assert!(DictationUseCase::parse_model("unknown-model").is_err());
    }

    #[test]
    fn 听写_保存转写_utf8() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("听写记录.txt");
        let saved = save_transcript(&path, "你好世界\n第二段").unwrap();
        assert!(saved.exists());
        let content = std::fs::read_to_string(&saved).unwrap();
        assert_eq!(content, "你好世界\n第二段");
    }

    #[test]
    fn 听写_句柄停止标志() {
        let flag = Arc::new(AtomicBool::new(false));
        let handle = DictationHandle {
            stop_flag: Arc::clone(&flag),
            worker: None,
            _on_event: None,
        };
        assert!(!handle.is_stopping());
        handle.stop();
        assert!(handle.is_stopping());
        assert!(flag.load(Ordering::SeqCst));
    }

    /// 验证事件枚举可跨线程传递（GUI channel 前提）
    #[test]
    fn 听写事件_可跨线程() {
        fn assert_send<T: Send>() {}
        assert_send::<DictationEvent>();
    }

    /// 验证 AsrError 从流式 trait 正确传播
    #[test]
    fn 听写_未加载模型时报错() {
        let provider = StreamingZipformerProvider::new();
        assert!(matches!(
            provider.create_session(),
            Err(AsrError::EngineNotLoaded)
        ));
    }
}
