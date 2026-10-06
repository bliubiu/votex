//! 泛型引擎生命周期管理
//!
//! 封装 `Mutex<Option<T>>` 模式，统一处理引擎的加载/卸载/状态查询，
//! 消除各 Provider 中重复的状态管理代码。

use std::sync::Mutex;

/// 引擎生命周期管理器
///
/// 封装 `Mutex<Option<T>>` 模式，提供统一的：
/// - 加载/卸载操作
/// - 状态查询（is_loaded）
/// - 安全访问（with / with_mut）
///
/// # 类型参数
/// - `T`: 引擎类型（如 `Session`、`OfflineRecognizer` 等）
///
/// # 示例
/// ```ignore
/// struct MyProvider {
///     state: EngineState<Session>,
/// }
///
/// impl MyProvider {
///     fn load(&self, model_path: &Path) -> Result<()> {
///         let session = OrtSessionFactory::create(model_path)?;
///         self.state.load(session);
///         Ok(())
///     }
///
///     fn is_loaded(&self) -> bool {
///         self.state.is_loaded()
///     }
///
///     fn run_inference(&self) -> Result<()> {
///         self.state.with(|session| {
///             // 使用 session 进行推理
///         })
///     }
/// }
/// ```
pub struct EngineState<T> {
    inner: Mutex<Option<T>>,
}

impl<T> EngineState<T> {
    /// 创建空的引擎状态
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    /// 加载引擎（替换现有实例）
    ///
    /// # 参数
    /// - `value`: 引擎实例
    pub fn load(&self, value: T) {
        *self.inner.lock().unwrap_or_else(|e| e.into_inner()) = Some(value);
    }

    /// 卸载引擎（释放资源）
    pub fn unload(&self) {
        *self.inner.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// 检查引擎是否已加载
    pub fn is_loaded(&self) -> bool {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    /// 安全访问引擎（只读）
    ///
    /// # 参数
    /// - `f`: 访问闭包
    ///
    /// # 返回
    /// 成功返回闭包结果，引擎未加载时返回错误
    pub fn with<F, R>(&self, f: F) -> Result<R, EngineNotLoaded>
    where
        F: FnOnce(&T) -> R,
    {
        let guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let engine = guard.as_ref().ok_or(EngineNotLoaded)?;
        Ok(f(engine))
    }

    /// 安全访问引擎（可变）
    ///
    /// # 参数
    /// - `f`: 访问闭包
    ///
    /// # 返回
    /// 成功返回闭包结果，引擎未加载时返回错误
    pub fn with_mut<F, R>(&self, f: F) -> Result<R, EngineNotLoaded>
    where
        F: FnOnce(&mut T) -> R,
    {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let engine = guard.as_mut().ok_or(EngineNotLoaded)?;
        Ok(f(engine))
    }

    /// 获取引擎引用（如果已加载）
    ///
    /// 返回守卫而非裸引用，是为了让调用方**无法**在守卫存活期间
    /// 触发对同一引擎的 `load` / `unload`（那会 panic）。
    /// 需要访问引擎内容时优先用 [`with`](Self::with) / [`with_mut`](Self::with_mut)，
    /// 在闭包内完成全部工作。
    pub fn get(&self) -> Option<std::sync::MutexGuard<'_, Option<T>>> {
        Some(self.inner.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

impl<T> Default for EngineState<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// 引擎未加载错误
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineNotLoaded;

impl std::fmt::Display for EngineNotLoaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "引擎未加载")
    }
}

impl std::error::Error for EngineNotLoaded {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_初始状态为空() {
        let state: EngineState<String> = EngineState::new();
        assert!(!state.is_loaded());
    }

    #[test]
    fn load_后状态为已加载() {
        let state = EngineState::new();
        state.load("test".to_string());
        assert!(state.is_loaded());
    }

    #[test]
    fn unload_后状态为空() {
        let state = EngineState::new();
        state.load("test".to_string());
        assert!(state.is_loaded());

        state.unload();
        assert!(!state.is_loaded());
    }

    #[test]
    fn with_已加载时执行闭包() {
        let state = EngineState::new();
        state.load(42);

        let result = state.with(|v| *v * 2);
        assert_eq!(result.unwrap(), 84);
    }

    #[test]
    fn with_未加载时返回错误() {
        let state: EngineState<i32> = EngineState::new();
        let result = state.with(|v| *v);
        assert_eq!(result.err(), Some(EngineNotLoaded));
    }

    #[test]
    fn with_mut_可以修改引擎状态() {
        let state = EngineState::new();
        state.load(vec![1, 2, 3]);

        let result = state.with_mut(|v| {
            v.push(4);
            v.len()
        });
        assert_eq!(result.unwrap(), 4);
    }

    #[test]
    fn load_替换现有实例() {
        let state = EngineState::new();
        state.load("first".to_string());
        assert_eq!(state.with(|s| s.clone()).unwrap(), "first");

        state.load("second".to_string());
        assert_eq!(state.with(|s| s.clone()).unwrap(), "second");
    }

    #[test]
    fn default_等同于_new() {
        let state: EngineState<String> = EngineState::default();
        assert!(!state.is_loaded());
    }
}
