use votex_domain::error::DomainError;
use votex_domain::shared::value_object::TaskId;
use votex_domain::tts::entity::TtsTask;
use votex_domain::repository::TtsTaskRepository;
use std::collections::HashMap;
use std::sync::Mutex;

/// TTS 任务仓储实现
pub struct FileTaskRepo {
    tasks: Mutex<HashMap<TaskId, TtsTask>>,
}

impl FileTaskRepo {
    pub fn new() -> Self {
        Self {
            tasks: Mutex::new(HashMap::new()),
        }
    }
}

impl TtsTaskRepository for FileTaskRepo {
    fn find_by_id(&self, id: &TaskId) -> Option<TtsTask> {
        crate::persistence::guard::lock(&self.tasks).get(id).cloned()
    }

    fn save(&self, task: &TtsTask) -> Result<(), DomainError> {
        let mut tasks = self.tasks.lock().map_err(|e| {
            DomainError::Config(votex_domain::error::ConfigError::ReadFailed(e.to_string()))
        })?;
        tasks.insert(task.id.clone(), task.clone());
        Ok(())
    }

    fn delete(&self, id: &TaskId) -> Result<(), DomainError> {
        let mut tasks = self.tasks.lock().map_err(|e| {
            DomainError::Config(votex_domain::error::ConfigError::ReadFailed(e.to_string()))
        })?;
        tasks.remove(id);
        Ok(())
    }

    fn find_pending(&self) -> Vec<TtsTask> {
        self.tasks
            .lock()
            .map(|tasks| {
                tasks
                    .values()
                    .filter(|t| matches!(t.status, votex_domain::shared::value_object::TaskStatus::Queued))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    fn find_all(&self) -> Vec<TtsTask> {
        crate::persistence::guard::lock(&self.tasks).values().cloned().collect()
    }
}
