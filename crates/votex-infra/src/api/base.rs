//! 在线 API 基础 HTTP 客户端
//!
//! 提供统一 HTTP 请求方法、API Key 解析、重试逻辑。
//! 被 DeepSeek、Azure Speech、阿里云等适配器共享使用。

use std::time::Duration;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_RETRY_MAX: u32 = 3;

/// API 请求配置
#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub api_key: Option<String>,
    pub api_secret: Option<String>,
    pub endpoint: Option<String>,
    pub region: Option<String>,
    pub timeout: Duration,
    pub retry_max: u32,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            api_secret: None,
            endpoint: None,
            region: None,
            timeout: DEFAULT_TIMEOUT,
            retry_max: DEFAULT_RETRY_MAX,
        }
    }
}

/// 基础 API 客户端
pub struct BaseApiClient {
    pub config: ApiConfig,
    client: reqwest::blocking::Client,
}

impl BaseApiClient {
    pub fn new(config: ApiConfig) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(config.timeout)
            .user_agent("Votex/1.0")
            .build()
            .expect("创建 HTTP 客户端失败");
        Self { config, client }
    }

    /// 解析 API Key，支持从环境变量读取
    pub fn resolve_api_key(&self, key: Option<&str>, env_var: &str) -> Result<String, ApiError> {
        match key {
            Some(k) if !k.is_empty() => Ok(k.to_string()),
            _ => match std::env::var(env_var) {
                Ok(val) if !val.is_empty() => Ok(val),
                _ => Err(ApiError::ApiKeyNotConfigured(env_var.to_string())),
            },
        }
    }

    /// GET 请求
    pub fn get(&self, url: &str, headers: Option<&[(String, String)]>) -> Result<reqwest::blocking::Response, ApiError> {
        self.request("GET", url, headers, None, None, None)
    }

    /// POST 请求（JSON body）
    pub fn post_json(&self, url: &str, headers: Option<&[(String, String)]>, json: serde_json::Value) -> Result<reqwest::blocking::Response, ApiError> {
        self.request("POST", url, headers, Some(json), None, None)
    }

    /// POST 请求（raw bytes）
    pub fn post_bytes(&self, url: &str, headers: Option<&[(String, String)]>, body: &[u8]) -> Result<reqwest::blocking::Response, ApiError> {
        self.request("POST", url, headers, None, Some(body), None)
    }

    /// POST 请求（form params）
    pub fn post_form(&self, url: &str, headers: Option<&[(String, String)]>, params: &[(&str, &str)]) -> Result<reqwest::blocking::Response, ApiError> {
        self.request("POST", url, headers, None, None, Some(params))
    }

    /// 通用请求（含重试）
    fn request(
        &self,
        method: &str,
        url: &str,
        headers: Option<&[(String, String)]>,
        json: Option<serde_json::Value>,
        body: Option<&[u8]>,
        form: Option<&[(&str, &str)]>,
    ) -> Result<reqwest::blocking::Response, ApiError> {
        let mut last_err = None;
        for attempt in 0..self.config.retry_max {
            let mut req = match method {
                "GET" => self.client.get(url),
                "POST" => self.client.post(url),
                _ => return Err(ApiError::RequestFailed(format!("不支持的 HTTP 方法: {}", method))),
            };
            if let Some(h) = headers {
                for (k, v) in h {
                    req = req.header(k, v);
                }
            }
            if let Some(j) = &json {
                req = req.json(j);
            }
            if let Some(b) = body {
                req = req.body(b.to_vec());
            }
            if let Some(f) = form {
                req = req.form(f);
            }

            match req.send() {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        return Ok(resp);
                    }
                    let status_code = status.as_u16();
                    let body_text = resp.text().unwrap_or_default();
                    if status_code == 401 || status_code == 403 {
                        return Err(ApiError::AuthenticationFailed);
                    }
                    if status_code == 429 && attempt < self.config.retry_max - 1 {
                        std::thread::sleep(Duration::from_secs(1 << attempt));
                        last_err = Some(ApiError::RateLimited);
                        continue;
                    }
                    return Err(ApiError::RequestFailed(format!("HTTP {}: {}", status_code, body_text)));
                }
                Err(e) => {
                    last_err = Some(ApiError::NetworkError(e.to_string()));
                    if attempt < self.config.retry_max - 1 {
                        std::thread::sleep(Duration::from_secs(1 << attempt));
                    }
                }
            }
        }
        Err(last_err.unwrap_or(ApiError::RequestFailed("请求失败".to_string())))
    }
}

/// API 错误
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("API Key 未配置: {0}")]
    ApiKeyNotConfigured(String),

    #[error("认证失败（401/403）")]
    AuthenticationFailed,

    #[error("请求被限流（429）")]
    RateLimited,

    #[error("请求失败: {0}")]
    RequestFailed(String),

    #[error("网络错误: {0}")]
    NetworkError(String),

    #[error("响应解析失败: {0}")]
    ParseFailed(String),
}
