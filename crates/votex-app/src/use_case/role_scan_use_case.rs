//! 小说角色扫描用例（docs/26 模块一 / CLI `role scan`）
//!
//! 输入整本小说 → 纯规则提取角色候选表 → 可选的自动音色分配 → JSON 落盘。
//!
//! # 为什么是纯规则而非 LLM
//!
//! 1. **成本**：100 万字逐章送 LLM 费用不可接受，规则方案零调用；
//! 2. **离线**：AGENTS.md 硬约束「用户文本内容不上传网络」；
//! 3. **延迟**：全书扫描是纯字符串操作，1083 章实测 < 1s。
//!
//! 代价是覆盖不到无引号的间接引语，但那类片段本就走 `dialogue_default` 兜底。
//!
//! # CLI / GUI 对等（AGENTS.md 硬约束）
//!
//! 业务逻辑全部在本层；`votex-cli::commands::role` 与 `votex-gui::pages::role_page`
//! 都只能调用这里，不得各自重写扫描规则。

use anyhow::{Context, Result};
use std::path::Path;
use std::time::Instant;

use votex_domain::tts::role::{
    build_role_map, extract_role_candidates, RoleAssignments, RoleCandidate, RoleVoiceMap,
};
use votex_domain::tts::value_object::{VoiceGender, VoiceMeta};

/// 角色扫描请求
#[derive(Debug, Clone)]
pub struct RoleScanRequest {
    /// 小说文件路径（txt / md / epub / docx / pdf）
    pub input: String,
    /// 返回候选上限
    pub top_n: usize,
    /// 模型根目录（用于枚举可用音色池）
    pub models_dir: String,
    /// 是否自动分配音色（生成可直接用于 `--role-map` 的映射表）
    pub assign_voices: bool,
}

/// 角色扫描结果
#[derive(Debug, Clone)]
pub struct RoleScanResult {
    /// 角色候选（按频次降序）
    pub candidates: Vec<RoleCandidate>,
    /// 自动生成的音色映射；`assign_voices=false` 时为 None
    pub role_map: Option<RoleVoiceMap>,
    /// 实际使用的音色池
    pub voices: Vec<VoiceMeta>,
    /// 全书耗时（毫秒）
    pub elapsed_ms: u128,
}

/// 可选的 JSON 输出格式（`--json`）
///
/// 与终端可读表格分开：CLI 默认给人看，`--json` 给脚本/GUI 消费。
#[derive(Debug, Clone, serde::Serialize)]
pub struct RoleScanJson {
    pub source: String,
    pub candidate_count: usize,
    pub elapsed_ms: u128,
    /// 音色池构成摘要（如 `中文女声 55/ 中文男声 45/ 英文 3`）
    pub voice_pool_summary: String,
    pub candidates: Vec<RoleCandidateJson>,
    /// 自动生成的role_map（`--assign-voices` 时存在）
    pub role_map: Option<RoleVoiceMap>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RoleCandidateJson {
    pub name: String,
    pub mentions: usize,
    /// 规则确认的性别（`neutral` = 未识别出，需人工确认）
    pub gender: &'static str,
    /// 推测性别（仅提示，不参与分配）
    pub suggested_gender: Option<&'static str>,
    pub first_chapter: Option<usize>,
    /// 首段真实台词，供试听
    pub sample_line: Option<String>,
    /// 自动分配到的音色
    pub assigned_voice: Option<String>,
}

/// 角色扫描用例
pub struct RoleScanUseCase;

impl RoleScanUseCase {
    /// 执行角色扫描
    pub fn execute(&self, req: RoleScanRequest) -> Result<RoleScanResult> {
        let path = Path::new(&req.input);
        if !path.exists() {
            anyhow::bail!("输入文件不存在: {}", req.input);
        }

        // 编码探测走门面（UTF-8/GBK 自动判别），CLI 不直接碰 infra
        let text = crate::platform::text::extract_document_text(path)
            .with_context(|| format!("读取小说失败: {}", req.input))?;

        let voices = crate::platform::tts::list_engine_voices(Path::new(&req.models_dir));

        let start = Instant::now();
        let candidates = extract_role_candidates(&text, req.top_n);
        let elapsed_ms = start.elapsed().as_millis();

        let role_map = if req.assign_voices {
            if voices.is_empty() {
                // 不静默失败：用户显式要求了分配，必须告知音色池为空的原因
                anyhow::bail!(
                    "音色池为空，无法自动分配音色。\n\
                     请确认已下载 Kokoro 模型（models/tts/kokoro-82m-v1.1-zh/voices/），\n\
                     或去掉 --assign-voices 仅查看角色候选。"
                );
            }
            Some(build_role_map(&candidates, &voices, &RoleAssignments::default()))
        } else {
            None
        };

        Ok(RoleScanResult {
            candidates,
            role_map,
            voices,
            elapsed_ms,
        })
    }

    /// 序列化为 JSON 输出格式
    pub fn to_json(&self, result: &RoleScanResult, source: &str) -> Result<String> {
        let candidates = result
            .candidates
            .iter()
            .map(|c| RoleCandidateJson {
                name: c.name.clone(),
                mentions: c.mentions,
                gender: gender_str(c.gender),
                suggested_gender: c.suggested_gender.map(gender_str),
                first_chapter: c.first_chapter,
                sample_line: c.sample_line.clone(),
                assigned_voice: result
                    .role_map
                    .as_ref()
                    .and_then(|m| m.roles.get(&c.name).cloned()),
            })
            .collect();

        let out = RoleScanJson {
            source: source.to_string(),
            candidate_count: result.candidates.len(),
            elapsed_ms: result.elapsed_ms,
            voice_pool_summary: summarize_pool(&result.voices),
            candidates,
            role_map: result.role_map.clone(),
        };
        serde_json::to_string_pretty(&out).context("序列化角色扫描结果失败")
    }
}

fn gender_str(g: VoiceGender) -> &'static str {
    match g {
        VoiceGender::Male => "male",
        VoiceGender::Female => "female",
        VoiceGender::Child => "child",
        VoiceGender::Neutral => "neutral",
    }
}

/// 音色池构成摘要（用户据此判断「有没有女声可选」等）
fn summarize_pool(voices: &[VoiceMeta]) -> String {
    if voices.is_empty() {
        return "空（未找到 Kokoro 音色）".to_string();
    }
    let count = |g: VoiceGender| voices.iter().filter(|v| v.gender == g).count();
    let mut parts = Vec::new();
    if count(VoiceGender::Female) > 0 {
        parts.push(format!("中文女声 {}", count(VoiceGender::Female)));
    }
    if count(VoiceGender::Male) > 0 {
        parts.push(format!("中文男声 {}", count(VoiceGender::Male)));
    }
    let english = voices
        .iter()
        .filter(|v| v.gender == VoiceGender::Female || v.gender == VoiceGender::Male)
        .filter(|v| v.locale == votex_domain::tts::value_object::VoiceLocale::English)
        .count();
    if english > 0 {
        parts.push(format!("英文 {}", english));
    }
    if count(VoiceGender::Child) > 0 {
        parts.push(format!("童声 {}", count(VoiceGender::Child)));
    }
    if parts.is_empty() {
        format!("{} 个（无性别分类）", voices.len())
    } else {
        format!("{}（共 {}）", parts.join(" / "), voices.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// 测试工作区根（绝对路径）
    ///
    /// 与项目既有测试一致：`CARGO_MANIFEST_DIR` 是 `crates/votex-app`，
    /// 直接 `join("../../")` 交由文件系统解析，**不要**手动 pop——
    /// 那样容易多退一层（`../../tmp` 会变成项目根的父目录）。
    /// 也不能用相对 `"models"`：单测 CWD 取决于运行方式，
    /// 会导致音色池枚举为空（项目内sensevoice / qwen3_asr 已踩过此坑）。
    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn fixture() -> RoleScanRequest {
        // 素材优先用整本小说，退化到章节样本
        let root = workspace_root();
        let input = [
            "tmp/《奶爸大文豪》作者：肉都督.txt",
            "tmp/e2e_ch1.txt",
            "tmp/novel.txt",
        ]
        .iter()
        .map(|p| root.join(p))
        .find(|p| p.exists())
        .unwrap_or_else(|| root.join("tmp/novel.txt"));
        RoleScanRequest {
            input: input.to_string_lossy().to_string(),
            top_n: 20,
            models_dir: root.join("models").to_string_lossy().to_string(),
            assign_voices: true,
        }
    }

    #[test]
    fn 角色扫描_真实素材可跑通() {
        let uc = RoleScanUseCase;
        let req = fixture();
        // 素材本身必须存在（否则无法验证扫描逻辑）
        assert!(
            std::path::Path::new(&req.input).exists(),
            "测试素材不存在: {}",
            req.input
        );
        let has_voices = std::path::Path::new(&req.models_dir)
            .join("tts/kokoro-82m-v1.1-zh/voices")
            .exists();

        let result = match uc.execute(req.clone()) {
            Ok(r) => r,
            Err(e) if !has_voices => {
                // 无 Kokoro 模型时--assign-voices 应报错并给出可操作提示
                assert!(
                    e.to_string().contains("音色池为空"),
                    "无模型时应明确告知音色池为空，实际: {}",
                    e
                );
                return; // 依赖外部模型资产的环境到此为止
            }
            Err(e) => panic!("扫描应成功: {}", e),
        };

        assert!(!result.candidates.is_empty(), "应提取出角色候选");

        // 候选按频次降序
        for pair in result.candidates.windows(2) {
            assert!(pair[0].mentions >= pair[1].mentions);
        }

        if has_voices {
            assert!(!result.voices.is_empty(), "应枚举到音色池");
            let map = result.role_map.as_ref().expect("assign_voices=true 时应生成映射表");
            // 旁白与兜底必须落到真实音色（真实池无中性音色，故不能为 None）
            assert!(map.narrator.is_some(), "旁白必须有音色");
            assert!(map.dialogue_default.is_some(), "台词兜底必须有音色");
            let json = serde_json::to_string(map).unwrap();
            let back = RoleVoiceMap::from_json_str(&json).expect("应可反序列化");
            assert_eq!(back.roles, map.roles);
        }
    }

    #[test]
    fn 角色扫描_不分配音色时跳过映射生成() {
        let uc = RoleScanUseCase;
        let mut req = fixture();
        req.assign_voices = false;
        let result = uc.execute(req).expect("扫描应成功");
        assert!(result.role_map.is_none(), "未要求分配时不应生成映射");
        assert!(!result.candidates.is_empty());
    }

    #[test]
    fn 角色扫描_文件不存在报明确错误() {
        let uc = RoleScanUseCase;
        let req = RoleScanRequest {
            input: "绝对不存在的小说.txt".to_string(),
            top_n: 10,
            models_dir: "models".to_string(),
            assign_voices: false,
        };
        let err = uc.execute(req).unwrap_err();
        assert!(
            err.to_string().contains("输入文件不存在"),
            "错误信息应指明文件不存在，实际: {}",
            err
        );
    }

    #[test]
    fn 角色扫描_音色池为空时分配应报错而非静默() {
        let uc = RoleScanUseCase;
        let mut req = fixture();
        req.models_dir = "绝对不存在的模型目录".to_string();
        req.assign_voices = true;
        let err = uc.execute(req).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("音色池为空"),
            "应明确告知音色池为空，实际: {}",
            msg
        );
    }

    #[test]
    fn JSON输出_含候选与池摘要() {
        let uc = RoleScanUseCase;
        let mut req = fixture();
        req.assign_voices = false; // 不依赖音色资产
        let result = uc.execute(req).expect("扫描应成功");
        let json = uc.to_json(&result, "测试来源").unwrap();

        assert!(json.contains("\"candidate_count\""));
        assert!(json.contains("\"voice_pool_summary\""));
        assert!(json.contains("\"assigned_voice\""));
        // JSON 必须可被反序列化（供 GUI/脚本消费）
        let _: serde_json::Value = serde_json::from_str(&json).expect("JSON 输出应合法");
    }

    #[test]
    fn 池摘要_空池有明确文案() {
        assert!(summarize_pool(&[]).contains("空"));
    }
}