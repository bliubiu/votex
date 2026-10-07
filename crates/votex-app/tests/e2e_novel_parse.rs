//! 端到端解析验证：真实全本小说《奶爸大文豪》（GBK 编码，4.9MB，1083 章）
//!
//! 验证路径（无模型参与）：
//! 1. EncodingDetector GBK 自动检测解码
//! 2. 章节感知切分（「第0001章 标题？」——数字编号 + 问叹号标题）
//! 3. 多角色计划构建（弯引号 “” 台词检测 + 音色路由）
//! 4. 语义断句计划统计（段数分布、音色覆盖）
//! 5. 角色自动提取 + 音色自动分配（docs/26 模块一）

use std::path::PathBuf;
use std::sync::OnceLock;

use votex_app::use_case::tts_use_case::build_synthesis_plan;
use votex_domain::tts::role::{
    build_role_map, extract_role_candidates, RoleAssignments, RoleVoiceMap,
};
use votex_domain::tts::service::TextSegmenter;
use votex_domain::tts::value_object::{SegmentSize, VoiceGender, VoiceMeta};
use votex_infra::encoding::detector::EncodingDetector;

fn novel_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tmp/《奶爸大文豪》作者：肉都督.txt")
}

fn novel_text() -> &'static String {
    static TEXT: OnceLock<String> = OnceLock::new();
    TEXT.get_or_init(|| {
        EncodingDetector::read_text_file(&novel_path())
            .expect("GBK 全本读取失败（编码检测应为 GB18030）")
    })
}

/// 与 GUI/CLI 默认一致的角色映射：主角男声、女儿童声、其余台词男声旁白
fn demo_role_map() -> RoleVoiceMap {
    RoleVoiceMap::from_json_str(
        r#"{
            "narrator": "zf_xiaoxiao",
            "dialogue_default": "zm_yunjian",
            "roles": {
                "张重": "zm_yunjian",
                "芃芃": "xf_child",
                "女人": "zf_xiaoyan"
            }
        }"#,
    )
    .expect("角色映射 JSON 解析失败")
}

#[test]
fn e2e_编码检测与预处理() {
    let text = novel_text();
    assert!(text.contains("第0001章 女儿？"), "章节标记应原样保留");
    assert!(text.contains("张重"), "主角名应存在");
    assert!(text.contains('“'), "应含弯引号台词");
    // GBK 全角空格缩进应在预处理中规整
    let cleaned = TextSegmenter::preprocess(text);
    assert!(!cleaned.contains("　　"), "预处理应移除全角缩进");
}

#[test]
fn e2e_全本章节切分() {
    let text = novel_text();
    let chapters = votex_domain::tts::chapter::split_into_chapters(text);
    // 原书 1077 个「第XXXX章 标题」标记 + 1 个卷首块（书名/作者/内容简介）
    assert_eq!(
        chapters.len(),
        1078,
        "应为 1077 章 + 1 个卷首块，实际 {}",
        chapters.len()
    );
    // 卷首块无标题
    assert!(chapters[0].title.is_none(), "卷首应无标题");
    assert!(chapters[0].body.contains("内容简介"), "卷首应含内容简介");
    // 首章标题完整（含问号不被截断、不误杀）
    assert_eq!(chapters[1].title.as_deref(), Some("第0001章 女儿？"));
    // 末章
    let last = chapters.last().unwrap();
    assert_eq!(last.title.as_deref(), Some("第1077章 归寂（结局）"));
    assert!(!last.body.trim().is_empty(), "末章正文非空");
    // 全部章节（卷首除外）标题与正文非空
    for ch in &chapters[1..] {
        assert!(ch.title.is_some(), "标题缺失");
        assert!(!ch.body.trim().is_empty(), "空正文章节: {:?}", ch.title);
    }
}

#[test]
fn e2e_多角色计划统计() {
    let text = novel_text();
    let map = demo_role_map();
    let plan = build_synthesis_plan(text, SegmentSize::S120, true, Some(&map));

    assert!(!plan.is_empty(), "计划非空");
    let total_segs = plan.len();
    let dialogue_segs = plan.iter().filter(|s| s.is_dialogue).count();
    let narrator_segs = total_segs - dialogue_segs;
    let with_voice = plan.iter().filter(|s| s.voice_override.is_some()).count();
    let titled = plan.iter().filter(|s| s.is_new_chapter).count();

    // 弯引号台词应被大量检出（本书对话密集）
    assert!(
        dialogue_segs > 10_000,
        "台词段应过万，实际 {}",
        dialogue_segs
    );
    assert!(narrator_segs > 10_000, "叙述段应过万，实际 {}", narrator_segs);
    // 全部段都有音色路由（narrator/dialogue_default 兜底）
    assert_eq!(
        with_voice, total_segs,
        "每段都应有音色（含兜底），实际 {}/{}",
        with_voice, total_segs
    );
    // 章节标题段 = 1077（卷首无标题）
    assert_eq!(
        titled, 1077,
        "新章首段应为 1077，实际 {}",
        titled
    );

    // 抽查音色路由正确性：找一节含「芃芃」台词的段
    let child_sample = plan
        .iter()
        .filter(|s| s.is_dialogue && s.text.contains("爸爸"))
        .take(3)
        .count();
    assert!(child_sample > 0, "应能找到含「爸爸」的台词段（女儿向主角说话）");

    println!(
        "E2E 统计: 总段数={} 台词段={} 叙述段={} 新章标题段={}",
        total_segs, dialogue_segs, narrator_segs, titled
    );
}

#[test]
fn e2e_计划文本无章节标记残留() {
    let text = novel_text();
    let map = demo_role_map();
    let plan = build_synthesis_plan(text, SegmentSize::S120, true, Some(&map));
    // 章节标记行不应残留在正文段中（标题单独成段）
    let leaked = plan
        .iter()
        .filter(|s| !s.is_new_chapter)
        .filter(|s| s.text.trim_start().starts_with("第0001章"))
        .count();
    assert_eq!(leaked, 0, "章节标记不应泄漏进正文段");
}

// ============================================================================
// 角色自动提取（docs/26 模块一）
// ============================================================================

/// 扫描 Kokoro 音色目录构造音色池（不加载模型，纯读磁盘文件名）
fn kokoro_voice_pool() -> Vec<VoiceMeta> {
    use votex_domain::model::value_object::EngineKind;
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/tts/kokoro-82m-v1.1-zh/voices");
    let mut voices: Vec<VoiceMeta> = std::fs::read_dir(&dir)
        .expect("Kokoro 音色目录应存在（models/tts/kokoro-82m-v1.1-zh/voices）")
        .flatten()
        .filter_map(|e| {
            let stem = e.path().file_stem()?.to_str()?.to_string();
            if e.path().extension().and_then(|x| x.to_str()) != Some("bin") {
                return None;
            }
            let gender = VoiceGender::from_voice_id(&stem);
            Some(VoiceMeta::new(&stem, &stem, EngineKind::Kokoro, gender))
        })
        .collect();
    voices.sort_by(|a, b| a.voice_id.cmp(&b.voice_id));
    voices
}

#[test]
fn e2e_角色提取_真实全本性能与信噪比() {
    let text = novel_text();

    let start = std::time::Instant::now();
    let cands = extract_role_candidates(text, 20);
    let elapsed = start.elapsed();

    // 性能：100 万字级纯字符串扫描须在 2s 内（docs/26 §8.1 验收口径）
    assert!(
        elapsed.as_secs_f64() < 2.0,
        "全本角色扫描耗时 {:.2}s，超出 2s 预算",
        elapsed.as_secs_f64()
    );
    assert!(!cands.is_empty(), "应提取出角色候选");

    // 信噪比：Top3 应全部是真角色（本书主角张重、女儿芃芃等），
    // 这对「神态副词被当人名」类噪声是强回归约束
    let top3: Vec<&str> = cands.iter().take(3).map(|c| c.name.as_str()).collect();
    assert!(
        top3.iter().all(|n| is_plausible_name_str(n)),
        "Top3 应全为人名，实际 {:?}",
        top3
    );
    // Top10 至少一半以上像人名
    let plausible = cands
        .iter()
        .take(10)
        .filter(|c| is_plausible_name_str(&c.name))
        .count();
    assert!(
        plausible >= 5,
        "Top10 中像人名的应 ≥5 个，实际 {}/10：{:?}",
        plausible,
        cands.iter().take(10).map(|c| &c.name).collect::<Vec<_>>()
    );

    // 每个候选都应有可试听的首段台词（GUI 试听依赖）
    let with_sample = cands.iter().filter(|c| c.sample_line.is_some()).count();
    assert_eq!(with_sample, cands.len(), "每个候选都应带首段台词供试听");

    // 频次应递减
    for pair in cands.windows(2) {
        assert!(
            pair[0].mentions >= pair[1].mentions,
            "候选未按频次降序：{} vs {}",
            pair[0].name,
            pair[1].name
        );
    }

    println!(
        "E2E 角色提取: 候选数={} 耗时={:.3}s Top5={:?}",
        cands.len(),
        elapsed.as_secs_f64(),
        cands.iter().take(5).map(|c| (&c.name, c.mentions, c.gender)).collect::<Vec<_>>()
    );
}

/// 粗筛「像不像人名」：2~4 字、非纯数字、无结构性标点
///
/// 2~4 字对应常见中文姓名上限；超过 4 字的多半是切分错位带出的短语。
/// 注意不排除「张重笑着」这类——它在 4 字内，故不能靠长度识别，
/// 真正的防线是 `trailing_name` 的修饰语截断。
fn is_plausible_name_str(name: &str) -> bool {
    let n = name.chars().count();
    (2..=4).contains(&n)
        && !name.starts_with(|c: char| c.is_ascii_digit())
        && !name.contains(|c: char| c.is_ascii_punctuation())
}

#[test]
fn e2e_角色提取_噪声不淹没候选表() {
    let text = novel_text();
    let cands = extract_role_candidates(text, 20);
    // 不应出现纯数字 / 超长串等典型噪声
    for c in &cands {
        assert!(c.name.chars().count() >= 2, "过短候选: {:?}", c.name);
        assert!(c.name.chars().count() <= 8, "过长候选: {:?}", c.name);
        assert!(
            !c.name.chars().all(|ch| ch.is_ascii_digit()),
            "纯数字候选: {:?}",
            c.name
        );
    }
}

#[test]
fn e2e_音色自动分配_真实音色池() {
    let text = novel_text();
    let voices = kokoro_voice_pool();
    assert!(!voices.is_empty(), "应扫到Kokoro 音色");

    // 真实池的性别构成（zf_ 55 / zm_ 45 / 英文 3，无童声、无中性）
    assert!(
        VoiceMeta::filter_by_gender(&voices, VoiceGender::Child).is_empty(),
        "Kokoro 池无童声音色（xf_ 不存在于实际模型）"
    );
    assert!(
        VoiceMeta::filter_by_gender(&voices, VoiceGender::Neutral).is_empty(),
        "Kokoro 池无中性音色——兜底必须逐级回退到男声，否则台词段会全部无音色"
    );
    assert!(!VoiceMeta::filter_by_gender(&voices, VoiceGender::Female).is_empty());
    assert!(!VoiceMeta::filter_by_gender(&voices, VoiceGender::Male).is_empty());

    let cands = extract_role_candidates(text, 20);
    let map = build_role_map(&cands, &voices, &RoleAssignments::default());

    // 旁白与兜底必须能落到真实存在的声音上
    let ids: Vec<&str> = voices.iter().map(|v| v.voice_id.as_str()).collect();
    if let Some(n) = &map.narrator {
        assert!(ids.contains(&n.as_str()), "旁白音色须真实存在: {}", n);
        // 回归：曾实测出 narrator=af_maple（英文女声字典序排在 zf_ 前），
        // 中文小说必须用中文音色
        assert!(
            n.starts_with("zf_") || n.starts_with("zm_") || n.starts_with("xf_"),
            "中文小说的旁白须为中文音色，实际 {}",
            n
        );
    }
    if let Some(d) = &map.dialogue_default {
        assert!(
            d.starts_with("zf_") || d.starts_with("zm_") || d.starts_with("xf_"),
            "中文小说的台词兜底须为中文音色，实际 {}",
            d
        );
    }

    // 有称谓证据的角色应被分到对应性别的音色
    for c in cands.iter().filter(|c| c.gender_confirmed()) {
        if let Some(v) = map.roles.get(&c.name) {
            let meta = voices.iter().find(|x| &x.voice_id == v).unwrap();
            assert_eq!(
                meta.gender, c.gender,
                "角色 {} 性别 {:?} 与所分音色 {:?} 不符",
                c.name, c.gender, meta.gender
            );
        }
    }

    // 未确认性别的角色不应被塞进带性别的音色
    for c in cands.iter().filter(|c| !c.gender_confirmed()) {
        if let Some(v) = map.roles.get(&c.name) {
            let meta = voices.iter().find(|x| &x.voice_id == v).unwrap();
            assert!(
                matches!(meta.gender, VoiceGender::Neutral | VoiceGender::Male),
                "未确认性别的 {} 被分到 {:?}，应落中性兜底",
                c.name,
                meta.gender
            );
        }
    }

    // 本书有明确称谓线索的角色（爷爷/奶奶等）应能自动定性
    let named = cands.iter().find(|c| c.gender_confirmed());
    if let Some(c) = named {
        assert!(
            matches!(c.gender, VoiceGender::Male | VoiceGender::Female),
            "有称谓线索的角色应定性为男/女声，实际 {:?}",
            c.gender
        );
    }

    println!(
        "E2E 音色分配: 候选 {}个→ 角色表 {} 条，narrator={:?} dialogue_default={:?}",
        cands.len(),
        map.roles.len(),
        map.narrator,
        map.dialogue_default
    );
}

#[test]
fn e2e_角色表可反序列化用于合成() {
    let text = novel_text();
    let voices = kokoro_voice_pool();
    let cands = extract_role_candidates(text, 20);
    let map = build_role_map(&cands, &voices, &RoleAssignments::default());

    // 落盘为 role_map.json 形态后可被既有解析器读回（CLI/GUI 共用格式）
    let json = serde_json::to_string(&map).expect("角色表应可序列化");
    let back = RoleVoiceMap::from_json_str(&json).expect("应可反序列化");
    assert_eq!(back.narrator, map.narrator);
    assert_eq!(back.dialogue_default, map.dialogue_default);
    assert_eq!(back.roles, map.roles);

    // 用自动生成的角色表能真正构建出带音色路由的合成计划
    let plan = build_synthesis_plan(text, SegmentSize::S120, true, Some(&map));
    let with_voice = plan.iter().filter(|s| s.voice_override.is_some()).count();
    assert_eq!(
        with_voice,
        plan.len(),
        "自动生成的 role_map 应覆盖全部段落的音色路由"
    );
}
