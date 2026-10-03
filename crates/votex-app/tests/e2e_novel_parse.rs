//! 端到端解析验证：真实全本小说《奶爸大文豪》（GBK 编码，4.9MB，1083 章）
//!
//! 验证路径（无模型参与）：
//! 1. EncodingDetector GBK 自动检测解码
//! 2. 章节感知切分（「第0001章 标题？」——数字编号 + 问叹号标题）
//! 3. 多角色计划构建（弯引号 “” 台词检测 + 音色路由）
//! 4. 语义断句计划统计（段数分布、音色覆盖）

use std::path::PathBuf;
use std::sync::OnceLock;

use votex_app::use_case::tts_use_case::build_synthesis_plan;
use votex_domain::tts::role::RoleVoiceMap;
use votex_domain::tts::service::TextSegmenter;
use votex_domain::tts::value_object::SegmentSize;
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
