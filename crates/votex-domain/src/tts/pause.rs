/// 内联停顿标签解析
///
/// 语法：`[[pause:值]]`，值支持两种写法：
/// - 毫秒整数：`[[pause:300]]` = 300 毫秒
/// - 秒（带 s 后缀，可为小数）：`[[pause:0.5s]]` = 500 毫秒
///
/// 解析后标签从朗读文本中剔除，停顿在合成时以静音插入。
/// 无效标签（格式错误/超时值上限）直接剔除并记日志，不会朗读出「pause」字样。
///
/// 停顿时长上限 5 分钟，防止误写超大值。

/// 单位毫秒上限：5 分钟
const MAX_PAUSE_MS: u32 = 300_000;

/// 文本片段（含内联停顿）
#[derive(Debug, Clone, PartialEq)]
pub enum TextPiece {
    /// 一段需要朗读的文本
    Text(String),
    /// 一段静音停顿（毫秒）
    Pause(u32),
}

/// 解析毫秒/秒值，非法返回 None
fn parse_pause_value(raw: &str) -> Option<u32> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    if let Some(sec) = raw.strip_suffix('s').or_else(|| raw.strip_suffix('S')) {
        let sec_f: f64 = sec.trim().parse().ok()?;
        if !(0.0..=300.0).contains(&sec_f) {
            return None;
        }
        Some((sec_f * 1000.0).round() as u32)
    } else {
        raw.parse::<u32>().ok()
    }
}

/// 把文本按内联停顿标签切分为「朗读片段 + 停顿」序列
///
/// 相邻 Text 片段不会自动合并（保留原样语义），空 Text 片段会被剔除。
pub fn split_pause_tags(text: &str) -> Vec<TextPiece> {
    const OPEN: &str = "[[pause:";
    let mut pieces = Vec::new();
    let mut rest = text;

    while let Some(start) = rest.find(OPEN) {
        let before = &rest[..start];
        if !before.is_empty() {
            pieces.push(TextPiece::Text(before.to_string()));
        }
        let after_open = &rest[start + OPEN.len()..];
        match after_open.find("]]") {
            Some(end) => {
                let raw = &after_open[..end];
                match parse_pause_value(raw) {
                    Some(ms) if ms <= MAX_PAUSE_MS && ms > 0 => {
                        pieces.push(TextPiece::Pause(ms));
                    }
                    _ => {
                        // 非法/超限标签：剔除，不朗读（调用方记日志）
                        let _ = raw;
                    }
                }
                rest = &after_open[end + 2..];
            }
            None => {
                // 未闭合标签：剩余部分当普通文本（保守，不吞内容）
                pieces.push(TextPiece::Text(rest[start..].to_string()));
                rest = "";
            }
        }
    }

    if !rest.is_empty() {
        pieces.push(TextPiece::Text(rest.to_string()));
    }

    // 过滤纯空白 Text 片段，并合并相邻 Text 片段
    // （非法标签剔除后左右文本应无缝拼回一段）
    let mut merged: Vec<TextPiece> = Vec::new();
    for piece in pieces {
        match piece {
            TextPiece::Text(t) => {
                if t.trim().is_empty() {
                    continue;
                }
                match merged.last_mut() {
                    Some(TextPiece::Text(prev)) => prev.push_str(&t),
                    _ => merged.push(TextPiece::Text(t)),
                }
            }
            p @ TextPiece::Pause(_) => merged.push(p),
        }
    }
    merged
}

/// 把片段序列还原为纯朗读文本（调试/展示用）
pub fn join_text(pieces: &[TextPiece]) -> String {
    pieces
        .iter()
        .map(|p| match p {
            TextPiece::Text(t) => t.as_str(),
            TextPiece::Pause(_) => "",
        })
        .collect()
}

/// 统计片段中的总停顿毫秒数
pub fn total_pause_ms(pieces: &[TextPiece]) -> u32 {
    pieces
        .iter()
        .map(|p| match p {
            TextPiece::Pause(ms) => *ms,
            _ => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 停顿标签_毫秒写法() {
        let pieces = split_pause_tags("第一句。[[pause:300]]第二句。");
        assert_eq!(
            pieces,
            vec![
                TextPiece::Text("第一句。".into()),
                TextPiece::Pause(300),
                TextPiece::Text("第二句。".into()),
            ]
        );
    }

    #[test]
    fn 停顿标签_秒写法() {
        let pieces = split_pause_tags("甲[[pause:0.5s]]乙");
        assert_eq!(
            pieces,
            vec![
                TextPiece::Text("甲".into()),
                TextPiece::Pause(500),
                TextPiece::Text("乙".into()),
            ]
        );
        assert_eq!(total_pause_ms(&pieces), 500);
    }

    #[test]
    fn 停顿标签_非法值剔除() {
        let pieces = split_pause_tags("甲[[pause:abc]]乙");
        assert_eq!(pieces, vec![TextPiece::Text("甲乙".into())]);
    }

    #[test]
    fn 停顿标签_未闭合不吞文本() {
        let pieces = split_pause_tags("甲[[pause:300乙");
        // 未闭合标签整体按普通文本保留
        assert_eq!(pieces.len(), 1);
        assert!(pieces[0] == TextPiece::Text("甲[[pause:300乙".into()));
    }

    #[test]
    fn 停顿标签_多个连续() {
        let pieces = split_pause_tags("甲[[pause:100]][[pause:1s]]乙");
        assert_eq!(total_pause_ms(&pieces), 1100);
        assert_eq!(join_text(&pieces), "甲乙");
    }

    #[test]
    fn 停顿标签_超上限剔除() {
        // 5 分钟上限（300_000ms），60 万 ms 超限剔除
        let pieces = split_pause_tags("甲[[pause:600000]]乙");
        assert_eq!(pieces, vec![TextPiece::Text("甲乙".into())]);
    }

    #[test]
    fn 停顿标签_空文本() {
        assert!(split_pause_tags("").is_empty());
    }
}
