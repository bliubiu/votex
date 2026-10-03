/// 构建脚本
///
/// ort crate 使用静态链接（features = ["ndarray"]，默认启用 download-binaries），
/// 无需额外配置。Kokoro v1.1-zh 模型已转换为 opset 18 以兼容当前版 ONNX Runtime。
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
