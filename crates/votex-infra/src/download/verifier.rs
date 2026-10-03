use anyhow::Result;
use sha2::{Digest, Sha256};
use std::path::Path;

/// SHA256 校验器
pub struct Verifier;

impl Verifier {
    /// 计算文件的 SHA256 哈希值
    pub fn sha256(file_path: &Path) -> Result<String> {
        let mut file = std::fs::File::open(file_path)?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        let hash = hasher.finalize();
        Ok(format!("{:x}", hash))
    }

    /// 校验文件 SHA256 是否匹配预期值
    pub fn verify_sha256(file_path: &Path, expected: &str) -> Result<bool> {
        let actual = Self::sha256(file_path)?;
        Ok(actual.eq_ignore_ascii_case(expected))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn sha256_计算文件哈希() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(b"hello world").unwrap();

        let hash = Verifier::sha256(&file_path).unwrap();
        // SHA256("hello world") 的已知值
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn verify_sha256_匹配() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(b"hello world").unwrap();

        let expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        assert!(Verifier::verify_sha256(&file_path, expected).unwrap());
    }

    #[test]
    fn verify_sha256_不匹配() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(b"hello world").unwrap();

        assert!(!Verifier::verify_sha256(&file_path, "0000dead").unwrap());
    }

    #[test]
    fn verify_sha256_大小写不敏感() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(b"hello world").unwrap();

        let expected = "B94D27B9934D3E08A52E52D7DA7DABFAC484EFE37A5380EE9088F7ACE2EFCDE9";
        assert!(Verifier::verify_sha256(&file_path, expected).unwrap());
    }
}
