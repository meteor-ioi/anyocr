use crate::error::AnyOcrError;
use crate::types::ModelProfile;
use std::path::PathBuf;

pub const DET_SMALL_FILENAME: &str = "PP-OCRv6_det_small.onnx";
pub const REC_MEDIUM_FILENAME: &str = "PP-OCRv6_rec_medium.onnx";
pub const REC_MOBILE_FILENAME: &str = "PP-OCRv6_rec_small.onnx";
pub const REC_SERVER_FILENAME: &str = "PP-OCRv6_rec_server.onnx";
pub const DICT_FILENAME: &str = "ppocrv6_dict.txt";
pub const TABLE_FILENAME: &str = "slanet-plus.onnx";

/// 获取本地模型缓存主目录 (优先 ANYOCR_CACHE_DIR，其次 ~/.cache/anyocr/models/)
pub fn get_default_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("ANYOCR_CACHE_DIR") {
        return PathBuf::from(dir);
    }

    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache").join("anyocr").join("models")
    } else {
        PathBuf::from("models").join("ocr")
    }
}

/// 解析指定 ModelProfile 下的模型文件绝对路径
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub det_path: PathBuf,
    pub rec_path: PathBuf,
    pub dict_path: PathBuf,
    pub table_path: Option<PathBuf>,
}

impl ModelPaths {
    /// 依据 ModelProfile 与全局/本地探测，解析模型路径
    pub fn resolve(profile: &ModelProfile) -> Result<Self, AnyOcrError> {
        match profile {
            ModelProfile::Custom {
                det_path,
                rec_path,
                table_path,
                dict_path,
            } => {
                let dict = dict_path.clone().unwrap_or_else(|| {
                    rec_path
                        .parent()
                        .map(|p| p.join(DICT_FILENAME))
                        .unwrap_or_else(|| PathBuf::from(DICT_FILENAME))
                });
                Ok(Self {
                    det_path: det_path.clone(),
                    rec_path: rec_path.clone(),
                    dict_path: dict,
                    table_path: table_path.clone(),
                })
            }
            ModelProfile::Standard | ModelProfile::Fast | ModelProfile::Accurate => {
                let cache_dir = get_default_cache_dir();
                let local_dir = PathBuf::from("models").join("ocr");

                // 优先顺序：本地工作目录 models/ocr ➔ 系统/环境变量缓存目录 (ANYOCR_CACHE_DIR 或 ~/.cache/anyocr/models)
                let search_dirs = [local_dir, cache_dir];

                let rec_filename = match profile {
                    ModelProfile::Fast => REC_MOBILE_FILENAME,
                    ModelProfile::Standard => REC_MEDIUM_FILENAME,
                    ModelProfile::Accurate => REC_SERVER_FILENAME,
                    _ => REC_MEDIUM_FILENAME,
                };

                for dir in &search_dirs {
                    let det = dir.join(DET_SMALL_FILENAME);
                    let mut rec = dir.join(rec_filename);
                    if !rec.exists() {
                        if profile == &ModelProfile::Fast && dir.join(REC_MEDIUM_FILENAME).exists() {
                            rec = dir.join(REC_MEDIUM_FILENAME);
                        } else if profile == &ModelProfile::Standard && dir.join(REC_MOBILE_FILENAME).exists() {
                            rec = dir.join(REC_MOBILE_FILENAME);
                        }
                    }
                    let dict = dir.join(DICT_FILENAME);
                    let table = dir.join(TABLE_FILENAME);

                    if det.exists() && rec.exists() && dict.exists() {
                        return Ok(Self {
                            det_path: det,
                            rec_path: rec,
                            dict_path: dict,
                            table_path: if table.exists() { Some(table) } else { None },
                        });
                    }
                }

                // 若未在本地找到，抛出待就绪提示
                Err(AnyOcrError::ModelNotReady(format!(
                    "未在搜索路径中检测到有效 OCR 模型套件。请配置 ANYOCR_CACHE_DIR 或调用下载器拉取模型。搜索目录: {:?}",
                    search_dirs
                )))
            }
        }
    }
}

/// 模型按需拉取器 (支持从 Release 或 ModelScope 镜像拉取)
pub struct ModelDownloader;

impl ModelDownloader {
    /// 确保指定 ModelProfile 的模型文件在本地就绪
    pub fn ensure_models(profile: &ModelProfile) -> Result<ModelPaths, AnyOcrError> {
        // 先尝试解析已有路径
        if let Ok(paths) = ModelPaths::resolve(profile) {
            return Ok(paths);
        }

        #[cfg(feature = "download")]
        {
            // 若开启了 download 特性，可在此执行自动网络拉取逻辑
            tracing::info!("正在检查并按需拉取轻量 OCR 模型组件...");
            // 真实下载逻辑在后续接入 Release URL
            ModelPaths::resolve(profile)
        }

        #[cfg(not(feature = "download"))]
        {
            Err(AnyOcrError::ModelNotReady(
                "未启用 'download' feature 且本地未找到模型文件，无法自动下载。".to_string(),
            ))
        }
    }
}
