use crate::error::AnyOcrError;
use crate::types::ExecutionProvider;
use ort::session::Session;
use std::path::Path;

/// 依据指定硬件加速提供者配置并加载 ONNX 推理会话 (支持 Auto 探测与安全平滑降级)
pub fn build_session(
    model_path: impl AsRef<Path>,
    provider: ExecutionProvider,
) -> Result<Session, AnyOcrError> {
    let path_ref = model_path.as_ref();
    if !path_ref.exists() {
        return Err(AnyOcrError::ModelNotReady(format!(
            "模型文件不存在: {}",
            path_ref.display()
        )));
    }

    // 探测系统可用逻辑核心数 (限制在 2 ~ 16 之间，避免过多线程导致调度竞争)
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(2, 16);

    let create_cpu_session = || -> Result<Session, AnyOcrError> {
        Session::builder()
            .map_err(|e| AnyOcrError::InferenceError(format!("创建 ONNX SessionBuilder 失败: {e}")))?
            .with_intra_threads(num_threads)
            .map_err(|e| AnyOcrError::InferenceError(format!("配置推理线程失败: {e}")))?
            .commit_from_file(path_ref)
            .map_err(|e| AnyOcrError::InferenceError(format!("加载模型 {} 失败: {e}", path_ref.display())))
    };

    match provider {
        ExecutionProvider::Cpu => {
            tracing::info!("使用纯 CPU SIMD 推理模式 ({} 线程)", num_threads);
            create_cpu_session()
        }
        ExecutionProvider::Auto => {
            #[cfg(all(target_os = "macos", feature = "coreml"))]
            {
                tracing::info!("Auto 模式：macOS 平台尝试 CoreML 硬件加速: {}", path_ref.display());
                let coreml_ep = ort::ep::CoreML::default().build();
                let res: Result<Session, ort::Error> = (|| {
                    Session::builder()?
                        .with_intra_threads(num_threads)?
                        .with_execution_providers([coreml_ep])?
                        .commit_from_file(path_ref)
                })();
                match res {
                    Ok(sess) => {
                        tracing::info!("成功启用 CoreML 硬件加速: {}", path_ref.display());
                        return Ok(sess);
                    }
                    Err(e) => {
                        tracing::warn!("CoreML 加载模型失败 ({e})，平滑降级至多核 CPU 推理: {}", path_ref.display());
                    }
                }
            }

            #[cfg(all(target_os = "windows", feature = "directml"))]
            {
                tracing::info!("Auto 模式：Windows 平台尝试 DirectML 硬件加速: {}", path_ref.display());
                let directml_ep = ort::ep::DirectML::default().build();
                let res: Result<Session, ort::Error> = (|| {
                    Session::builder()?
                        .with_intra_threads(num_threads)?
                        .with_execution_providers([directml_ep])?
                        .commit_from_file(path_ref)
                })();
                match res {
                    Ok(sess) => {
                        tracing::info!("成功启用 DirectML 硬件加速: {}", path_ref.display());
                        return Ok(sess);
                    }
                    Err(e) => {
                        tracing::warn!("DirectML 加载模型失败 ({e})，平滑降级至多核 CPU 推理: {}", path_ref.display());
                    }
                }
            }

            create_cpu_session()
        }
        ExecutionProvider::CoreML => {
            #[cfg(feature = "coreml")]
            {
                #[cfg(target_os = "macos")]
                {
                    let coreml_ep = ort::ep::CoreML::default().build();
                    Session::builder()
                        .map_err(|e| AnyOcrError::InferenceError(format!("创建 ONNX SessionBuilder 失败: {e}")))?
                        .with_intra_threads(num_threads)
                        .map_err(|e| AnyOcrError::InferenceError(format!("配置推理线程失败: {e}")))?
                        .with_execution_providers([coreml_ep])
                        .map_err(|e| AnyOcrError::InferenceError(format!("配置 CoreML 失败: {e}")))?
                        .commit_from_file(path_ref)
                        .map_err(|e| AnyOcrError::InferenceError(format!("加载 CoreML 模型失败: {e}")))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Err(AnyOcrError::InferenceError("当前平台非 macOS，不支持 CoreML".to_string()))
                }
            }
            #[cfg(not(feature = "coreml"))]
            {
                Err(AnyOcrError::InferenceError("未开启 'coreml' feature，请在 Cargo.toml 中启用对应特性".to_string()))
            }
        }
        ExecutionProvider::DirectML(_device_id) => {
            #[cfg(feature = "directml")]
            {
                #[cfg(target_os = "windows")]
                {
                    let directml_ep = ort::ep::DirectML::default().with_device_id(_device_id).build();
                    Session::builder()
                        .map_err(|e| AnyOcrError::InferenceError(format!("创建 ONNX SessionBuilder 失败: {e}")))?
                        .with_intra_threads(num_threads)
                        .map_err(|e| AnyOcrError::InferenceError(format!("配置推理线程失败: {e}")))?
                        .with_execution_providers([directml_ep])
                        .map_err(|e| AnyOcrError::InferenceError(format!("配置 DirectML 失败: {e}")))?
                        .commit_from_file(path_ref)
                        .map_err(|e| AnyOcrError::InferenceError(format!("加载 DirectML 模型失败: {e}")))
                }
                #[cfg(not(target_os = "windows"))]
                {
                    Err(AnyOcrError::InferenceError("当前平台非 Windows，不支持 DirectML".to_string()))
                }
            }
            #[cfg(not(feature = "directml"))]
            {
                Err(AnyOcrError::InferenceError("未开启 'directml' feature，请在 Cargo.toml 中启用对应特性".to_string()))
            }
        }
        ExecutionProvider::Cuda(_device_id) => {
            #[cfg(feature = "cuda")]
            {
                let cuda_ep = ort::ep::CUDA::default().with_device_id(_device_id).build();
                Session::builder()
                    .map_err(|e| AnyOcrError::InferenceError(format!("创建 ONNX SessionBuilder 失败: {e}")))?
                    .with_intra_threads(num_threads)
                    .map_err(|e| AnyOcrError::InferenceError(format!("配置推理线程失败: {e}")))?
                    .with_execution_providers([cuda_ep])
                    .map_err(|e| AnyOcrError::InferenceError(format!("配置 CUDA 失败: {e}")))?
                    .commit_from_file(path_ref)
                    .map_err(|e| AnyOcrError::InferenceError(format!("加载 CUDA 模型失败: {e}")))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(AnyOcrError::InferenceError("未开启 'cuda' feature，请在 Cargo.toml 中启用对应特性".to_string()))
            }
        }
    }
}
