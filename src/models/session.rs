use crate::error::AnyOcrError;
use crate::types::ExecutionProvider;
use ort::session::builder::SessionBuilder;
use ort::session::Session;
use std::path::Path;

pub fn build_session(
    model_path: impl AsRef<Path>,
    provider: ExecutionProvider,
) -> Result<Session, AnyOcrError> {
    let path_ref = model_path.as_ref();
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(2, 16);

    let mut builder = Session::builder()
        .map_err(|e| AnyOcrError::InferenceError(format!("创建 ONNX SessionBuilder 失败: {e}")))?
        .with_intra_threads(num_threads)
        .map_err(|e| AnyOcrError::InferenceError(format!("配置推理线程失败: {e}")))?;

    // 硬件加速提供者装配
    builder = apply_execution_provider(builder, provider)?;

    let session = builder
        .commit_from_file(path_ref)
        .map_err(|e| AnyOcrError::InferenceError(format!("加载模型 {} 失败: {e}", path_ref.display())))?;

    Ok(session)
}

fn apply_execution_provider(
    builder: SessionBuilder,
    provider: ExecutionProvider,
) -> Result<SessionBuilder, AnyOcrError> {
    match provider {
        ExecutionProvider::Cpu => {
            tracing::info!("使用纯 CPU SIMD 推理模式");
            Ok(builder)
        }
        ExecutionProvider::Auto => {
            #[cfg(target_os = "macos")]
            {
                tracing::info!("Auto 模式：探测到 macOS 平台，尝试启用 Apple CoreML / Neural Engine 硬件加速");
                let coreml_ep = ort::ep::CoreML::default().build();
                match builder.with_execution_providers([coreml_ep]) {
                    Ok(b) => return Ok(b),
                    Err(e) => {
                        tracing::warn!("启用 CoreML 失败，平滑回退到多核 CPU 推理: {e}");
                        let num_threads = std::thread::available_parallelism()
                            .map(|n| n.get())
                            .unwrap_or(4)
                            .clamp(2, 16);
                        return Session::builder()
                            .map_err(|err| AnyOcrError::InferenceError(format!("创建回退 SessionBuilder 失败: {err}")))?
                            .with_intra_threads(num_threads)
                            .map_err(|err| AnyOcrError::InferenceError(format!("配置回退线程失败: {err}")));
                    }
                }
            }

            #[cfg(target_os = "windows")]
            {
                tracing::info!("Auto 模式：探测到 Windows 平台，尝试启用 DirectML GPU 硬件加速");
                let directml_ep = ort::ep::DirectML::default().build();
                match builder.with_execution_providers([directml_ep]) {
                    Ok(b) => return Ok(b),
                    Err(e) => {
                        tracing::warn!("启用 DirectML 失败，平滑回退到多核 CPU 推理: {e}");
                        let num_threads = std::thread::available_parallelism()
                            .map(|n| n.get())
                            .unwrap_or(4)
                            .clamp(2, 16);
                        return Session::builder()
                            .map_err(|err| AnyOcrError::InferenceError(format!("创建回退 SessionBuilder 失败: {err}")))?
                            .with_intra_threads(num_threads)
                            .map_err(|err| AnyOcrError::InferenceError(format!("配置回退线程失败: {err}")));
                    }
                }
            }

            #[allow(unreachable_code)]
            {
                tracing::info!("Auto 模式：使用高并发多核 CPU 推理");
                Ok(builder)
            }
        }
        ExecutionProvider::CoreML => {
            #[cfg(target_os = "macos")]
            {
                let coreml_ep = ort::ep::CoreML::default().build();
                builder.with_execution_providers([coreml_ep])
                    .map_err(|e| AnyOcrError::InferenceError(format!("配置 CoreML 执行提供者失败: {e}")))
            }
            #[cfg(not(target_os = "macos"))]
            {
                Err(AnyOcrError::InferenceError("当前平台非 macOS，不支持 CoreML".to_string()))
            }
        }
        ExecutionProvider::DirectML(_device_id) => {
            #[cfg(target_os = "windows")]
            {
                let directml_ep = ort::ep::DirectML::default().with_device_id(_device_id).build();
                builder.with_execution_providers([directml_ep])
                    .map_err(|e| AnyOcrError::InferenceError(format!("配置 DirectML 执行提供者失败: {e}")))
            }
            #[cfg(not(target_os = "windows"))]
            {
                Err(AnyOcrError::InferenceError("当前平台非 Windows，不支持 DirectML".to_string()))
            }
        }
        ExecutionProvider::Cuda(_device_id) => {
            Err(AnyOcrError::InferenceError("未编译 CUDA 支持".to_string()))
        }
    }
}
