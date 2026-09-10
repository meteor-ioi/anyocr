use crate::error::AnyOcrError;
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;

/// 图像缩放信息，记录原图到缩放图像的映射比例，便于坐标逆变换回原图物理像素
#[derive(Debug, Clone, Copy)]
pub struct ResizeInfo {
    pub original_width: u32,
    pub original_height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub pad_left: u32,
    pub pad_top: u32,
}

/// 图像摄入与预处理管道
pub struct ImagePreprocessor;

impl ImagePreprocessor {
    /// 从内存字节流解码图像 (支持 PNG, JPEG, BMP, WEBP, TIFF)
    pub fn decode_image(bytes: &[u8]) -> Result<DynamicImage, AnyOcrError> {
        image::load_from_memory(bytes).map_err(AnyOcrError::from)
    }

    /// 针对文本检测模型 (DBNet) 的尺寸归一化预处理
    ///
    /// - 保持宽高比将最长边约束在 max_side_len 内 (默认 960)
    /// - 保证输出宽高均为 32 的整数倍 (满足 DBNet 多层下采样步长要求)
    /// - 输出 NCHW (1, 3, H, W) 归一化浮点 Tensor，均值方差采用标准 ImageNet 参数
    pub fn prepare_det_input(
        img: &DynamicImage,
        max_side_len: u32,
    ) -> Result<(Array4<f32>, ResizeInfo), AnyOcrError> {
        let (orig_w, orig_h) = img.dimensions();
        if orig_w == 0 || orig_h == 0 {
            return Err(AnyOcrError::Other("输入图像尺寸为 0".to_string()));
        }

        // 计算等比缩放比率
        let max_dim = orig_w.max(orig_h);
        let ratio = if max_dim > max_side_len {
            max_side_len as f32 / max_dim as f32
        } else {
            1.0
        };

        // 宽高对齐至 32 的倍数
        let mut target_w = (orig_w as f32 * ratio) as u32;
        let mut target_h = (orig_h as f32 * ratio) as u32;

        target_w = (target_w / 32).max(1) * 32;
        target_h = (target_h / 32).max(1) * 32;

        let resized = img.resize_exact(target_w, target_h, image::imageops::FilterType::Triangle);
        let rgb_img = resized.to_rgb8();

        let scale_x = target_w as f32 / orig_w as f32;
        let scale_y = target_h as f32 / orig_h as f32;

        let resize_info = ResizeInfo {
            original_width: orig_w,
            original_height: orig_h,
            resized_width: target_w,
            resized_height: target_h,
            scale_x,
            scale_y,
            pad_left: 0,
            pad_top: 0,
        };

        // PP-OCR 标准检测归一化参数
        let mean = [0.485f32, 0.456f32, 0.406f32];
        let std = [0.229f32, 0.224f32, 0.225f32];

        let mut tensor = Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));

        for y in 0..target_h {
            for x in 0..target_w {
                let pixel = rgb_img.get_pixel(x, y);
                for c in 0..3 {
                    let val = pixel[c] as f32 / 255.0;
                    tensor[[0, c, y as usize, x as usize]] = (val - mean[c]) / std[c];
                }
            }
        }

        Ok((tensor, resize_info))
    }

    /// 针对文本识别模型 (SVTR / CRNN) 的单行切片预处理
    ///
    /// - 固定高度为 target_height (PP-OCRv6 推荐 48px)
    /// - 宽度按比例缩放，并保证最小宽度为 16px
    /// - 归一化采用标准公式: (pixel / 255.0 - 0.5) / 0.5
    pub fn prepare_rec_input(
        crop: &DynamicImage,
        target_height: u32,
    ) -> Result<Array4<f32>, AnyOcrError> {
        let (w, h) = crop.dimensions();
        if w == 0 || h == 0 {
            return Err(AnyOcrError::Other("文本行切片尺寸无效".to_string()));
        }

        let ratio = target_height as f32 / h as f32;
        let target_w = ((w as f32 * ratio) as u32).max(16);

        let resized = crop.resize_exact(target_w, target_height, image::imageops::FilterType::Triangle);
        let rgb_img = resized.to_rgb8();

        let mut tensor = Array4::<f32>::zeros((1, 3, target_height as usize, target_w as usize));

        for y in 0..target_height {
            for x in 0..target_w {
                let pixel = rgb_img.get_pixel(x, y);
                for c in 0..3 {
                    let val = pixel[c] as f32 / 255.0;
                    tensor[[0, c, y as usize, x as usize]] = (val - 0.5) / 0.5;
                }
            }
        }

        Ok(tensor)
    }

    /// 针对表格结构预测模型 (SLANet_plus) 的等比例缩放与填充预处理
    ///
    /// - 保持宽高比将最长边缩放至 488px
    /// - 居左上放置在 488x488 的固定输入张量中，空白区域补 0
    pub fn prepare_table_input(
        img: &DynamicImage,
    ) -> Result<(Array4<f32>, ResizeInfo), AnyOcrError> {
        let (orig_w, orig_h) = img.dimensions();
        if orig_w == 0 || orig_h == 0 {
            return Err(AnyOcrError::Other("输入图像尺寸为 0".to_string()));
        }

        let max_dim = orig_w.max(orig_h) as f32;
        let ratio = 488.0 / max_dim;
        let resize_w = ((orig_w as f32 * ratio).round() as u32).max(1);
        let resize_h = ((orig_h as f32 * ratio).round() as u32).max(1);

        let resized = img.resize_exact(resize_w, resize_h, image::imageops::FilterType::Triangle);
        let rgb_img = resized.to_rgb8();

        let resize_info = ResizeInfo {
            original_width: orig_w,
            original_height: orig_h,
            resized_width: resize_w,
            resized_height: resize_h,
            scale_x: ratio,
            scale_y: ratio,
            pad_left: 0,
            pad_top: 0,
        };

        let mean = [0.485f32, 0.456f32, 0.406f32];
        let std = [0.229f32, 0.224f32, 0.225f32];

        let mut tensor = Array4::<f32>::zeros((1, 3, 488, 488));

        for y in 0..resize_h {
            for x in 0..resize_w {
                let pixel = rgb_img.get_pixel(x, y);
                for c in 0..3 {
                    let val = pixel[c] as f32 / 255.0;
                    tensor[[0, c, y as usize, x as usize]] = (val - mean[c]) / std[c];
                }
            }
        }

        Ok((tensor, resize_info))
    }

    /// 依据坐标框裁剪原图像中的单行文本切片
    pub fn crop_box(img: &DynamicImage, box_coords: &[f32; 4]) -> DynamicImage {
        let (img_w, img_h) = img.dimensions();
        let x1 = (box_coords[0].max(0.0) as u32).min(img_w);
        let y1 = (box_coords[1].max(0.0) as u32).min(img_h);
        let x2 = (box_coords[2].max(0.0) as u32).min(img_w);
        let y2 = (box_coords[3].max(0.0) as u32).min(img_h);

        let w = x2.saturating_sub(x1).max(1);
        let h = y2.saturating_sub(y1).max(1);

        img.crop_imm(x1, y1, w, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn test_det_input_align_to_32() {
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(250, 130, Rgb([100, 100, 100])));
        let (tensor, info) = ImagePreprocessor::prepare_det_input(&img, 960).unwrap();

        assert_eq!(info.original_width, 250);
        assert_eq!(info.original_height, 130);
        assert_eq!(info.resized_width % 32, 0);
        assert_eq!(info.resized_height % 32, 0);
        assert_eq!(tensor.shape(), &[1, 3, info.resized_height as usize, info.resized_width as usize]);
    }

    #[test]
    fn test_rec_input_fixed_height() {
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(120, 30, Rgb([50, 50, 50])));
        let tensor = ImagePreprocessor::prepare_rec_input(&img, 48).unwrap();
        assert_eq!(tensor.shape()[2], 48);
        assert!(tensor.shape()[3] >= 16);
    }

    #[test]
    fn test_table_input_fixed_488() {
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(300, 400, Rgb([200, 200, 200])));
        let (tensor, info) = ImagePreprocessor::prepare_table_input(&img).unwrap();

        assert_eq!(info.resized_height, 488);
        assert_eq!(tensor.shape(), &[1, 3, 488, 488]);
    }
}
