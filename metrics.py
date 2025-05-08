import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse


def calculate_psnr(img1, img2):
    """
    计算两张图像之间的PSNR值
    
    Args:
        img1: 参考图像
        img2: 测试图像
        
    Returns:
        PSNR值
    """
    return psnr(img1, img2, data_range=255)


def calculate_ssim(img1, img2):
    """
    计算两张图像之间的SSIM值
    
    Args:
        img1: 参考图像
        img2: 测试图像
        
    Returns:
        SSIM值
    """
    return ssim(img1, img2, data_range=255, channel_axis=2 if len(img1.shape) > 2 else None)


def evaluate_images(reference_path, test_path):
    """
    评估参考图像和测试图像之间的PSNR和SSIM
    
    Args:
        reference_path: 参考图像的路径
        test_path: 测试图像的路径
        
    Returns:
        psnr_value: PSNR值
        ssim_value: SSIM值
    """
    # 读取图像
    ref_img = cv2.imread(reference_path)
    test_img = cv2.imread(test_path)
    
    # 检查图像是否读取成功
    if ref_img is None or test_img is None:
        raise ValueError("无法读取图像文件")
    
    # 确保两张图像具有相同的大小
    if ref_img.shape != test_img.shape:
        test_img = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]))
    
    # 将BGR转换为RGB
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    # 计算指标
    psnr_value = calculate_psnr(ref_img, test_img)
    ssim_value = calculate_ssim(ref_img, test_img)
    
    return psnr_value, ssim_value


def evaluate_directory(reference_dir, test_dir):
    """
    评估两个目录中所有对应图像的PSNR和SSIM平均值
    
    Args:
        reference_dir: 参考图像目录路径
        test_dir: 测试图像目录路径
        
    Returns:
        avg_psnr: 平均PSNR值
        avg_ssim: 平均SSIM值
        results: 每对图像的评估结果
    """
    if not os.path.isdir(reference_dir) or not os.path.isdir(test_dir):
        raise ValueError("提供的路径不是有效目录")
    
    # 获取参考目录中的所有图像文件
    ref_files = [f for f in os.listdir(reference_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    results = []
    total_psnr = 0
    total_ssim = 0
    count = 0
    
    for ref_file in ref_files:
        # 查找测试目录中的对应文件
        test_file = ref_file
        
        ref_path = os.path.join(reference_dir, ref_file)
        test_path = os.path.join(test_dir, test_file)
        
        if os.path.exists(test_path):
            try:
                psnr_value, ssim_value = evaluate_images(ref_path, test_path)
                results.append({
                    'reference': ref_file,
                    'test': test_file,
                    'psnr': psnr_value,
                    'ssim': ssim_value
                })
                
                total_psnr += psnr_value
                total_ssim += ssim_value
                count += 1
                
            except Exception as e:
                print(f"处理图像对 {ref_file} 时出错: {str(e)}")
    
    if count == 0:
        return 0, 0, []
    
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    
    return avg_psnr, avg_ssim, results


def main():
    # parser = argparse.ArgumentParser(description='计算图像的PSNR和SSIM指标')
    # parser.add_argument('--ref', type=str, required=True, help='参考图像文件或目录的路径')
    # parser.add_argument('--test', type=str, required=True, help='测试图像文件或目录的路径')
    #
    # args = parser.parse_args()
    
    # ref_path = args.ref
    # test_path = args.test

    ref_path = "dataset-bt-withno/valdata"
    test_path = "../RCS-YOLO/dataset-brain-tumor/dpereduced/valdata"
    result_path = "metric_results_clahe.txt"
    
    # 检查路径是文件还是目录
    if os.path.isfile(ref_path) and os.path.isfile(test_path):
        # 评估单对图像
        psnr_value, ssim_value = evaluate_images(ref_path, test_path)

        # 打印结果
        print(f"PSNR: {psnr_value:.4f} dB")
        print(f"SSIM: {ssim_value:.4f}")
        
        # 将结果写入文件
        with open(result_path, "w") as f:
            f.write(f"PSNR: {psnr_value:.4f} dB\n")
            f.write(f"SSIM: {ssim_value:.4f}\n")
        
        print(f"结果已保存到 {result_path}")
    
    elif os.path.isdir(ref_path) and os.path.isdir(test_path):
        # 评估目录中的所有图像
        avg_psnr, avg_ssim, results = evaluate_directory(ref_path, test_path)

        # 打印结果
        print(f"平均 PSNR: {avg_psnr:.4f} dB")
        print(f"平均 SSIM: {avg_ssim:.4f}")
        
        # 将结果写入文件
        with open(result_path, "w") as f:
            f.write(f"平均 PSNR: {avg_psnr:.4f} dB\n")
            f.write(f"平均 SSIM: {avg_ssim:.4f}\n")
            
            f.write("\n详细结果:\n")
            for result in results:
                f.write(f"图像: {result['reference']}\n")
                f.write(f"  PSNR: {result['psnr']:.4f} dB\n")
                f.write(f"  SSIM: {result['ssim']:.4f}\n")
        
        print(f"结果已保存到 {result_path}")
    
    else:
        print("错误: 提供的路径必须同时为文件或同时为目录")


if __name__ == "__main__":
    main()
