#评估MSE,PSNR,SSIM,ED,MED,MAE,UIQE,UIQM
import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from collections import Counter
import pprint

##输入文件路径
##---------------------------------------------------------------------------------------------
FILE_PATH = "../result_img"
##---------------------------------------------------------------------------------------------

# --- START: Added functions for UIQE and UIQM ---

# --- UCIQE (Underwater Color Image Quality Evaluation) Calculation ---
def calculate_uciqe(img_bgr):
    """
    Calculates the Underwater Color Image Quality Evaluation (UCIQE) metric.
    """
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    chroma = np.sqrt(np.square(a_channel.astype(np.float64)) + np.square(b_channel.astype(np.float64)))
    sigma_c = np.std(chroma)
    
    max_lum = np.max(l_channel)
    min_lum = np.min(l_channel)
    contrast_lum = (max_lum - min_lum) / (max_lum + min_lum + 1e-6)
    
    mean_sat = np.mean(chroma)
    
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    
    print(f"sigma_c:{sigma_c},contrast_lum:{contrast_lum},mean_sat:{mean_sat}")

    uciqe_score = c1 * sigma_c + c2 * contrast_lum + c3 * mean_sat
    return uciqe_score

# --- UIQM (Underwater Image Quality Measure) Calculation ---
def _calculate_ucm(img_rgb):
    """ Helper function to calculate UCM component """
    rg = img_rgb[:, :, 0].astype(np.float64) - img_rgb[:, :, 1].astype(np.float64)
    yb = 0.5 * (img_rgb[:, :, 0].astype(np.float64) + img_rgb[:, :, 1].astype(np.float64)) - img_rgb[:, :, 2].astype(np.float64)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    ucm = -0.0268 * np.sqrt(mean_rg**2 + mean_yb**2) + 0.1586 * np.sqrt(std_rg**2 + std_yb**2)
    return ucm

def _calculate_usm(img_gray):
    """ Helper function to calculate USM component """
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_map = np.sqrt(sobel_x**2 + sobel_y**2)
    
    h, w = img_gray.shape
    block_size = 8
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    
    eme = 0.0
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = gradient_map[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            max_val = np.max(block)
            min_val = np.min(block)
            if min_val == 0 or max_val == 0: continue
            ratio = max_val / min_val
            if ratio == 0: continue
            eme += 20 * np.log(ratio)
            
    usm = eme / (num_blocks_h * num_blocks_w + 1e-6)
    return usm

def _calculate_uconm(img_gray):
    """ Helper function to calculate UConM component """
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-6))
    return entropy

def calculate_uiqm(img_bgr):
    """
    Calculates the Underwater Image Quality Measure (UIQM) metric.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    ucm = _calculate_ucm(img_rgb)
    usm = _calculate_usm(img_gray)
    uconm = _calculate_uconm(img_gray)
    
    print(f"ucm:{ucm},usm:{usm},uconm:{uconm}")

    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753
    
    uiqm_score = c1 * ucm + c2 * usm + c3 * uconm
    return uiqm_score

# --- END: Added functions for UIQE and UIQM ---


def mae(img1,img2):
    mae = np.mean(abs(img1 - img2), dtype=np.float64)
    return mae

def calculate_metrics(image_path1,image_path2):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    if img1 is None:
        raise FileNotFoundError(f"无法加载图像: {image_path1}")
    if img2 is None:
        raise FileNotFoundError(f"无法加载图像: {image_path2}")

    if img1.shape != img2.shape:
        raise ValueError("图片尺寸不同")
    
    # --- Reference-based metrics ---
    mse_value = mse(img1,img2)
    psnr_value = psnr(img1,img2,data_range=255)
    ssim_value = ssim(img1,img2,data_range=255,multichannel=True,channel_axis=2)

    img1_float = img1.astype(np.float64)
    img2_float = img2.astype(np.float64)

    diff = img1_float - img2_float

    ed_value = np.sqrt(np.sum(diff ** 2))
    med_value = np.mean(np.abs(diff))
    mae_value = mae(img1,img2)

    # --- Non-reference metrics (calculated on the first image) ---
    uciqe_value = calculate_uciqe(img1)
    uiqm_value = calculate_uiqm(img1)

    return {
        "MSE": mse_value,
        "PSNR": psnr_value,
        "SSIM": ssim_value,
        "ED": ed_value,
        "MED": med_value,
        "MAE": mae_value,
        "UCIQE": uciqe_value, # Added metric
        "UIQM": uiqm_value    # Added metric
    }

if __name__ == "__main__":
    result_counter = Counter()

    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"找不到图像存储路径 {FILE_PATH}")
        
    # Get the number of relevant files to correctly calculate the total
    all_files = os.listdir(FILE_PATH)
    # Assuming files are always in triplets (_hls_img, _img, etc.)
    num_triplets = len([f for f in all_files if f.endswith('hls_img.jpg')]) 
    print(f"找到 {num_triplets} 组图像进行评估。")

    if num_triplets == 0:
        print("在指定路径中未找到符合命名规则的图像 (hls_img.jpg)。")
    else:
        for i in range(num_triplets):
            # Assuming you want to compare the enhanced image with a ground truth/original
            enhanced_img_path = f"{FILE_PATH}/{i}_hls_img.jpg"
            original_img_path = f"{FILE_PATH}/{i}_img.jpg"    
            
            print(f"\n--- 正在处理第 {i} 组图像 ---")
            print(f"增强图: {enhanced_img_path}")
            print(f"原图/参考图: {original_img_path}")
            
            try:
                result = calculate_metrics(enhanced_img_path, original_img_path)
                result_counter += Counter(result)
                pprint.pprint(result)
            except (FileNotFoundError, ValueError) as e:
                print(f"处理第 {i} 组图像时出错: {e}")
                continue # Skip to the next pair if there is an error
        
        # Calculate and print the final average results
        final_dict = {key: value / num_triplets for key, value in dict(result_counter).items()}
        print("\n\n--- 所有图像的平均指标 ---")
        pprint.pprint(final_dict)