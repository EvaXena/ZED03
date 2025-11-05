#评估MSE,PSNR,SSIM,ED,MED
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


def mae(img1,img2):
    mae = np.mean(abs(img1 - img2), dtype=np.float64)
    return mae

def calculate_metrics(image_path1,image_path2):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    if img1.shape != img2.shape:
        raise ValueError("图片尺寸不同")
    
    mse_value = mse(img1,img2)
    psnr_value = psnr(img1,img2,data_range=255)
    ssim_value = ssim(img1,img2,data_range=255,multichannel=True,channel_axis=2)

    img1_float = img1.astype(np.float64)
    img2_float = img2.astype(np.float64)

    diff = img1_float - img2_float

    ed_value = np.sqrt(np.sum(diff ** 2))

    med_value = np.mean(np.abs(diff))

    mae_value = mae(img1,img2)
    return {
        "MSE:":mse_value,
        "PSNR:":psnr_value,
        "SSIM:":ssim_value,
        "ED":ed_value,
        "MED":med_value,
        "MAE":mae_value
    }

if __name__ == "__main__":
    result_counter = Counter()

    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"找不到图像存储路径 {FILE_PATH}")
    for root,dirs,files in os.walk(FILE_PATH):
        files_num = len(files)
        print(files_num)

    for i in range(int(files_num/3)):
        img1 = f"{FILE_PATH}/{i}_hls_img.jpg"
        img2 = f"{FILE_PATH}/{i}_img.jpg"    
        result = calculate_metrics(img1,img2)
        result_counter += Counter(result)
        pprint.pprint(result)
        pprint.pprint(result_counter)
    final_dict = {key:value / (files_num/3) for key,value in dict(result_counter).items()}
    pprint.pprint(final_dict)
    



        