import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

"""
Evluated results:
Dataset: /home/sixuan-wu/research/MicroClear/data_splits/test_rbc.txt
Average SSIM: 0.9384 ± 0.0218, Average PSNR: 28.60 ± 2.51 dB
Dataset: /home/sixuan-wu/research/MicroClear/data_splits/test_hist.txt
Average SSIM: 0.4132 ± 0.2080, Average PSNR: 17.66 ± 5.54 dB
Dataset: /home/sixuan-wu/research/MicroClear/data_splits/test_bbbc006_w2.txt
Average SSIM: 0.5760 ± 0.1442, Average PSNR: 22.83 ± 3.77 dB
Dataset: /home/sixuan-wu/research/MicroClear/data_splits/test_focus.txt
Average SSIM: 0.4575 ± 0.1489, Average PSNR: 20.99 ± 3.95 dB
Dataset: /home/sixuan-wu/research/MicroClear/data_splits/test_bbbc006_w1.txt
Average SSIM: 0.5784 ± 0.2152, Average PSNR: 20.93 ± 5.59 dB
"""


def calculate_ssim_psnr(img1, img2):
    """
    Compute SSIM and PSNR between two images.    
    Args:
        img1, img2: numpy arrays (H x W x C) or (H x W), same shape, dtype float or uint8.
    Returns:
        ssim_value: float
        psnr_value: float
    """
    # Convert to float64 for precision
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Handle RGB or grayscale automatically
    multichannel = img1.ndim == 3 and img1.shape[2] > 1
    
    ssim_value = ssim(img1, img2, channel_axis=-1 if multichannel else None, data_range=img1.max() - img1.min())
    psnr_value = psnr(img1, img2, data_range=img1.max() - img1.min())
    
    return ssim_value, psnr_value

def main():
    splits_files = [
        '/home/sixuan-wu/research/MicroClear/data_splits/test_bbbc006_w1.txt',
        '/home/sixuan-wu/research/MicroClear/data_splits/test_bbbc006_w2.txt',
        '/home/sixuan-wu/research/MicroClear/data_splits/test_focus.txt',
        '/home/sixuan-wu/research/MicroClear/data_splits/test_hist.txt',
        '/home/sixuan-wu/research/MicroClear/data_splits/test_rbc.txt'
    ]
    blurry_clear_pairs = {}
    filename_dataset_map = {}
    for split_file in splits_files:
        with open(split_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                blurry_path = parts[0]
                clear_path = parts[1]
                blurry_filename = blurry_path.split('/')[-1]
                filename_dataset_map[blurry_filename] = split_file
                blurry_clear_pairs[blurry_filename] = clear_path
    dataset_eval = {}
    for filename in tqdm(os.listdir('microscope_results')):
        if 'kernel' in filename: # Skip kernel images
            continue
        if filename in blurry_clear_pairs:
            deblurred_path = os.path.join('microscope_results', filename)
            clear_path = blurry_clear_pairs[filename]
            deblurred_img = cv2.imread(deblurred_path)
            clear_img = cv2.imread(clear_path)
            if deblurred_img is None or clear_img is None:
                print(f'Error reading images for {filename}')
                continue
            if deblurred_img.shape != clear_img.shape:
                print(f'Skipping {filename} due to shape mismatch: {deblurred_img.shape} vs {clear_img.shape}')
                continue
            ssim_value, psnr_value = calculate_ssim_psnr(deblurred_img, clear_img)
            if filename_dataset_map[filename] not in dataset_eval:
                dataset_eval[filename_dataset_map[filename]] = []
            dataset_eval[filename_dataset_map[filename]].append((ssim_value, psnr_value))
        else:
            print(f'No ground truth found for {filename}, skipping.')
    for dataset, metrics in dataset_eval.items():
        avg_ssim = np.mean([m[0] for m in metrics])
        std_ssim = np.std([m[0] for m in metrics])
        avg_psnr = np.mean([m[1] for m in metrics])
        std_psnr = np.std([m[1] for m in metrics])
        print(f'Dataset: {dataset}')
        print(f'Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}, Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB')

if __name__ == "__main__":
    main()