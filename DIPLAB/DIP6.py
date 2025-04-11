import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.util

# Function to calculate PSNR and MSE
def calculate_metrics(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse != 0 else float('inf')
    return mse, psnr

# Load the image
image = cv2.imread(r"D:\DIPLAB\FT1.jpg", cv2.IMREAD_GRAYSCALE)  # Use grayscale for better noise handling

# Add Gaussian Noise
gaussian_noise = skimage.util.random_noise(image, mode='gaussian', var=0.01)
gaussian_noise = (gaussian_noise * 255).astype(np.uint8)

# Add Salt & Pepper Noise
sp_noise = skimage.util.random_noise(image, mode='s&p', amount=0.02)
sp_noise = (sp_noise * 255).astype(np.uint8)

# Apply Filters
mean_filtered = cv2.blur(gaussian_noise, (5,5))  # Mean filter
median_filtered = cv2.medianBlur(sp_noise, 5)  # Median filter
gaussian_filtered = cv2.GaussianBlur(gaussian_noise, (5,5), 0)  # Gaussian filter
bilateral_filtered = cv2.bilateralFilter(gaussian_noise, 9, 75, 75)  # Bilateral filter

# Compute Metrics
metrics = {
    "Mean Filter": calculate_metrics(image, mean_filtered),
    "Median Filter": calculate_metrics(image, median_filtered),
    "Gaussian Filter": calculate_metrics(image, gaussian_filtered),
    "Bilateral Filter": calculate_metrics(image, bilateral_filtered)
}

# Display images
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes[0, 0].imshow(image, cmap='gray'); axes[0, 0].set_title("Original")
axes[0, 1].imshow(gaussian_noise, cmap='gray'); axes[0, 1].set_title("Gaussian Noise")
axes[0, 2].imshow(sp_noise, cmap='gray'); axes[0, 2].set_title("Salt & Pepper Noise")
axes[1, 0].imshow(mean_filtered, cmap='gray'); axes[1, 0].set_title("Mean Filter")
axes[1, 1].imshow(median_filtered, cmap='gray'); axes[1, 1].set_title("Median Filter")
axes[1, 2].imshow(gaussian_filtered, cmap='gray'); axes[1, 2].set_title("Gaussian Filter")
axes[2, 0].imshow(bilateral_filtered, cmap='gray'); axes[2, 0].set_title("Bilateral Filter")

# Hide extra subplots
axes[2,1].axis('off')
axes[2,2].axis('off')

plt.tight_layout()
plt.show()

# Print Performance Metrics
for filter_name, (mse, psnr) in metrics.items():
    print(f"{filter_name}: MSE = {mse:.2f}, PSNR = {psnr:.2f} dB")