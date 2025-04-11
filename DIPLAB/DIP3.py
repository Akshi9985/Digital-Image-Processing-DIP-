import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Image (Grayscale)
image = cv2.imread("D:\\DIPLAB\\FT1.jpg", cv2.IMREAD_GRAYSCALE)

# 1. Smoothing Filters
# Mean Filter (Averaging)
mean_filter = cv2.blur(image, (5, 5))

# Gaussian Filter
gaussian_filter = cv2.GaussianBlur(image, (5, 5), 0)

# Median Filter
median_filter = cv2.medianBlur(image, 5)

# 2. Sharpening Filters
# Laplacian Filter
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# Sobel Filter
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)

# Unsharp Masking
gaussian_blur = cv2.GaussianBlur(image, (5,5), 0)
unsharp_masking = cv2.addWeighted(image, 1.5, gaussian_blur, -0.5, 0)

# Display Results
titles = ['Original', 'Mean Filter', 'Gaussian Filter', 'Median Filter', 
          'Laplacian', 'Sobel', 'Unsharp Masking']
images = [image, mean_filter, gaussian_filter, median_filter, 
          laplacian, sobel, unsharp_masking]

plt.figure(figsize=(12, 8))
for i in range(7):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

