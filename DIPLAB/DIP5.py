import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = r"D:\DIPLAB\FT1.jpg"
image = cv2.imread(image_path)


# Convert to grayscale and apply histogram equalization
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized_gray = cv2.equalizeHist(gray)

# Apply histogram equalization to each channel in a color image
def equalize_hist_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  # Convert to YUV color space
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # Equalize only the luminance channel
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)  # Convert back to BGR

equalized_color = equalize_hist_color(image)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(1, 3, 2), plt.imshow(equalized_gray, cmap='gray'), plt.title('Equalized Grayscale')
plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(equalized_color, cv2.COLOR_BGR2RGB)), plt.title('Equalized Color')
plt.show()

# Save the output images
cv2.imwrite("equalized_gray.jpg", equalized_gray)
cv2.imwrite("equalized_color.jpg", equalized_color)