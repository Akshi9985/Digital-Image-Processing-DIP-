import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_detection(image_path):
    # Load image in grayscale
    img = cv2.imread(r"D:\DIPLAB\FT1.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Apply Sobel operator
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    
    # Apply Prewitt operator
    prewitt_x = cv2.filter2D(img, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    prewitt_y = cv2.filter2D(img, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
    prewitt = cv2.magnitude(prewitt_x.astype(np.float64), prewitt_y.astype(np.float64))
    
    # Apply Canny edge detection
    canny = cv2.Canny(img, 100, 200)
    
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    
    # Display results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(232), plt.imshow(sobel, cmap='gray'), plt.title('Sobel Edge Detection')
    plt.subplot(233), plt.imshow(prewitt, cmap='gray'), plt.title('Prewitt Edge Detection')
    plt.subplot(234), plt.imshow(canny, cmap='gray'), plt.title('Canny Edge Detection')
    plt.subplot(235), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian Edge Detection')
    
    plt.tight_layout()
    plt.show()

# Example usage
# Provide a valid image path
edge_detection(r"D:\DIPLAB\FT1.jpg")




