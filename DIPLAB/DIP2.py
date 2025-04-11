import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a grayscale image
image = cv2.imread("DIPLAB/FT1.jpg", cv2.IMREAD_GRAYSCALE)

# Compute the 2D Fourier Transform
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)  # Shift the zero frequency component to the center

# Compute magnitude spectrum
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# Perform the Inverse DFT
dft_ishift = np.fft.ifftshift(dft_shift)  # Shift back
img_reconstructed = cv2.idft(dft_ishift)
img_reconstructed = cv2.magnitude(img_reconstructed[:, :, 0], img_reconstructed[:, :, 1])  # Convert to magnitude

# Normalize the reconstructed image
cv2.normalize(img_reconstructed, img_reconstructed, 0, 255, cv2.NORM_MINMAX)
img_reconstructed = np.uint8(img_reconstructed)

# Display all images in one figure
plt.figure(figsize=(15,5))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title("Original Image")
plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title("Magnitude Spectrum")
plt.subplot(133), plt.imshow(img_reconstructed, cmap='gray'), plt.title("Reconstructed Image")
plt.show()
