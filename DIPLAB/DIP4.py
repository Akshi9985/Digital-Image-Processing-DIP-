import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread("D:\\DIPLAB\\FT1.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not found. Check the file path!")

# Compute DFT (Fourier Transform) and shift zero frequency to the center
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)

# Get image dimensions
rows, cols = image.shape
crow, ccol = rows // 2 , cols // 2

# Create different filters
def create_filter(filter_type, cutoff, order=2):
    """Creates low-pass or high-pass filters."""
    mask = np.zeros((rows, cols), np.float32)
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)  # Distance from center

            if filter_type == "lowpass":
                mask[u, v] = 1 if D <= cutoff else 0  # Ideal LPF
            elif filter_type == "highpass":
                mask[u, v] = 0 if D <= cutoff else 1  # Ideal HPF
            elif filter_type == "butterworth_lowpass":
                mask[u, v] = 1 / (1 + (D / cutoff) ** (2 * order))  # BLPF
            elif filter_type == "butterworth_highpass":
                mask[u, v] = 1 - (1 / (1 + (D / cutoff) ** (2 * order)))  # BHPF
            elif filter_type == "gaussian_lowpass":
                mask[u, v] = np.exp(-(D**2) / (2 * (cutoff**2)))  # GLPF
            elif filter_type == "gaussian_highpass":
                mask[u, v] = 1 - np.exp(-(D**2) / (2 * (cutoff**2)))  # GHPF
    
    return mask

# Apply filters
cutoff = 50  # Adjust cutoff frequency
order = 2    # Butterworth order

lpf_ideal = create_filter("lowpass", cutoff)
hpf_ideal = create_filter("highpass", cutoff)
lpf_gaussian = create_filter("gaussian_lowpass", cutoff)
hpf_gaussian = create_filter("gaussian_highpass", cutoff)
lpf_butterworth = create_filter("butterworth_lowpass", cutoff, order)
hpf_butterworth = create_filter("butterworth_highpass", cutoff, order)

# Function to apply filter in frequency domain
def apply_filter(dft_shift, filter_mask):
    dft_filtered = dft_shift * filter_mask
    dft_inverse = np.fft.ifftshift(dft_filtered)  # Shift back
    img_filtered = np.fft.ifft2(dft_inverse)  # Inverse DFT
    return np.abs(img_filtered)  # Take magnitude

# Apply filters
img_lpf_ideal = apply_filter(dft_shift, lpf_ideal)
img_hpf_ideal = apply_filter(dft_shift, hpf_ideal)
img_lpf_gaussian = apply_filter(dft_shift, lpf_gaussian)
img_hpf_gaussian = apply_filter(dft_shift, hpf_gaussian)
img_lpf_butterworth = apply_filter(dft_shift, lpf_butterworth)
img_hpf_butterworth = apply_filter(dft_shift, hpf_butterworth)

# Display results
titles = ["Original", "Ideal LPF", "Gaussian LPF", "Butterworth LPF",
          "Ideal HPF", "Gaussian HPF", "Butterworth HPF"]
images = [image, img_lpf_ideal, img_lpf_gaussian, img_lpf_butterworth,
          img_hpf_ideal, img_hpf_gaussian, img_hpf_butterworth]

plt.figure(figsize=(12, 8))
for i in range(7):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()