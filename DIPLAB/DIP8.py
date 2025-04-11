import cv2
import numpy as np
import matplotlib.pyplot as plt

def fourier_analysis(image_path):
    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Unable to load image. Check the file path.")
        return

    # Resize for efficiency
    img = cv2.resize(img, (256, 256))

    # Compute Fourier Transform of the original image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log1p(np.abs(fshift))

    # Rotate image by 45 degrees
    rows, cols = img.shape
    center = (cols // 2, rows // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (cols, rows))

    # Fourier Transform of rotated image
    f_rot = np.fft.fft2(rotated_img)
    fshift_rot = np.fft.fftshift(f_rot)
    magnitude_spectrum_rot = np.log1p(np.abs(fshift_rot))

    # Create a Gaussian filter
    kernel_size = 21
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, 5)
    gaussian_kernel = gaussian_kernel @ gaussian_kernel.T

    # Apply Gaussian filter
    img_filtered = cv2.filter2D(img, -1, gaussian_kernel)

    # Fourier Transform of filtered image
    f_filtered = np.fft.fft2(img_filtered)
    fshift_filtered = np.fft.fftshift(f_filtered)
    magnitude_spectrum_filtered = np.log1p(np.abs(fshift_filtered))

    # Fourier Transform of Gaussian kernel
    f_kernel = np.fft.fft2(gaussian_kernel, s=img.shape)
    fshift_kernel = np.fft.fftshift(f_kernel)
    magnitude_spectrum_kernel = np.log1p(np.abs(fshift_kernel))

    # Multiplication in frequency domain (Convolution theorem)
    f_convolution = fshift * fshift_kernel
    magnitude_spectrum_convolution = np.log1p(np.abs(f_convolution))

    # Inverse Fourier Transform (Reconstructed Image)
    img_reconstructed = np.fft.ifft2(np.fft.ifftshift(fshift))
    img_reconstructed = np.abs(img_reconstructed)

    # **Fix: Normalize and Convert to uint8**
    img_reconstructed = cv2.normalize(img_reconstructed, None, 0, 255, cv2.NORM_MINMAX)
    img_reconstructed = np.uint8(img_reconstructed)

    # Debugging: Ensure all images are valid
    images = [
        (img, "Original Image"),
        (magnitude_spectrum, "Fourier Spectrum"),
        (rotated_img, "Rotated Image (45°)"),
        (magnitude_spectrum_rot, "Fourier Spectrum (Rotated)"),
        (img_filtered, "Gaussian Filtered Image"),
        (magnitude_spectrum_filtered, "Fourier of Filtered Image"),
        (magnitude_spectrum_kernel, "Fourier of Gaussian Kernel"),
        (magnitude_spectrum_convolution, "Multiplication in Frequency Domain"),
        (img_reconstructed, "Reconstructed Image (Inverse FT)"),
    ]

    # Print shapes and values for debugging
    for idx, (image, title) in enumerate(images):
        print(f"Image {idx+1}: {title}, Shape: {image.shape}, Min: {image.min()}, Max: {image.max()}")

    # Display results
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for ax, (image, title) in zip(axes.flat, images):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# Example usage
image_path = "D:/DIPLAB/FT1.jpg"  # Change this to your image path
fourier_analysis(image_path)
