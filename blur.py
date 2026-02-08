import cv2
import numpy as np
import matplotlib.pyplot as plt

def complete_vision_assignment(image_path):
    # ---- LOAD IMAGE ----
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    h, w = img.shape

    # ---- GAUSSIAN KERNEL ----
    ksize = 51
    sigma = 10
    g1d = cv2.getGaussianKernel(ksize, sigma)
    kernel = np.outer(g1d, g1d)
    kh, kw = kernel.shape

    # ---- SPATIAL DOMAIN CONVOLUTION ----
    spatial = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    # ---- FREQUENCY DOMAIN (LINEAR CONVOLUTION) ----
    H = h + kh - 1
    W = w + kw - 1

    img_pad = np.zeros((H, W), dtype=np.float32)
    ker_pad = np.zeros((H, W), dtype=np.float32)

    img_pad[:h, :w] = img
    ker_pad[:kh, :kw] = kernel  # correct alignment

    F_img = np.fft.fft2(img_pad)
    F_ker = np.fft.fft2(ker_pad)

    conv_full = np.real(np.fft.ifft2(F_img * F_ker))

    # SAME-size crop
    y0 = kh // 2
    x0 = kw // 2
    freq = conv_full[y0:y0+h, x0:x0+w]

    # ---- ERROR METRICS ----
    diff = np.abs(spatial - freq)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Max Difference = {max_diff:.2e}")
    print(f"Equivalent = {np.allclose(spatial, freq, atol=1e-6)}")

    # ---- VISUALIZATION PREP ----
    # Normalize difference for visibility
    diff_norm = diff / max_diff

    # Log spectra (shifted for viewing)
    img_spec = np.log1p(np.abs(np.fft.fftshift(F_img)))
    blur_spec = np.log1p(np.abs(np.fft.fftshift(F_img * F_ker)))

    # ---- PLOTTING ----
    plt.figure(figsize=(16, 9))

    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Spatial Convolution (Blur)")
    plt.imshow(spatial, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("FFT Convolution (Blur)")
    plt.imshow(freq, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Original Image Spectrum (log)")
    plt.imshow(img_spec, cmap='magma')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Normalized Difference Map")
    plt.imshow(diff_norm, cmap='inferno')
    plt.colorbar(label="Relative Error")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Blurred Spectrum (log)")
    plt.imshow(blur_spec, cmap='magma')
    plt.axis('off')

    plt.suptitle(
        f"Convolution Theorem Verification\n"
        f"Max Error = {max_diff:.2e}, Mean Error = {mean_diff:.2e}",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    complete_vision_assignment("IMG_9675.JPG")
