import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided

def create_window_views(image, window_size):
    """Create sliding window views of the image."""
    h, w = image.shape
    window = (window_size, window_size)
    
    # Calculate output shape and strides for the rolling window
    out_shape = (h - window[0] + 1, w - window[1] + 1) + window
    strides = (image.strides[0], image.strides[1]) + image.strides
    
    # Create rolling window view
    windows = as_strided(image, shape=out_shape, strides=strides)
    return windows

def pad_image(image, pad_size):
    """Pad image with reflected borders."""
    return np.pad(image, pad_size, mode='reflect')

def arithmetic_mean_filter(image, window_size=3):
    """Apply arithmetic mean filter to the image."""
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
        
    pad_size = window_size // 2
    padded = pad_image(image, pad_size)
    
    # Create window views
    windows = create_window_views(padded, window_size)
    
    # Calculate arithmetic mean
    return np.mean(windows, axis=(-1, -2))

def geometric_mean_filter(image, window_size=3):
    """Apply geometric mean filter to the image."""
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
        
    pad_size = window_size // 2
    
    # Ensure image is non-negative by shifting all values up if necessary
    min_val = np.min(image)
    if min_val < 0:
        shifted_image = image - min_val + 1e-8
    else:
        shifted_image = image + 1e-8
        
    padded = pad_image(shifted_image, pad_size)
    
    # Create window views
    windows = create_window_views(padded, window_size)
    
    # Calculate geometric mean using a numerically stable method
    log_mean = np.mean(np.log(windows), axis=(-1, -2))
    result = np.exp(log_mean)
    
    # If we shifted the image up, shift the result back down
    if min_val < 0:
        result = result + min_val - 1e-8
        
    return result

def harmonic_mean_filter(image, window_size=3):
    """Apply harmonic mean filter to the image."""
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
        
    pad_size = window_size // 2
    padded = pad_image(image, pad_size)
    
    # Create window views
    windows = create_window_views(padded, window_size)
    
    # Calculate harmonic mean
    # Add small epsilon to avoid division by zero
    n_elements = window_size * window_size
    return n_elements / np.sum(1.0 / (windows + 1e-8), axis=(-1, -2))

# Example usage:
if __name__ == "__main__":
    # Create a sample noisy image (100x100)
    # image = np.random.normal(0.5, 0.2, (100, 100))

    image = cv2.imread('grayscale1_image.jpg', cv2.IMREAD_GRAYSCALE)
    image = image / 255.0  # Normalize to [0,1] range
    
    # Add some salt noise
    salt_noise = np.random.random(image.shape) > 0.95
    image[salt_noise] = 1.0
    
    # Apply filters
    arithmetic_filtered = arithmetic_mean_filter(image, window_size=3)
    geometric_filtered = geometric_mean_filter(image, window_size=3)
    harmonic_filtered = harmonic_mean_filter(image, window_size=3)

    # Display results
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(arithmetic_filtered, cmap='gray')
    axes[1].set_title("Arithmetic Mean Filter")
    axes[1].axis("off")

    axes[2].imshow(geometric_filtered, cmap='gray')
    axes[2].set_title("Geometric Mean Filter")
    axes[2].axis("off")

    axes[3].imshow(harmonic_filtered, cmap='gray')
    axes[3].set_title("Harmonic Mean Filter")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()