import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the grayscale image
img = cv2.imread('image-1.jpg', 0)

# Step 1: Calculate histogram
hist = np.zeros(256)
rows, cols = img.shape
for i in range(rows):
    for j in range(cols):
        hist[img[i, j]] += 1

# Step 2: Normalize the histogram (divide by total number of pixels)
hist = hist / (rows * cols)

# Step 3: Calculate the cumulative distribution function (CDF)
cdf = np.zeros(256)
cdf[0] = hist[0]
for i in range(1, 256):
    cdf[i] = cdf[i-1] + hist[i]

# Step 4: Normalize the CDF to map it to the full intensity range (0-255)
cdf_normalized = np.round(cdf * 255).astype(np.uint8)

# Step 5: Map the original image pixels to the equalized values
equalized_img = np.zeros_like(img)
for i in range(rows):
    for j in range(cols):
        equalized_img[i, j] = cdf_normalized[img[i, j]]

# Apply histogram equalization
auto_equalized_img = cv2.equalizeHist(img)

# Display the original and equalized images
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(equalized_img, cmap='gray'), plt.title('Equalized Image')
# plt.subplot(122), plt.imshow(auto_equalized_img, cmap='gray'), plt.title('Auto Equalized Image')
plt.show()
