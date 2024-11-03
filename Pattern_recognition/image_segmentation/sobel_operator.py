import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('human-face.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Sobel kernels for edge detection in x and y directions
Gx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

Gy = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

# Dimensions of the grayscale image
M, N = gray_image.shape

# Initialize output arrays for the gradients
sobel_x = np.zeros((M, N), dtype=np.float32)
sobel_y = np.zeros((M, N), dtype=np.float32)

# Perform convolution to apply the Sobel kernels manually
for i in range(1, M - 1):
    for j in range(1, N - 1):
        # Extract the 3x3 region around the current pixel
        region = gray_image[i - 1:i + 2, j - 1:j + 2]

        # Apply the Sobel kernel for x and y directions
        gx = np.sum(Gx * region)
        gy = np.sum(Gy * region)

        # Store the results
        sobel_x[i, j] = gx
        sobel_y[i, j] = gy

# Calculate the gradient magnitude (overall edge strength)
edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Convert results to 8-bit for visualization
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
edge_magnitude = cv2.convertScaleAbs(edge_magnitude)

# Display the images
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(1, 4, 2)
plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel X')

plt.subplot(1, 4, 3)
plt.imshow(sobel_y, cmap='gray')
plt.title('Sobel Y')

plt.subplot(1, 4, 4)
plt.imshow(edge_magnitude, cmap='gray')
plt.title('Edge Magnitude')

plt.show()
