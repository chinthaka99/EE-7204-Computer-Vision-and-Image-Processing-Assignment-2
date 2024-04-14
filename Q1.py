import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage.util import random_noise
from skimage.io import imread

# Add Gaussian noise to the image
def add_gaussian_noise(image, mean=0, std=0.1):
    noisy_image = random_noise(image, mode='gaussian', mean=mean, var=std**2)
    return noisy_image

# Implement Otsu's algorithm
def apply_otsu_threshold(image):
    threshold_value = filters.threshold_otsu(image)
    binary_image = image > threshold_value
    return binary_image

# Main function
def main():
    # Load the image 
    image_path = 'test.jpg'
    original_image = imread(image_path)
    original_image_gray = imread(image_path, as_gray=True)

    # Add Gaussian noise to the image
    noisy_image = add_gaussian_noise(original_image_gray)

    # Apply Otsu's algorithm
    segmented_image = apply_otsu_threshold(noisy_image)

    # Display original image, noisy image, and segmented image
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax = axes.ravel()
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[1].imshow(noisy_image, cmap='gray')
    ax[1].set_title('Noisy Image')
    ax[2].imshow(segmented_image, cmap='gray')
    ax[2].set_title('Segmented Image (Otsu)')
    for a in ax:
        a.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
