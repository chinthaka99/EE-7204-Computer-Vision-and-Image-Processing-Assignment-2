import numpy as np
import cv2

def region_growing(image, seed, threshold):
    # Create a mask that will contain the segmented region
    segmented = np.zeros_like(image)
    
    # Mark the seed point as visited
    visited = np.zeros_like(image, dtype=bool)
    
    # Define the connectivity (8-connectivity)
    connectivity = [(x, y) for x in range(-1, 2) for y in range(-1, 2) if not (x == 0 and y == 0)]
    
    # Define a function to check if a pixel is within the image bounds
    def is_valid_pixel(pixel):
        return 0 <= pixel[0] < image.shape[0] and 0 <= pixel[1] < image.shape[1]
    
    # Define a function to check if a pixel is in the threshold range
    def is_in_threshold(pixel, seed_value):
        return abs(image[pixel[0], pixel[1]] - seed_value) <= threshold
    
    # Start growing the region from the seed point
    stack = [seed]
    seed_value = image[seed[0], seed[1]]
    while stack:
        current_pixel = stack.pop()
        segmented[current_pixel[0], current_pixel[1]] = 255  
        visited[current_pixel[0], current_pixel[1]] = True  
        
        # Check neighboring pixels
        for dx, dy in connectivity:
            neighbor = (current_pixel[0] + dx, current_pixel[1] + dy)
            if is_valid_pixel(neighbor) and not visited[neighbor[0], neighbor[1]]:
                if is_in_threshold(neighbor, seed_value):
                    stack.append(neighbor)
    
    return segmented

# Load the image
image = cv2.imread('test2.png', cv2.IMREAD_GRAYSCALE)

seed_point = (100, 100)

threshold_value = 20

# Perform region growing segmentation
segmented_image = region_growing(image, seed_point, threshold_value)

# Concatenate original and segmented images horizontally
combined_image = np.hstack((image, segmented_image))


cv2.imshow('Original and Segmented Image', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the combined image
cv2.imwrite('combined_image.png', combined_image)
