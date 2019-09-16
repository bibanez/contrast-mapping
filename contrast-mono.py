import cv2
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

# Hyperparameters
IMG_SIZE = 224
pattern_3x3 = [-1, 0, 1] # Pattern of distribution
correction = 2 # The factor of correction in the contrast
threshold = 8 # Minimum value for contrast to be processed

img_array = cv2.imread('zooniverse.png', cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

# First contrast processing
contrast = np.zeros(new_array.shape)

for x, y in product(range(new_array.shape[0]), range(new_array.shape[1])):
    total_sum = 0
    pixels = 0
    for n, m in product(pattern_3x3, pattern_3x3):
        try:
            if n + x >= 0 and m + y >= 0 and (n != 0 and m != 0):
                total_sum += new_array[n + x][m + y]
                pixels += 1
        except:
            continue
    if abs(new_array[x][y] - total_sum / pixels) >= threshold:
        contrast[x][y] = abs(new_array[x][y] - total_sum / pixels) * correction

#Second contrast processing
processed = contrast

for x, y in product(range(new_array.shape[0]), range(new_array.shape[1])):
    if contrast[x][y] == 0:
        total_sum = 0
        pixels = 0
        for n, m in product(pattern_3x3, pattern_3x3):
            try:
                if n + x >= 0 and m + y >= 0 and (n != 0 and m != 0):
                    total_sum += contrast[n + x][m + y]
                    pixels += 1
            except:
                continue
        processed[x][y] = abs(contrast[x][y] - total_sum / pixels) * correction

# Subtracting the processed contrast from the original image
subtracted = np.zeros(new_array.shape)

for x, y in product(range(new_array.shape[0]), range(new_array.shape[1])):
    subtract = new_array[x][y] - contrast[x][y]
    if subtract >= 0:
        subtracted[x][y] = subtract
    else:
        subtracted[x][y] = 0

#print(new_array)
#print(contrast)
#print(abs(new_array - contrast))
plt.imshow(subtracted)
#plt.imshow(new_array)
#plt.imshow(contrast)
#plt.imshow(processed)
plt.show()
