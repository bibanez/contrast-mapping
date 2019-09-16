import cv2
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

# Method explanation:
# 100 | 120 | 100
#  80 |_130_| 120 >>>>> Average: 145 >>>>> Result: 145 - 130 = 15 on the center
# 230 | 210 | 200
#
# The pixel in the center is the one studied. For the contrast mapping, you
# compute the average of the pixels around it and then subtract that from the
# center. This then outputs the first contrast layer. The second contrast layer
# is the same as the first but applied on it and only on the pixels that average
# empty. In the first and second layer a correction factor is applied to the
# pixel values and a threshold for writing the pixels is also in place.

# Hyperparameters
IMG_SIZE = 224 # For resizing
pattern_3x3 = [-1, 0, 1] # Pattern of distribution
correction = 1.5 # The factor of correction in the contrast
threshold = 8 # Minimum value for contrast to be processed

img_array = cv2.imread('zooniverse.png') # Image name
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
(channel_b, channel_g, channel_r) = cv2.split(new_array)

contrasts = {} # First layer of contrast mapping
processed = {} # Second layer of contrast mapping
subtracted = {} # Processed result

for c in ["red", "green", "blue"]:

    channel = []
    if c == "blue": channel = channel_b
    if c == "green": channel = channel_g
    if c == "red": channel = channel_r

    # First contrast processing
    contrasts[c] = np.zeros(channel.shape)

    for x, y in product(range(channel.shape[0]), range(channel.shape[1])):
        total_sum = 0
        pixels = 0
        for n, m in product(pattern_3x3, pattern_3x3):
            try:
                if n + x >= 0 and m + y >= 0 and (n != 0 and m != 0):
                    total_sum += channel[n + x][m + y]
                    pixels += 1
            except:
                continue
        if abs(channel[x][y] - total_sum / pixels) >= threshold:
            contrasts[c][x][y] = abs(channel[x][y] - total_sum / pixels) * correction

    #Second contrast processing
    processed[c] = contrasts[c]

    for x, y in product(range(channel.shape[0]), range(channel.shape[1])):
        if contrasts[c][x][y] == 0:
            total_sum = 0
            pixels = 0
            for n, m in product(pattern_3x3, pattern_3x3):
                try:
                    if n + x >= 0 and m + y >= 0 and (n != 0 and m != 0):
                        total_sum += contrasts[c][n + x][m + y]
                        pixels += 1
                except:
                    continue
            processed[c][x][y] = abs(contrasts[c][x][y] - total_sum / pixels) * correction

    # Subtracting the processed contrast from the original image
    subtracted[c] = np.zeros(channel.shape)

    for x, y in product(range(channel.shape[0]), range(channel.shape[1])):
        subtract = channel[x][y] - contrasts[c][x][y]
        if subtract >= 0:
            subtracted[c][x][y] = subtract
        else:
            subtracted[c][x][y] = 0

final = np.dstack((subtracted["red"], subtracted["green"], subtracted["blue"])).astype(int)

#plt.imshow(new_array)
plt.imshow(final)
plt.show()
