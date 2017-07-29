import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# Plot the result
# Plot the result
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(gradx, cmap='gray')
ax1.set_title('gradx', fontsize=50)
ax2.imshow(grady, cmap='gray')
ax2.set_title('grady', fontsize=50)
ax3.imshow(mag_binary, cmap='gray')
ax3.set_title('mag_binary', fontsize=50)
ax4.imshow(dir_binary, cmap='gray')
ax4.set_title('dir_binary', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)