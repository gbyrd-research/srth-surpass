# import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import cv2

img = np.zeros((500, 500, 3), dtype=np.uint8)
cv2.imshow("image", img)

while True:
    key = cv2.waitKey(0)
    if key == ord('q'):  # Press 'q' to exit
        break
# plt.axis('off')
# plt.title("Click to get coordinates")
# plt.show()
# point = plt.ginput(1)
# print("Clicked:", point)