import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('smarties.jpg')

# conversion to grayscale.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h = img_gray.shape[0] # height of the image
w = img_gray.shape[1] # width
# threshold
ret, img_threshold = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY_INV)
ret, img_thresh = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY)

#erosion
circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
circle[0,3] = 0
circle[6,3] = 0
print(circle)
img_erosion = cv2.erode(img_threshold, circle, iterations = 6)

ret, labels = cv2.connectedComponents(img_erosion)
print('labels=', labels)
ret, label_fond = cv2.connectedComponents(img_thresh)

img_mark = labels+label_fond

# distance map
carte_dist = cv2.distanceTransform(img_threshold, cv2.DIST_L2, 3)

dist_map = np.uint8(255 - (carte_dist - carte_dist.min())*255/carte_dist.max() - img_thresh)
print('dist min= ', dist_map)


plt.figure(1)
plt.subplot(221)
plt.imshow(img_gray, 'gray')
plt.title("Grayscale image")
plt.subplot(222)
plt.imshow(img_threshold, 'gray')
plt.title("Binary image : white for smarties and black for the background")
plt.subplot(223)
plt.imshow(img_erosion, 'gray')
plt.title("Erosion of the binary image with ellipsoidal structuring element")
plt.subplot(224)
plt.imshow(img_mark)
plt.title("Creation of labels for each smarties")
plt.colorbar()
plt.show()

imgmark_copy = img_mark

plt.figure(2)
plt.imshow(dist_map, 'gray')
plt.title("Distance map")
plt.colorbar()
plt.show()


h = img_mark.shape[0]
w = img_mark.shape[1]
print(h, w)
#FAH
#initialization of hierarchical queues
FAH_x = [[] for i in range(256)]
FAH_y = [[] for i in range(256)]

# We check the value of the distance map and we fill the queue with corresponding markers.
for i in range(h):
    for j in range(w):
        if (img_mark[i][j] != 0):
            val = dist_map[i,j]
            FAH_x[val].append(i)
            FAH_y[val].append(j)

file = 0

# Watershed implementation
while (file < 256):
    if (FAH_x[file]):
        x = FAH_x[file].pop(0)
        y = FAH_y[file].pop(0)
        if (x > 0):
            if (img_mark[x-1][y] == 0):
                FAH_x[dist_map[x-1][y]].append(x-1)
                FAH_y[dist_map[x-1][y]].append(y)
                img_mark[x-1][y] = img_mark[x][y]

        if (x < h-1):
            if (img_mark[x+1][y] == 0):
                FAH_x[dist_map[x +1][y]].append(x + 1)
                FAH_y[dist_map[x +1][y]].append(y)
                img_mark[x +1][y] = img_mark[x][y]

        if (y > 0):
            if (img_mark[x][y-1] == 0):
                FAH_x[dist_map[x][y-1]].append(x)
                FAH_y[dist_map[x][y-1]].append(y-1)
                img_mark[x][y-1] = img_mark[x][y]

        if (y < w-1):
            if (img_mark[x][y+1] == 0):
                FAH_x[dist_map[x][y+1]].append(x)
                FAH_y[dist_map[x][y+1]].append(y+1)
                img_mark[x][y+1] = img_mark[x][y]

    else:
        file = file + 1

plt.figure(3)
plt.imshow(img_mark)
plt.title("Image after watershed implementation")
plt.show()
# There are a little loophole in the final image, we can see that some smarties are cropped.