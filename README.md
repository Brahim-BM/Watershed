# Watershed : Python2.7 and OpenCV
package : OpenCV2, numpy, matplotlib.pyplot

An image contains a certain number of identical objects. We want to separate the objects present on the image, that is to say segment the image. In order to implement the watershed method, we need first to apply image pre-processing.

1. Conversion of the image in grayscale image.
2. Threshold to separate objects from the background.
3. Morphological erosion of the image with an ellipsoid structuring element to obtain a marker for each object in the image.
4. Assigning a label to each marker and background
5. Calculation of the distance map
