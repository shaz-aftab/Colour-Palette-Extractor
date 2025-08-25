import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()

# Read, Convert & Flatten 
img = cv.imread("images/image_1.jpg")
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
pixels = scaler.fit_transform(img_rgb.reshape((-1, 3)))

# Clustering
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixels)

colours_normalized = kmeans.cluster_centers_

h, w = 50, 50 * k

strip = np.zeros((h, w, 3))

for i, colour in enumerate(colours_normalized):
    StartX = i * 50
    EndX = (i + 1) * 50
    strip[:, StartX:EndX, :] = colour

# Display

plt.figure(figsize=(8, 2))
plt.imshow(strip)
plt.axis("off")
plt.show()