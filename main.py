import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn import preprocessing

# def normalize(*args):
#     for i in range(1, len(args)):
#         return args[i]/255

scaler = preprocessing.MinMaxScaler()

# Read, Convert & Flatten 
img = cv.imread("images/image_1.jpg")
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
pixels = scaler.fit_transform(img_rgb.reshape((-1, 3)))

# Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(pixels)