import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn import preprocessing
import tkinter 
from tkinter import *
from tkinter import filedialog as fidia

scaler = preprocessing.MinMaxScaler()
root = tkinter.Tk()

# Clustering
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)

def main():
    # Read, Convert & Flatten 
    file = fidia.askopenfilename()
    if file: 
        with open(file, "rb"):
            img = cv.imread(file)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pixels = scaler.fit_transform(img_rgb.reshape((-1, 3)))
    
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

root.geometry("350x150")
frame = Frame(root)
frame.pack()

CENTER_FRAME = Frame(root)
CENTER_FRAME.pack()
CENTER_FRAME.place(relx=.5, rely=.5, anchor="center")

label = Label(frame, text = "\nDetect Colour Palette")
label.pack()

detect = Button(CENTER_FRAME, text = "Detect", command= main() )
detect.pack(padx = 3, pady = 3)

root.title("Colour Palette Detector")
root.mainloop()