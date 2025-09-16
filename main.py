import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn import preprocessing
import sys
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QPushButton

scaler = preprocessing.MinMaxScaler()

# Clustering
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)

img = None

# Function to get the file
def get_file():
    file, _ = QFileDialog.getOpenFileName(
        None,
        "Select a File you would like to read the colour palette of.",
        "",
        "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)"
    )

    if file:
        with open(file, "rb"):
            return file
    else:
        print("Error: No file has been selected.")

def palette():    

    file = get_file()
    if not file:
        return
    
    # Read, Convert & Flatten 
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Colour Palette Detector")

        button = QPushButton("Find Palette")
        button.setCheckable(True)
        button.clicked.connect(palette)

        self.setFixedSize(QSize(300, 200))
        self.setCentralWidget(button)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
