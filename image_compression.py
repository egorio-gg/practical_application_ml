import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Image:

    def __init__(self, path: str, name: str = None):
            self.image = cv.imread(path)
            if name:
                self.name = name
            else:
                self.name = path

            self.image_grayscale = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

    def print(self, image, title):
        plt.figure(figsize=(100, 100))
        plt.imshow(image)
        plt.axis('off')
        plt.title(title)
        plt.show()

    def __str__(self):
        self.__print__(self.image, self.name)

    def __repr__(self):
        self.__print__(self.image, self.name)

    def change_cmap(self, cmap: int):
        self.image = cv.cvtColor(self.image, cmap)

    def pca_compression(self, 
                        n_components: int, 
                        image: np.ndarray = None):
        if image is None:
            image = self.image

        if image.shape[1] < n_components:
            print('For using PCA n_components ({0}) \
                  must be less then the width of the image ({1})'.format(n_components, M))
            return image
        
        if (len(image.shape) < 3):
            pca = PCA(n_components)
            image_compressed = pca.fit_transform(self.image)
            image_decompressed = pca.inverse_transform(image_compressed)
        else:
            R = image[:, :, 0]
            G = image[:, :, 1]
            B = image[:, :, 2]
            colors = [R, G, B]
            colors_decompressed = []
            for color in colors:
                pca = PCA(n_components)
                color_compressed = pca.fit_transform(color)
                color_decompressed = pca.inverse_transform(color_compressed)[:, :, np.newaxis]
                vmin, vmax = color_decompressed.min(), color_decompressed.max()
                color_decompressed = (255 * (color_decompressed - vmin)/(vmax - vmin)).astype(int)
                colors_decompressed += [color_decompressed]
            image_decompressed = np.concatenate(colors_decompressed, axis=2)

        return image_decompressed


def pair_plot(original: np.ndarray, compressed: np.ndarray, titles: list, cmap = None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(100, 100))
    axes[0].imshow(original, cmap=cmap)
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    axes[1].imshow(compressed, cmap=cmap)
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    plt.show()

path = r"sources\bridge_golden_gate.jpg"
image = Image(path)
image.change_cmap(cv.COLOR_BGR2RGB)

n_components = 50
compressed_image = image.pca_compression(n_components)
titles = ['Original Image', f'Compressed image with ratio {round(image.image.shape[1] / n_components, 2)}']
pair_plot(image.image, compressed_image, titles)