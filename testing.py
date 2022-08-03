import get_images as gi
from matplotlib import pyplot as plt


imgs = gi.load_images()
for i in range(20, 500, 100):
    plt.imshow(imgs[i, 0], interpolation='nearest')
    plt.show()