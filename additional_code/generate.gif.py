import sys
sys.path.append("../graph_neural_network")

import imageio  # to save GIFs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import cv2  # optional (for resizing the filter to look better)

from graph_neural_network import BorisGraphNet

def visualize_filter(A, filter_name):
    scale = 10
    img_list = []
    cmap = mpl.cm.get_cmap('plasma')
    for i in np.arange(0, img_size, 4):  # for every row with step 4
        for j in np.arange(0, img_size, 4):  # for every col with step 4
            k = i * img_size + j
            img = A[k, :].reshape(img_size, img_size)
            img = (img - img.min()) / (img.max() - img.min())
            img = cmap(img)
            img[i, j] = np.array([1., 0, 0, 0])  # add the red dot
            img = cv2.resize(img, (img_size*scale, img_size*scale))
            img_list.append((img * 255).astype(np.uint8))
    imageio.mimsave(filter_name, img_list, format='GIF', duration=0.2)

obj = BorisGraphNet()
img_size = 48
sparse_adj = obj.precompute_adjacency_images(img_size)
gauss_adj = obj.guassian_precompute_adjacency_images(img_size)
visualize_filter(sparse_adj, "sparse_filter.gif")
visualize_filter(gauss_adj, "gaussian_filter.gif")
