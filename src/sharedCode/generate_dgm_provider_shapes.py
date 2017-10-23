import skimage.morphology
import numpy as np
from pershombox import calculate_discrete_NPHT_2d


DGM_MIN_PERSISTENCE_THRESHOLD = 0.01


def reduce_to_largest_connected_component(img):
    label_map, n = skimage.morphology.label(img, neighbors=4, background=0, return_num=True)
    volumes = []
    for i in range(n):
        volumes.append(np.count_nonzero(label_map == (i + 1)))

    arg_max = np.argmax(volumes)
    img = (label_map == (arg_max + 1))

    return img


def get_npht(img, number_of_directions):
    img = np.ndarray.astype(img, bool)

    npht = calculate_discrete_NPHT_2d(img, number_of_directions)
    return npht


def threhold_dgm(dgm):
    return list(p for p in dgm if p[1]-p[0] > DGM_MIN_PERSISTENCE_THRESHOLD)
