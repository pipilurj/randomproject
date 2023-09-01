import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_masks(mask_pairs):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    axs = axs.flatten()

    for i, (mask1, mask2) in enumerate(mask_pairs):
        ax = axs[i]

        for polygon in mask1:
            poly = Polygon(polygon, edgecolor='r', facecolor='none')
            ax.add_patch(poly)

        for polygon in mask2:
            poly = Polygon(polygon, edgecolor='b', facecolor='none')
            ax.add_patch(poly)

        ax.autoscale_view()
        ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

# Example usage
mask1 = [
    [
        [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]],
        [[0.6, 0.6], [0.8, 0.6], [0.8, 0.8], [0.6, 0.8]],
        [[0.1, 0.4], [0.3, 0.4], [0.3, 0.6], [0.1, 0.6]],
        [[0.5, 0.2], [0.7, 0.2], [0.7, 0.4], [0.5, 0.4]],
        [[0.2, 0.6], [0.4, 0.6], [0.4, 0.8], [0.2, 0.8]],
        [[0.7, 0.4], [0.9, 0.4], [0.9, 0.6], [0.7, 0.6]],
        [[0.4, 0.1], [0.6, 0.1], [0.6, 0.3], [0.4, 0.3]],
        [[0.8, 0.1], [1.0, 0.1], [1.0, 0.3], [0.8, 0.3]],
        [[0.1, 0.8], [0.3, 0.8], [0.3, 1.0], [0.1, 1.0]]
    ]
]

mask2 = [
    [
        [[0.1, 0.5], [0.3, 0.5], [0.3, 0.7], [0.1, 0.7]],
        [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],
        [[0.5, 0.6], [0.7, 0.6], [0.7, 0.8], [0.5, 0.8]],
        [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]],
        [[0.6, 0.2], [0.8, 0.2], [0.8, 0.4], [0.6, 0.4]],
        [[0.2, 0.7], [0.4, 0.7], [0.4, 0.9], [0.2, 0.9]],
        [[0.5, 0.8], [0.7, 0.8], [0.7, 1.0], [0.5, 1.0]],
        [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]],
        [[0.6, 0.9], [0.8, 0.9], [0.8, 1.0], [0.6, 1.0]]
    ]
]

mask_pairs = list(zip(mask1[0], mask2[0]))

plot_masks(mask_pairs)