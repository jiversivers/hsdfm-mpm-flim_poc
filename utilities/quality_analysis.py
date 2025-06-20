
import numpy as np
import matplotlib.pyplot as plt

# Display the image and let user select a line
def get_line_profile(image):
    from scipy.ndimage import map_coordinates

    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.title("Click two points to define a line")
    pts = plt.ginput(2)
    plt.close()

    (x0, y0), (x1, y1) = pts
    length = int(np.hypot(x1 - x0, y1 - y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
    intensities = map_coordinates(image, [y, x])

    # Plot the intensity profile
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 15))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original image")
    ax[0].plot((x0, x1), (y0, y1))

    ax[1].plot(intensities)
    ax[1].set_title("Intensity Profile")
    ax[1].set_xlabel("Distance along line")
    ax[1].set_ylabel("Intensity")
    plt.show()

    return fig


# Map the gradients of the image
def map_edge_contrast(image):
    from scipy.ndimage import sobel
    fig, ax = plt.subplots(figsize=(8, 6))

    dx = sobel(image, axis=0)
    dy = sobel(image, axis=1)
    grad_image = np.hypot(dx, dy)

    ax.imshow(grad_image, cmap='gray')
    ax.set_title("Edge Contrast")
    ax.set(
        xlabel=f"Min. = {grad_image.min():0.3f}. Max = {grad_image.max():0.3f}",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
    return fig


# # Add scale bar
# def add_scale_bar(image):


if __name__ == '__main__':
    import argparse
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument('--function', '-f', type=str, choices=['get_line_profile', 'map_edge_contrast'])
    parser.add_argument('--image-path', '-i', type=str)
    args = parser.parse_args()
    img = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    fn = globals()[args.function]
    fn(img)