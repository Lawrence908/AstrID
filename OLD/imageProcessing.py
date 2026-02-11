import cv2
import numpy as np
import pandas as pd
from astropy.io import fits
import os
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerPatch

from scripts.dataGathering import getPixelCoordsFromStar, extractStarsFromFits

def extractImageFromFits(fits_file):
    """
    Extract the image data from a FITS file.

    Parameters:
    fits_file (str): The path to the FITS file.

    Returns:
    numpy.ndarray: The image data extracted from the FITS file.
    """
    with fits.open(fits_file) as hdul:
        image_data = hdul[0].data
    return image_data

def extractPixelMaskFromFits(fits_file):
    """
    Extract the pixel mask from a FITS file.

    Parameters:
    fits_file (str): The path to the FITS file.

    Returns:
    numpy.ndarray: The pixel mask extracted from the FITS file.
    """
    with fits.open(fits_file) as hdul:
        pixel_mask = hdul['pixel_mask'].data
        return pixel_mask

def extractWCSFromFits(fits_file):
    """
    Extract the WCS (World Coordinate System) information from a FITS file.

    Parameters:
    fits_file (str): The path to the FITS file.

    Returns:
    astropy.wcs.WCS: The WCS object extracted from the FITS file.
    """
    with fits.open(fits_file) as hdul:
        wcs = WCS(hdul[0].header)
    return wcs

def extractOverlayFromFits(fits_file):
    """
    Extract the overlay image from a FITS file.

    Parameters:
    fits_file (str): The path to the FITS file.

    Returns:
    numpy.ndarray: The overlay image extracted from the FITS file.
    """
    with fits.open(fits_file) as hdul:
        overlay_image = hdul['star_overlay'].data
    return overlay_image

def stackImages(images):
    """
    Stack images to create a 3D array.

    Parameters:
    images (list of numpy.ndarray): List of 2D image arrays.

    Returns:
    numpy.ndarray: 3D array of stacked images.
    """
    stacked_images = np.array([np.stack([image, image, image], axis=-1) for image in images])

    prepared_images = np.array(stacked_images)

    return prepared_images


def stackMasks(masks):
    """
    Stack masks to create a an additional dimension.

    Parameters:
    masks (list of numpy.ndarray): List of 2D mask arrays.

    Returns:
    numpy.ndarray: Array of stacked masks with another dimension.
    """
    masks = np.array([np.expand_dims(mask, axis=-1) for mask in masks])
    
    prepared_masks = np.array(masks)

    return prepared_masks
    

def normalizeImages(images):
    """
    Normalize images to a range between 0 and 1.

    Parameters:
    images (numpy.ndarray): Array of images to normalize.

    Returns:
    numpy.ndarray: Normalized images.
    """
    images = images.copy()

    # Find the minimum and maximum values in the dataset
    min_val = np.min(images)
    max_val = np.max(images)

    # Apply min-max normalization
    images_normalized = (images - min_val) / (max_val - min_val)

    return images_normalized


def convert_to_grayscale(image):
    """
    Convert an image to grayscale.

    Parameters:
    image (numpy.ndarray): Input image.

    Returns:
    numpy.ndarray: Grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    Apply Gaussian blur to an image.

    Parameters:
    image (numpy.ndarray): Input image.
    kernel_size (tuple, optional): Size of the Gaussian kernel. Default is (5, 5).

    Returns:
    numpy.ndarray: Blurred image.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_threshold(image, threshold_value=30):
    """
    Apply binary thresholding to an image.

    Parameters:
    image (numpy.ndarray): Input image.
    threshold_value (int, optional): Threshold value. Default is 30.

    Returns:
    numpy.ndarray: Binary thresholded image.
    """
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary

def apply_morphological_operations(image, kernel_size=(3, 3)):
    """
    Apply morphological operations (dilation and erosion) to an image.

    Parameters:
    image (numpy.ndarray): Input image.
    kernel_size (tuple, optional): Size of the structuring element. Default is (3, 3).

    Returns:
    numpy.ndarray: Image after morphological operations.
    """
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=9)
    eroded = cv2.erode(dilated, kernel, iterations=9)
    return eroded

def normalize_image(image):
    """
    Normalize an image to a range between 0 and 1.

    Parameters:
    image (numpy.ndarray): Input image.

    Returns:
    numpy.ndarray: Normalized image.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val)
    return normalized

def preprocessImage(image, kernel_size=(1, 1), threshold_value=100):
    """
    Preprocess an image by applying grayscale conversion, Gaussian blur, normalization, thresholding, and morphological operations.

    Parameters:
    image (numpy.ndarray): Input image.
    kernel_size (tuple, optional): Size of the Gaussian kernel. Default is (1, 1).
    threshold_value (int, optional): Threshold value. Default is 100.

    Returns:
    numpy.ndarray: Preprocessed image.
    """
    # Convert to 3-channel if the image is single-channel
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        image = np.stack([image, image, image], axis=-1)

    gray = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(gray, kernel_size)
    normalized = normalize_image(blurred)
    binary = apply_threshold(normalized * 255, threshold_value)  # Scale back to 0-255 for thresholding
    morphed = apply_morphological_operations(binary)
    final_normalized = normalize_image(morphed)
    return final_normalized

def extractStarPredictions(prediction, threshold=0.5, wcs_data=None):
    """
    Extract star predictions from a prediction array.

    Parameters:
    prediction (numpy.ndarray): Prediction array.
    threshold (float, optional): Threshold value for star detection. Default is 0.5.
    wcs_data (astropy.wcs.WCS, optional): WCS object for coordinate transformation. Default is None.

    Returns:
    tuple: A tuple containing the star data and the prediction mask.
    """
    # Normalize the prediction array to be between 0 and 1
    prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())

    # Ensure the prediction array is 2D
    if prediction.ndim == 3:
        prediction = prediction[:, :, 0]

    # Threshold the prediction array to get the star locations
    stars = np.argwhere(prediction > threshold)

    # Create a list to store the star data
    star_data = []

    # Create a prediction mask of the same shape as the prediction array
    prediction_mask = np.zeros_like(prediction, dtype=np.uint8)

    # Iterate over the star locations and add them to the star data list and prediction mask
    for star_location in stars:
        y, x = star_location
        star_data.append((x, y))
        prediction_mask[y, x] = 1

    return star_data, prediction_mask


def getPredictionComparison(fits_file, model, threshold=0.5, save_prediction=False):
    """
    Generate a comparison plot of the original image, pixel mask, and model prediction.

    Parameters:
    fits_file (str): The path to the FITS file.
    model (keras.Model): The trained model.
    threshold (float, optional): Threshold value for prediction. Default is 0.5.
    save_prediction (bool, optional): Whether to save the prediction plot. Default is False.
    """
    file_path = os.path.join("data", "fits", "validate", fits_file)
    image = extractImageFromFits(file_path)
    test_image = stackImages(image)
    pixel_mask = extractPixelMaskFromFits(file_path)
    wcs = extractWCSFromFits(file_path)
    overlay_image = extractOverlayFromFits(file_path)
    stars = extractStarsFromFits(file_path)
    
    pred_mask = model.predict(np.expand_dims(test_image, axis=0), batch_size=1)[0]
    # Normalize the prediction array to be between 0 and 1
    pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min())
    # Apply the threshold to create a binary mask
    pred_mask = (pred_mask > threshold).astype(np.uint8)

    fig, ax = plt.subplots(1, 3, figsize=(30, 10), subplot_kw={'projection': wcs})
    ax[0].imshow(image, cmap='gray', origin='lower')
    ax[0].set_title('Image')
    ax[0].coords.grid(True, color='white', ls='dotted')
    ax[0].coords[0].set_axislabel('RA')
    ax[0].coords[1].set_axislabel('Dec')

    ax[1].imshow(pixel_mask, cmap='gray', origin='lower')
    ax[1].set_title('Mask')
    ax[1].coords.grid(True, color='white', ls='dotted')
    ax[1].coords[0].set_axislabel('RA')
    ax[1].coords[1].set_axislabel('Dec')

    ax[2].imshow(pred_mask, cmap='gray', origin='lower')
    ax[2].set_title('Prediction')
    ax[2].coords.grid(True, color='white', ls='dotted')
    ax[2].coords[0].set_axislabel('RA')
    ax[2].coords[1].set_axislabel('Dec')

    image_title = fits_file + " Image, Pixel Mask, and Prediction Comparison"
    plt.suptitle(image_title, fontsize=24)
    
    if save_prediction:
        file_path = 'results/figures/prediction_comparison/' + fits_file.replace('.fits', '.png')
        plt.savefig(file_path)
        print(f'Saving Prediction Comparison: {file_path}')
        # Do not show plot if saving
        plt.close()
    else:
        plt.show()



def getPredictionOverlay(fits_file, model, threshold=0.5, cmap='gray_r', save_prediction=False):
    """
    Generate an overlay plot of the original image with star locations and model predictions.

    Parameters:
    fits_file (str): The path to the FITS file.
    model (keras.Model): The trained model.
    threshold (float, optional): Threshold value for prediction. Default is 0.5.
    cmap (str, optional): Colormap for the image. Default is 'gray_r'.
    save_prediction (bool, optional): Whether to save the prediction plot. Default is False.
    """
    file_path = os.path.join("data", "fits", "validate", fits_file)
    image = extractImageFromFits(file_path)
    test_image = stackImages(image)
    wcs = extractWCSFromFits(file_path)
    pixel_mask = extractPixelMaskFromFits(file_path)
    stars = extractStarsFromFits(file_path)
    pred_star_data, prediction_mask = extractStarPredictions(model.predict(np.expand_dims(test_image, axis=0), batch_size=1)[0], threshold=threshold)
    print("Number of stars detected:", len(pred_star_data))
    # image = image[:, :, 0]


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=wcs)

    # Draw blue circles on the image for pixel mask
    x_dim = wcs.pixel_shape[0]
    y_dim = wcs.pixel_shape[1]

    # Pixel-mask of stars
    pixel_mask = np.zeros((x_dim, y_dim))

    for star in stars:
        pixel_coords = getPixelCoordsFromStar(star, wcs)
        # Ensure the pixel coordinates are within bounds
        x, y = int(np.round(pixel_coords[0])), int(np.round(pixel_coords[1]))
        if 0 <= x < x_dim and 0 <= y < y_dim:
            pixel_mask[x][y] = 1

        drawing_colored_circle = plt.Circle((pixel_coords[0], pixel_coords[1]), 7, fill=False, edgecolor='blue', linewidth=0.75)
        ax.add_artist(drawing_colored_circle)

    # Plot the image
    ax.imshow(image, cmap=cmap, origin='lower')

    # Draw red circles on the image for star predictions
    x_dim = wcs.pixel_shape[0]
    y_dim = wcs.pixel_shape[1]

    # Pixel-mask of stars
    pixel_mask = np.zeros((x_dim, y_dim))

    for star in pred_star_data:
        pixel_coords = star
        # Ensure the pixel coordinates are within bounds
        x, y = int(np.round(pixel_coords[0])), int(np.round(pixel_coords[1]))
        if 0 <= x < x_dim and 0 <= y < y_dim:
            pixel_mask[x][y] = 1

        drawing_colored_circle = plt.Circle((pixel_coords[0], pixel_coords[1]), 1, fill=False, edgecolor='red', linewidth=0.1)
        ax.add_artist(drawing_colored_circle)

    image_title = fits_file + " with Star Location and Star Prediction Overlays" 
    ax.set_title(f'{image_title}')
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.grid(color='white', ls='dotted')

    # Add legend
    def make_legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        return Circle((width / 2, height / 2), 0.25 * height, fill=False, edgecolor=orig_handle.get_edgecolor(), linewidth=orig_handle.get_linewidth())


    # Display a legend for the circles
    blue_circle = Circle((0, 0), 1, fill=False, edgecolor='blue', linewidth=1)
    red_circle = Circle((0, 0), 1, fill=False, edgecolor='red', linewidth=1)
    ax.legend([blue_circle, red_circle], ['Pixel Mask', 'Star Prediction'], loc='upper right', handler_map={Circle: HandlerPatch(patch_func=make_legend_circle)})
    

    image_title = fits_file + " Prediction Overlay"
    plt.suptitle(image_title, fontsize=24)
    
    if save_prediction:
        file_path = 'results/figures/prediction_overlay/' + fits_file.replace('.fits', '.png')
        plt.savefig(file_path)
        print(f'Saving Prediction Overlay: {file_path}')
        # Do not show plot if saving
        plt.close()
    else:
        plt.show()
