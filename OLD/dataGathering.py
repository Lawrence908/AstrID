from urllib.error import HTTPError
from astroquery.skyview import SkyView
from astroquery.mast import Observations
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy.visualization import astropy_mpl_style
from astropy.table import Table
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import re
import pandas as pd





"""
Possible function imports from this file:
# Import custom functions to extract our Image arrays and Pixel Mask arrays from our created fits files dataset
from dataGathering import extractImageArray, extractPixelMaskArray, extractStarCatalog
from dataGathering import createStarDataset, getCoordRangeFromPixels, getStarsInImage, getPixelCoordsFromStar, getImagePlot, getPixelMaskPlot
from dataGathering import displayRawImage, displayRawPixelMask, displayImagePlot, displayPixelMaskPlot, displayPixelMaskOverlayPlot
from dataGathering import saveFitsImages, importDataset, printFitsHeader, printFitsContents, printStarData

# Import custom functions to import the dataset
from dataGathering import importDataset

"""

# Set the default plot style
plt.style.use(astropy_mpl_style)


def savePlotAsImage(ax, filename, pixels=1024):
    """
    Save a Matplotlib plot as an image file.

    This function saves the plot associated with the given Axes object `ax` as a PNG image file.
    It temporarily switches to the 'Agg' backend to render the plot without displaying it.

    Parameters:
    ax (matplotlib.axes.Axes): The Axes object containing the plot to save.
    filename (str): The base filename for the output image. The function will replace the '.fits' extension with '.png'.
    pixels (int, optional): The size of the output image in pixels. Default is 1024.

    Returns:
    str: The filename of the saved image.
    """
    plt.close('all')
    # Temporarily switch to the Agg backend
    original_backend = plt.get_backend()
    plt.switch_backend('Agg')

    # Save the plot as an image file
    image_filename = filename.replace('.fits', '.png')
    fig = ax.figure
    fig.set_size_inches(pixels / fig.dpi, pixels / fig.dpi)
    plt.savefig(image_filename, format='png', bbox_inches='tight', pad_inches=0)

    plt.close('all')
    # Switch back to the original backend
    plt.switch_backend(original_backend)

    return image_filename


def convertImageToFits(image_filename, fits_filename, hdu_name, pixels=1024):
    """
    Convert a saved image file to FITS format and append it to an existing FITS file.

    Parameters:
    image_filename (str): The filename of the saved image.
    fits_filename (str): The filename of the existing FITS file.
    hdu_name (str): The name of the HDU to create.
    pixels (int, optional): The size of the output image in pixels. Default is 512.
    """
    # Read the saved image file
    image_data = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to match the FITS dimensions if necessary
    image_data = cv2.resize(image_data, (pixels, pixels), interpolation=cv2.INTER_AREA)
    
    # Create a new ImageHDU for the image data
    image_hdu = fits.ImageHDU(image_data, name=hdu_name)
    
    # Append the ImageHDU to the existing FITS file
    with fits.open(fits_filename, mode='update') as hdul:
        hdul.append(image_hdu)
        hdul.flush()


def getRandomCoordinates(avoid_galactic_plane=True):
    """
    Generate random sky coordinates, optionally avoiding the galactic plane.

    Parameters:
    avoid_galactic_plane (bool, optional): Whether to avoid the galactic plane. Default is True.

    Returns:
    tuple: A tuple containing the RA and Dec coordinates.
    """
    if avoid_galactic_plane:
        while True:
            ra = random.uniform(0, 360)
            # Limit dec upper and lower bound to avoid the "galactic plane"
            dec = random.uniform(-60, 60)
            coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            galactic_coords = coords.galactic
            if abs(galactic_coords.b.deg) > 10:  # Avoiding ±10 degrees around the galactic plane
                break
    else:
        ra = random.uniform(0, 360)
        dec = random.uniform(-90, 90)
    
    return ra, dec


def cleanDecValue(dec_value):
    """
    Clean the declination value by removing invalid characters.

    Parameters:
    dec_value (str): The declination value to clean.

    Returns:
    str: The cleaned declination value.
    """
    # Regular expression to keep only valid characters
    valid_chars = re.compile(r'[^0-9+\-dms.]')
    return valid_chars.sub('', dec_value)


def createCircularMask(h, w, center=None, radius=None):
    """
    Create a circular mask.

    Parameters:
    h (int): The height of the mask.
    w (int): The width of the mask.
    center (tuple, optional): The center of the circle. Default is the center of the image.
    radius (int, optional): The radius of the circle. Default is the smallest distance between the center and image walls.

    Returns:
    numpy.ndarray: A boolean array representing the circular mask.
    """
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask


def createStarDataset(catalog_type='II/246', iterations=1, file_path='data/fits/data/', filename='data', pixels=512):
    """
    Create a dataset of star images and pixel masks.

    Parameters:
    catalog_type (str, optional): The catalog type to use. Default is 'II/246'.
    iterations (int, optional): The number of iterations to run. Default is 1.
    filename (str, optional): The base filename for the output files. Default is 'data'.
    pixels (int, optional): The size of the output images in pixels. Default is 512.
    """
    # Create a new directory to store the
    if not os.path.exists('data'):
        os.makedirs('data')


    for i in range(iterations):

        filename_str = filename + str(i)
        file_path = file_path + filename_str + '.fits'
        attempts = 0

        while attempts < 100:
            try:
                ra, dec = getRandomCoordinates()
                coords = SkyCoord(ra, dec, unit='deg', frame='icrs')


                # Fetch image data from SkyView
                image_list = SkyView.get_images(position=coords, survey=['DSS'], radius=0.25 * u.deg, pixels=pixels)

                # Extract the image data from the list
                image_hdu = image_list[0][0]
                image = image_list[0][0].data

                # Extract WCS information from image
                wcs = WCS(image_hdu.header)


                # Fetch star data from Vizier using the 2MASS catalog
                v = Vizier(columns=['*'])
                v.ROW_LIMIT = -1
                catalog_list = v.query_region(coords, radius=0.35 * u.deg, catalog=catalog_type)
                catalog = catalog_list[0]


                # Save the image as a FITS file
                image_hdu = fits.PrimaryHDU(image, header=image_hdu.header)
                hdul = fits.HDUList([image_hdu])
                hdul.writeto(file_path, overwrite=True)


                # Save the star catalog
                with fits.open(file_path, mode='update') as hdul:
                    # Sanitize the header if necessary
                    sanitized_catalog = Table(catalog, meta=sanitizeHeader(catalog.meta))
                    
                    # Create a binary table HDU for the star catalog
                    star_hdu = fits.BinTableHDU(sanitized_catalog, name='STAR_CATALOG')
                    
                    # Append the star catalog HDU to the FITS file
                    hdul.append(star_hdu)
                    hdul.flush()


                coord_range = getCoordRangeFromPixels(wcs)

                # Copy the catalog and convert the table to a pandas DataFrame for easier manipulation
                catalog_df = catalog.copy().to_pandas()


                stars_in_image = getStarsInImage(wcs, catalog_df, coord_range)
                # print("Stars in image: ", stars_in_image)
                print("Number of cataloged stars in image: ", len(stars_in_image))

                
                # Get the pixel coordinates of the first star in the image
                pixel_coords = getPixelCoordsFromStar(stars_in_image[1], wcs)


                # return
                break

            except HTTPError as e:
                if e.code == 404:
                    print(f"HTTP Error 404: Not Found. Generating new coordinates and retrying...")
                    print(f"HTTP Error 404: Not Found. Generating new coordinates and retrying...")
                else:
                    raise e  # Re-raise the exception if it's not a 404 error
            except Exception as e:
                print(f"An error occurred: {e}. Generating new coordinates and retrying...")
                attempts += 1




        x_dim = wcs.pixel_shape[0] # May need to swap x and y dim! (but I think it's right...)
        y_dim = wcs.pixel_shape[1]

        # Pixel-mask of stars
        pixel_mask = np.zeros((x_dim, y_dim))


        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs)

        # Calculate the min and max Jmag values
        jmag_values = catalog_df['Jmag']
        min_jmag = jmag_values.min()
        max_jmag = jmag_values.max()

        # Dynamically determine the min and max radius based on image dimensions
        min_radius = 1  # Minimum radius in pixels
        max_radius = min(x_dim, y_dim) * 0.005859375  # Maximum radius as a fraction of the smaller dimension


        def calculateRadius(jmag, min_jmag, max_jmag, min_radius, max_radius):
            # Normalize the jmag value (inverted)
            normalized_jmag = (max_jmag - jmag) / (max_jmag - min_jmag)
            # Scale the normalized value to the desired range of pixel sizes
            radius = min_radius + (normalized_jmag * (max_radius - min_radius))
            return radius

        for star in stars_in_image: 

            pixel_coords = getPixelCoordsFromStar(star, wcs)
            jmag = star[1]['Jmag']

            # Ensure the pixel coordinates are within bounds
            x, y = int(np.round(pixel_coords[0])), int(np.round(pixel_coords[1]))
            if 0 <= x < x_dim and 0 <= y < y_dim:
                # pixel_mask[x][y] = 1   # Previously was setting the pixel mask to 1 at the star's pixel coordinates

                # Now we will create a circular mask around the star's pixel coordinates
                # Calculate the radius based on a normalized Jmag value
                radius = calculateRadius(jmag, min_jmag, max_jmag, min_radius, max_radius)
                # Create a circular mask
                circular_mask = createCircularMask(x_dim, y_dim, center=(x, y), radius=radius)
                # Update the pixel mask with the circular mask
                pixel_mask[circular_mask] = 1


            Drawing_colored_circle = plt.Circle(( pixel_coords[0] , pixel_coords[1] ), 0.1, fill=False, edgecolor='Blue')
            ax.add_artist( Drawing_colored_circle )
            ax.set_title(f'{filename}.fits')
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            ax.grid(color='white', ls='dotted')

        # Save the plot as an image file
        image_filename = savePlotAsImage(ax, file_path, pixels=pixels)
        
        # Convert the saved image to FITS format and append it to the FITS file
        convertImageToFits(image_filename, file_path, 'star_overlay', pixels=pixels)


        print(f"Saved {filename}{i}.fits with pixel mask and star overlay")
        


        # Save the pixel mask to the FITS file
        with fits.open(file_path, mode='update') as hdul:
            hdu = fits.ImageHDU(pixel_mask, name='pixel_mask')
            hdul.append(hdu)
            hdul.flush()



def sanitizeHeader(header):
    """
    Ensure the header keywords conform to FITS standards.

    Parameters:
    header (dict): The header to sanitize.

    Returns:
    dict: The sanitized header.
    """
    sanitized_header = {}
    for key, value in header.items():
        if len(key) > 8:
            key = key[:8]  # Truncate to 8 characters
        sanitized_header[key] = value
    return sanitized_header


def getCoordRangeFromPixels(wcs):
    """
    Get the range of ICRS coordinates in the image.

    Parameters:
    wcs (astropy.wcs.WCS): The WCS object of the image.

    Returns:
    dict: A dictionary containing the coordinates of the corners of the image.
    """
    x_dim = wcs.pixel_shape[0] # May need to swap x and y dim! (but I think it's right...)
    y_dim = wcs.pixel_shape[1]

    coord_range = {}

    coord_range['lower_left'] = wcs.all_pix2world([0], [0], 1)
    coord_range['lower_right'] = wcs.all_pix2world([x_dim], [0], 1)
    coord_range['upper_left'] = wcs.all_pix2world([0], [y_dim], 1)
    coord_range['upper_right'] = wcs.all_pix2world([x_dim], [y_dim], 1)
    
    return coord_range



def getStarsInImage(wcs, catalog_df, coord_range):
    """
    Get all the stars in the image.

    Parameters:
    wcs (astropy.wcs.WCS): The WCS object of the image.
    catalog_df (pandas.DataFrame): The DataFrame containing the star catalog.
    coord_range (dict): The dictionary containing the coordinates of the corners of the image.

    Returns:
    list: A list of stars in the image.
    """
    # NOTE: X Max and min are reversed for some reason.. orientation of image in coord system...?

    x_max = coord_range['lower_left'][0]
    x_min = coord_range['lower_right'][0]

    y_min = coord_range['lower_left'][1]
    y_max = coord_range['upper_left'][1]

    stars_in_image = []

    print("Number of stars in catalog query: ", len(catalog_df))
    
    for star in catalog_df.iterrows(): 

        # rej = star[1][0]
        # dej = star[1][1]    
        
        # NOTE : Above was causing warning:
        # FutureWarning: Series.getitem treating keys as positions is deprecated. In a future version, 
        # integer keys will always be treated as labels (consistent with DataFrame behavior). 
        # To access a value by position, use ser.iloc[pos] rej = star[1][0] 
        
        rej = star[1].iloc[0]
        dej = star[1].iloc[1]

        if rej < x_max and rej > x_min: 

            # print('Star is in x-coords')

            if dej < y_max and dej > y_min: 

                # Then star is within bounds of image! Add it to a list of stars in the image
                # print('Star is in y-coords')

                stars_in_image.append(star)


    return stars_in_image



def getPixelCoordsFromStar(star, wcs):
    """
    Get the pixel coordinates of a star from the catalog.

    Parameters:
    star (pandas.Series): The Series containing the star data.
    wcs (astropy.wcs.WCS): The WCS object of the image.

    Returns:
    tuple: The pixel coordinates of the star.
    """
    star_coords = star[1]['_2MASS']

    def parseStarCoords(coords):
        """
        Parse the star coordinates.

        Parameters:
        coords (str): The star coordinates.

        Returns:
        str: The parsed star coordinates.
        """
        if '-' in coords:

            rej, dej = coords.split('-')
            rej = rej[0:2] + 'h' + rej[2:4] + 'm' + rej[4:6] + '.' + rej[6:] + 's'
            dej = '-' + dej[0:2] + 'd' + dej[2:4] + 'm' + dej[4:6] + '.' + dej[6:] + 's'

        elif '+' in coords:

            rej, dej = coords.split('+')
            rej = rej[0:2] + 'h' + rej[2:4] + 'm' + rej[4:6] + '.' + rej[6:] + 's'
            dej = '+' + dej[0:2] + 'd' + dej[2:4] + 'm' + dej[4:6] + '.' + dej[6:] + 's'

        # print('COORDS:', rej + ' ' + dej)

        dej = cleanDecValue(dej)  # Clean the declination value

        return rej + dej
    


    # coords = parseStarCoords(star_coords)

    # c = SkyCoord(coords, frame=ICRS)



    # NOTE: The above code was not working when an incorrect value (an 'A' or a 'B') came through from a star_coords:
    # ValueError: Cannot parse first argument data "- 46d40m13.7As" for attribute dec

    # I added a function to clean the declination value inside the parseStarCoords function and a try block to catch the ValueError
    coords = parseStarCoords(star_coords)

    try:
        c = SkyCoord(coords, frame=ICRS)
    except ValueError as e:
        print(f"Error parsing coordinates: {coords}")
        raise e

    pixel_coords = wcs.world_to_pixel(c)
    # print('Pixel Coords:', pixel_coords)
    return pixel_coords


def extractStarCatalog(file_path):
    """
    Extract the star catalog from a FITS file.

    Parameters:
    file_path (str): The path to the FITS file.

    Returns:
    astropy.table.Table: The star catalog.
    """
    with fits.open(file_path) as hdul:
        # Locate the STAR_CATALOG HDU
        star_hdu = hdul['STAR_CATALOG']
        
        # Read the star catalog into an astropy Table
        catalog = Table(star_hdu.data)
    
    return catalog



def displayRawImage(file_path):
    """
    Display the raw image data from a FITS file.

    Parameters:
    file_path (str): The path to the FITS file.
    """
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data

    plt.figure(figsize=(10, 10))
    plt.imshow(image_data, cmap='gray', origin='lower')
    plt.title('Raw Image Data')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.grid(False)
    plt.show()

def displayRawPixelMask(file_path):

    with fits.open(file_path) as hdul:
            image_hdu = hdul[0]
            wcs = WCS(image_hdu.header)

            pixel_mask = hdul['pixel_mask'].data

    plt.figure(figsize=(10, 10))
    plt.imshow(pixel_mask, cmap='gray', origin='lower')
    plt.title('Raw Pixel Mask')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.grid(False)
    plt.show()



def displayImagePlot(file_path):
    """
    Display an astronomical image with WCS projection.

    This function reads a FITS file, extracts the image data and WCS (World Coordinate System)
    information from the header, and displays the image using Matplotlib with WCS projection.

    Parameters:
    file_path (str): The path to the FITS file to be displayed.

    Returns:
    None
    """

    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(image_hdu.header)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs)

        ax.imshow(image_hdu.data, cmap='gray', origin='lower')
        ax.set_title(file_path)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.grid(color='white', ls='dotted')

        plt.show()



def getImagePlot(file_path):
    """
    Generates a plot of an astronomical image with WCS (World Coordinate System) projection.

    Parameters:
    file_path (str): The path to the FITS file containing the astronomical image.

    Returns:
    tuple: A tuple containing the matplotlib figure and axis objects.
    """
    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(image_hdu.header)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs)

        ax.imshow(image_hdu.data, cmap='gray', origin='lower')
        ax.set_title(file_path)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.grid(color='white', ls='dotted')

        return fig, ax
    

def extractImageArray(file_path):
    """
    Extracts the image array from a FITS file.

    Parameters:
    file_path (str): The path to the FITS file.

    Returns:
    numpy.ndarray: The image data array extracted from the FITS file.
    """

    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]

        image = image_hdu.data

        return image
    

def displayPixelMaskPlot(file_path):
    """
    Displays a plot of the pixel mask from a FITS file with WCS projection.

    Parameters:
    file_path (str): The path to the FITS file containing the pixel mask data.

    Returns:
    None
    """

    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(image_hdu.header)

        pixel_mask = hdul['pixel_mask'].data

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs)

        ax.imshow(pixel_mask, cmap='gray', origin='lower')
        ax.set_title(file_path)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.grid(color='white', ls='dotted')

        plt.show()


def getPixelMaskPlot(file_path):
    """
    Generates a plot of the pixel mask from a FITS file.
    Parameters:
    file_path (str): The path to the FITS file containing the pixel mask.
    Returns:
    tuple: A tuple containing the matplotlib figure and axis objects.
    """
    

    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(image_hdu.header)

        pixel_mask = hdul['pixel_mask'].data

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs)

        ax.imshow(pixel_mask, cmap='gray', origin='lower')
        ax.set_title(file_path)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.grid(color='white', ls='dotted')

        return fig, ax
    

def extractPixelMaskArray(file_path):
    """
    Extracts the pixel mask array from a FITS file.
    Parameters:
    file_path (str): The path to the FITS file.
    Returns:
    numpy.ndarray: The pixel mask array extracted from the FITS file.
    """
    

    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(image_hdu.header)

        pixel_mask = hdul['pixel_mask'].data

        return pixel_mask
    

def displayPixelMaskOverlayPlot(file_path, catalog='II/246'):
    """
    Displays a pixel mask overlay plot of stars from a given FITS file and star catalog.

    Parameters:
    file_path (str): The path to the FITS file containing the image data.
    catalog (str, optional): The star catalog identifier. Default is 'II/246'.

    Returns:
    None

    This function performs the following steps:
    1. Opens the FITS file and extracts the image data and WCS (World Coordinate System) information.
    2. Determines the coordinate range from the image pixels.
    3. Extracts the star catalog data and converts it to a pandas DataFrame.
    4. Identifies the stars within the image based on the WCS and coordinate range.
    5. Creates a pixel mask of the stars and overlays it on the image.
    6. Plots the image with the pixel mask and star positions marked with circles.

    Example usage:
    displayPixelMaskOverlayPlot('data/star0.fits', catalog='II/246')
    """

    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(image_hdu.header)

        coord_range = getCoordRangeFromPixels(wcs)

        catalog = extractStarCatalog(file_path)

        # Convert the table to a pandas DataFrame for easier manipulation
        catalog_df = catalog.to_pandas()

        stars_in_image = getStarsInImage(wcs, catalog_df, coord_range)
        print("Number of cataloged stars in image: ", len(stars_in_image))

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs)

        x_dim = wcs.pixel_shape[0]
        y_dim = wcs.pixel_shape[1]

        # Pixel-mask of stars
        pixel_mask = np.zeros((x_dim, y_dim))

        print('Drawing')  # DEBUG

        for star in stars_in_image:
            pixel_coords = getPixelCoordsFromStar(star, wcs)

            x, y = int(np.round(pixel_coords[0])), int(np.round(pixel_coords[1]))
            if 0 <= x < x_dim and 0 <= y < y_dim:
                pixel_mask[x][y] = 1

            Drawing_colored_circle = plt.Circle((pixel_coords[0], pixel_coords[1]), 2, fill=False, edgecolor='Blue')
            ax.add_artist(Drawing_colored_circle)

        ax.set_title(f'{file_path}')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.grid(color='white', ls='dotted')

        ax.imshow(image_hdu.data, cmap='gray', origin='lower')
        plt.show()


def getPixelMaskOverlayPlot(file_path, catalog='II/246'):
    """
    Generates a plot overlaying a pixel mask of stars on a FITS image.

    Parameters:
    file_path (str): The path to the FITS file.
    catalog (str): The star catalog identifier. Default is 'II/246'.

    Returns:
    tuple: A tuple containing:
        - fig (matplotlib.figure.Figure): The figure object of the plot.
        - ax (matplotlib.axes._subplots.AxesSubplot): The axes object of the plot.
        - stars_in_image (pandas.DataFrame): DataFrame containing the stars in the image.
        - wcs (astropy.wcs.WCS): The WCS (World Coordinate System) object of the image.
    """

    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(image_hdu.header)

        coord_range = getCoordRangeFromPixels(wcs)

        catalog = extractStarCatalog(file_path)

        # Convert the table to a pandas DataFrame for easier manipulation
        catalog_df = catalog.to_pandas()

        stars_in_image = getStarsInImage(wcs, catalog_df, coord_range)
        print("Number of cataloged stars in image: ", len(stars_in_image))

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs)

        x_dim = wcs.pixel_shape[0]
        y_dim = wcs.pixel_shape[1]

        # Pixel-mask of stars
        pixel_mask = np.zeros((x_dim, y_dim))

        print('Drawing')  # DEBUG

        for star in stars_in_image:
            pixel_coords = getPixelCoordsFromStar(star, wcs)
            # pixel_mask[int(np.round(pixel_coords[0]))][int(np.round(pixel_coords[1]))] = 1
            # Ensure the pixel coordinates are within bounds
            x, y = int(np.round(pixel_coords[0])), int(np.round(pixel_coords[1]))
            if 0 <= x < x_dim and 0 <= y < y_dim:
                pixel_mask[x][y] = 1

            Drawing_colored_circle = plt.Circle((pixel_coords[0], pixel_coords[1]), 2, fill=False, edgecolor='Blue')
            ax.add_artist(Drawing_colored_circle)

        ax.set_title(f'{file_path}')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.grid(color='white', ls='dotted')

        ax.imshow(image_hdu.data, cmap='gray', origin='lower')

        return fig, ax, stars_in_image, wcs


def saveFitsImages(filename, file_path, catalog_type='II/246'):
    """
    Save FITS images with WCS projection and overlay star catalog information.

    Parameters:
    filename (str): The name of the FITS file to be processed.
    file_path (str): The path where the FITS file is located. If None, defaults to 'data/fits/'.
    catalog_type (str, optional): The type of star catalog to use for overlay. Default is 'II/246'.

    Returns:
    None

    This function performs the following steps:
    1. Opens the FITS file and extracts the primary image data and WCS information.
    2. Creates a matplotlib figure with three subplots:
        - The original image.
        - The pixel mask (if available).
        - The original image with an overlay of cataloged stars.
    3. Saves the combined image as a PNG file in the specified file path.
    """

    plt.style.use(astropy_mpl_style)

    if file_path is None:
        file_path = 'data/fits/' + filename
    # else:
        # file_path = file_path + filename


    with fits.open(file_path + filename) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(hdul[0].header)

        fig, axs = plt.subplots(1, 3, figsize=(21, 7), subplot_kw={'projection': wcs})

        # Get the first image (Original Image)
        axs[0].imshow(image_hdu.data, cmap='gray', origin='lower')
        axs[0].set_title('Original Image')
        axs[0].set_xlabel('RA')
        axs[0].set_ylabel('Dec')
        axs[0].grid(color='white', ls='dotted')


        # Get the second image (Pixel Mask)
        if 'pixel_mask' in hdul:
            pixel_mask_hdu = hdul['pixel_mask']
            axs[1].imshow(pixel_mask_hdu.data, cmap='gray', origin='lower')
        else:
            axs[1].text(0.5, 0.5, 'No Pixel Mask', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].set_title('Pixel Mask')
        axs[1].set_xlabel('RA')
        axs[1].set_ylabel('Dec')
        axs[1].grid(color='white', ls='dotted')


        # Get the third image (Pixel Mask Overlay)
        coord_range = getCoordRangeFromPixels(wcs)
        catalog = extractStarCatalog(file_path + filename)
        catalog_df = catalog.to_pandas()
        stars_in_image = getStarsInImage(wcs, catalog_df, coord_range)
        print("Number of cataloged stars in image: ", len(stars_in_image))

        axs[2].imshow(image_hdu.data, cmap='gray', origin='lower')
        for star in stars_in_image:
            pixel_coords = getPixelCoordsFromStar(star, wcs)
            x, y = int(np.round(pixel_coords[0])), int(np.round(pixel_coords[1]))
            if 0 <= x < wcs.pixel_shape[0] and 0 <= y < wcs.pixel_shape[1]:
                Drawing_colored_circle = plt.Circle((pixel_coords[0], pixel_coords[1]), 2, fill=False, edgecolor='Blue')
                axs[2].add_artist(Drawing_colored_circle)
        axs[2].set_title('Pixel Mask Overlay')
        axs[2].set_xlabel('RA')
        axs[2].set_ylabel('Dec')
        axs[2].grid(color='white', ls='dotted')

        # Save the combined image
        plt.tight_layout()
        image_filename = filename.replace('.fits', '.png')
        image_path = file_path + image_filename
        plt.savefig(image_path)
        plt.show()



def getFitsContents(file_path):
    """
    Opens a FITS file and returns its content information.

    Parameters:
    file_path (str): The path to the FITS file.

    Returns:
    None: Prints the information of the FITS file content.
    """
    FITS_content = fits.open(file_path)
    return FITS_content.info()


def getFitsHeader(file_path):
        """
        Extracts the header from a FITS file.

        Parameters:
        file_path (str): The path to the FITS file.

        Returns:
        dict: A dictionary containing the header information from the FITS file.
        """
        with fits.open(file_path) as hdul:
            header = hdul[0].header
            header = dict(header)
            return header



def getStarTable(file_path):
    """
    Extracts the star catalog table from a FITS file.

    Parameters:
    file_path (str): The path to the FITS file containing the star catalog.

    Returns:
    astropy.table.Table: A table containing the star catalog data.
    """
    with fits.open(file_path) as hdul:
        star_catalog = hdul['STAR_CATALOG'].data
        star_table = Table(star_catalog)
        return star_table

            


def importDataset(dataset_path='data/fits/data/'):
    """
    Imports a dataset of FITS files from the specified directory, extracting image arrays, pixel masks, WCS data, 
    and star data from each file.

    Args:
        dataset_path (str): The path to the directory containing the FITS files. Default is 'data/fits/data/'.

    Returns:
        tuple: A tuple containing the following elements:
            - images (list): A list of image arrays extracted from the FITS files.
            - masks (list): A list of pixel mask arrays extracted from the FITS files.
            - stars_in_image (list): A list of star data for each image, extracted from the FITS files.
            - wcs_data (list): A list of WCS (World Coordinate System) data extracted from the FITS files.
            - fits_files (list): A list of the FITS files found in the dataset directory.
    """
    # Create images and masks arrays lists
    images = []
    masks = []

    # Create a list of all the wcs data in the dataset folder
    wcs_data = []

    # Create an array to store the star data inside each fits file
    stars_in_image = []

    # Create a list of all the fits files in the dataset folder
    fits_files = []

    # For all the fits files in the dataset folder specified in file_path, extract the image and mask arrays to the respective lists
    file_path = dataset_path
    # for file in os.listdir(file_path):
    for file in os.listdir(file_path):
        if file.endswith('.png'):
            os.remove(file_path + file)
        if file.endswith('.fits'):
            fits_files.append(file)
            images.append(extractImageArray(file_path + file))
            masks.append(extractPixelMaskArray(file_path + file))
            wcs = wcs_data.append(WCS(fits.open(file_path + file)[0].header))
            wcs_data.append(WCS(fits.open(file_path + file)[0].header))
            stars_in_image.append(getStarsInImage(wcs_data[-1], extractStarCatalog(file_path + file).to_pandas(), getCoordRangeFromPixels(wcs_data[-1])))

            print(file + ' added to dataset')

    return images, masks, stars_in_image, wcs_data, fits_files

# Extract stars from fits file
def extractStarsFromFits(fits_file):
    """
    Extracts star data from a FITS file.

    This function opens a FITS file, extracts the World Coordinate System (WCS) information
    from the header, and retrieves star data within the image using the WCS and a star catalog.

    Args:
        fits_file (str): The path to the FITS file.

    Returns:
        pandas.DataFrame: A DataFrame containing the star data extracted from the FITS file.
    """
    with fits.open(fits_file) as hdul:
        wcs = WCS(hdul[0].header)
        stars = getStarsInImage(wcs, extractStarCatalog(fits_file).to_pandas(), getCoordRangeFromPixels(WCS(fits.open(fits_file)[0].header)))
    return stars