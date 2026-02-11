# WCS to Pixel Conversion Reference

This document describes **where** and **how** WCS (World Coordinate System) ↔ pixel conversions are done in this project, so you can repeat the process to align WCS from one satellite/image to another.

---

## 1. Where the conversions happen

| Conversion | File | Function | Purpose |
|------------|------|----------|---------|
| **World → Pixel** | `scripts/dataGathering.py` | `getPixelCoordsFromStar(star, wcs)` | Convert RA/Dec (catalog) → (x, y) pixel |
| **Pixel → World** | `scripts/dataGathering.py` | `getCoordRangeFromPixels(wcs)` | Convert image corners (pixels) → RA/Dec |
| **Extract WCS** | `scripts/imageProcessing.py` | `extractWCSFromFits(fits_file)` | Get WCS from FITS primary header |
| **Extract WCS** | `scripts/dataGathering.py` | inline `WCS(image_hdu.header)` | Same when building from HDU |

`finalDemo/dataGathering.py` is a copy of the scripts version and uses the same logic.

---

## 2. Exact conversion flow

### 2.1 Get a WCS object from a FITS image

```python
from astropy.wcs import WCS
from astropy.io import fits

# From file path (imageProcessing.py style)
with fits.open(fits_file) as hdul:
    wcs = WCS(hdul[0].header)

# Or from an already-open HDU (dataGathering.py style)
wcs = WCS(image_hdu.header)
```

The WCS is built from the **primary HDU header** (`hdul[0].header`). That header must contain the standard FITS WCS keywords (e.g. `CTYPE`, `CRVAL`, `CRPIX`, `CD` or `CDELT`, etc.) for the conversion to be valid.

### 2.2 World (RA/Dec) → Pixel

Used to find where a sky position (RA, Dec) falls in the image.

**Location:** `scripts/dataGathering.py`, `getPixelCoordsFromStar()` (lines ~428–491).

**Steps:**

1. Express the sky position as an **astropy** `SkyCoord` in **ICRS** (RA/Dec):
   ```python
   from astropy.coordinates import SkyCoord
   from astropy.coordinates import ICRS

   # Example: RA/Dec in degrees
   c = SkyCoord(ra=ra_deg, dec=dec_deg, unit='deg', frame='icrs')
   # Or from a string like "12h34m56.7s -45d40m13.7s"
   c = SkyCoord(coords_string, frame=ICRS)
   ```
2. Convert to pixel using the **image’s** WCS:
   ```python
   pixel_coords = wcs.world_to_pixel(c)  # (x, y) in pixel units
   ```
3. Use `wcs.pixel_shape` for image size when checking bounds:
   ```python
   x_dim, y_dim = wcs.pixel_shape[0], wcs.pixel_shape[1]
   ```

So: **same sky position (RA/Dec) + different WCS → different (x, y)**. That is exactly what you use to align “WCS from one satellite picture to another”: one image gives WCS₁, the other WCS₂; you convert the same RA/Dec with both to get pixel coords in each image.

### 2.3 Pixel → World (image corners / any pixel)

Used to get the sky coordinate range of the image (e.g. to query a catalog or to know the footprint).

**Location:** `scripts/dataGathering.py`, `getCoordRangeFromPixels()` (lines ~351–371).

**Code (concept):**

```python
x_dim = wcs.pixel_shape[0]
y_dim = wcs.pixel_shape[1]

# Corners: (x, y) in pixel → (RA, Dec); the "1" is "origin" (1-based vs 0-based)
coord_range['lower_left']  = wcs.all_pix2world([0], [0], 1)
coord_range['lower_right'] = wcs.all_pix2world([x_dim], [0], 1)
coord_range['upper_left']  = wcs.all_pix2world([0], [y_dim], 1)
coord_range['upper_right'] = wcs.all_pix2world([x_dim], [y_dim], 1)
```

So: **pixel (x, y) + WCS → (RA, Dec)**. Astropy’s WCS uses 0-based pixel indices in `world_to_pixel`; `all_pix2world(..., 1)` uses 1-based (FITS convention). Be consistent with origin when you copy this into a new project.

---

## 3. Aligning WCS from one satellite image to another

To align by WCS:

1. **Load WCS for each image** from its FITS (or other) header:
   - Image A: `wcs_a = WCS(header_a)`
   - Image B: `wcs_b = WCS(header_b)`

2. **Choose reference positions in sky coordinates** (RA, Dec), e.g.:
   - from a star catalog, or
   - by picking pixels in one image and converting to sky:  
     `ra, dec = wcs_a.all_pix2world([x], [y], 1)` then use that (RA, Dec) for both.

3. **Convert the same (RA, Dec) to pixels in each image:**
   ```python
   from astropy.coordinates import SkyCoord

   c = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
   x_a, y_a = wcs_a.world_to_pixel(c)
   x_b, y_b = wcs_b.world_to_pixel(c)
   ```
   So (x_a, y_a) is the position in image A and (x_b, y_b) in image B for the **same** sky position.

4. **Use (x_a, y_a) and (x_b, y_b)** for:
   - overlays,
   - resampling image B onto image A’s grid (or vice versa),
   - or computing a world-to-pixel transform (e.g. for reprojection).

So the **only** project-specific conversion you need to repeat is: **WCS from header → `SkyCoord` (RA/Dec) → `wcs.world_to_pixel(c)`** (and, if needed, `all_pix2world` for pixel → RA/Dec). The rest is standard astropy.

---

## 4. Dependencies

```python
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord, ICRS
```

Optional: `astropy.units` for `u.deg` if you pass angles in degrees.

---

## 5. File reference

- **World → Pixel:** `scripts/dataGathering.py` — `getPixelCoordsFromStar()` (uses `SkyCoord` + `wcs.world_to_pixel()`).
- **Pixel → World:** `scripts/dataGathering.py` — `getCoordRangeFromPixels()` (uses `wcs.all_pix2world()`).
- **Extract WCS from FITS:** `scripts/imageProcessing.py` — `extractWCSFromFits()`; same idea inline in `scripts/dataGathering.py` and `finalDemo/dataGathering.py`.

Using the same (RA, Dec) with two different WCSs (from two different satellite images) and calling `world_to_pixel` for each gives you aligned pixel positions for that sky position in both images.
