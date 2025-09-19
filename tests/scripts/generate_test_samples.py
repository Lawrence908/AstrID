from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from astropy.io import fits  # type: ignore

    ASTROPY_AVAILABLE = True
except Exception:  # pragma: no cover
    ASTROPY_AVAILABLE = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def gen_numpy_samples(base: Path) -> None:
    np.random.seed(42)
    out = base / "npy"
    ensure_dir(out)

    # A tiny 64x64 image with two Gaussian sources
    img = np.random.normal(1000, 30, (64, 64)).astype(np.float32)
    yy, xx = np.mgrid[:64, :64]
    img += 5000 * np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / (2 * 3**2))
    img += 3500 * np.exp(-((xx - 45) ** 2 + (yy - 40) ** 2) / (2 * 2**2))

    np.save(out / "sample_image_64.npy", img)

    # A tiny difference image
    diff = np.random.normal(0, 5, (64, 64)).astype(np.float32)
    diff += 300 * np.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / (2 * 2**2))
    np.save(out / "sample_diff_64.npy", diff)


def gen_fits_samples(base: Path) -> None:
    if not ASTROPY_AVAILABLE:
        return

    out = base / "fits"
    ensure_dir(out)

    np.random.seed(7)
    data = np.random.normal(1000, 25, (64, 64)).astype(np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.header["OBJECT"] = "Test Object"
    hdu.header["TELESCOP"] = "Test Telescope"
    hdu.header["INSTRUME"] = "Test Instrument"
    hdu.header["FILTER"] = "g"
    hdu.header["EXPTIME"] = 60.0
    hdu.header["RA"] = 180.0
    hdu.header["DEC"] = 45.0

    (out / "sample_64.fits").unlink(missing_ok=True)
    hdu.writeto(out / "sample_64.fits")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tests/data", help="Output directory root")
    args = parser.parse_args()

    base = Path(args.out)
    ensure_dir(base)

    gen_numpy_samples(base)
    gen_fits_samples(base)

    print(f"Samples written under {base}")


if __name__ == "__main__":
    main()
