from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from hsdfmpm.hsdfm.utils import find_cycles, gabor_filter_bank, naive_leastsq_reflectance
from hsdfmpm.hsdfm.fit import reduced_chi_squared
from hsdfmpm.utils import apply_kernel_bank
import pandas as pd
from hsdfmpm.hsdfm import HyperspectralImage, MergedHyperspectralImage
from pathlib import Path
from hsdfmpm.utils import truncate_colormap, colorize
from matplotlib.colors import Normalize

from photon_canon.contrib.bio import hemoglobin_mus
from photon_canon.lut import LUT
from photon_canon.contrib.bio import wl, eps
import cv2
import imageio.v3 as iio
from datetime import datetime, timedelta
import warnings

from scipy.optimize import least_squares
from hsdfmpm.hsdfm.fit import make_residual

# Dir stuff
root = Path(r'E:\new df\POC Study')
processed = root / 'Processed'
processed.mkdir(exist_ok=True)

# Find raw data
cycles = find_cycles(root / 'Animals')

# Choose fitting wavelengths
wavelengths = np.arange(500, 730, 10)

# Load normalization data
standard_paths = find_cycles(root / 'Standards')
background = MergedHyperspectralImage(image_paths=find_cycles(root / 'Background' / 'dark_04282025'), wavelengths=wavelengths)
background.normalize()

lut = LUT(dimensions=['mu_s', 'mu_a'], scale=50000)

def parse_categorical(cycle):
    # Prepare categorical data
    animal = cycle.parts[4]
    date = datetime.strptime(cycle.parts[5], '%m%d%Y')
    oxygen = cycle.parts[6]
    fov = cycle.parts[7]
    polarized = 'polar' in str(cycle) and not 'unp' in str(cycle)
    return animal, date, oxygen, fov, polarized


def find_dated_dir(date, paths):
    matches = []
    for path in paths:
        if date.strftime('%m%d%Y') in str(path):
            matches.append(path)
    if not matches:
        warnings.warn(f'No paths found for date {date}. Auto-incrementing ahead one day', category=RuntimeWarning, stacklevel=2)
        date += timedelta(days=1)
        return find_dated_dir(date, paths)
    return matches


def model(a, b, t, s):
    mu_s, mu_a, _ = hemoglobin_mus(a, b, t, s, wavelengths, force_feasible=False)
    r = lut(mu_s, mu_a, extrapolate=True)
    return r


def main():
    # Prep reusable parts
    f = np.geomspace(0.01, 1, 16)
    gabor_bank = gabor_filter_bank(frequency=f, sigma_x=4 / f, sigma_y=1 / f)

    output = []

    for cycle in tqdm.tqdm(cycles):
        # Parse categorical data
        animal, date, oxygen, fov, polarization = parse_categorical(cycle)
        out_path = processed / animal / datetime.strftime(date,
                                                          '%m%d%Y') / oxygen / fov / f'{'polarized' if polarization else 'nonpolarized'}'
        out_path.mkdir(exist_ok=True, parents=True)

        # Load standard
        standard_path = find_dated_dir(date, standard_paths)
        if polarization:
            standard_path = [path for path in standard_path if 'polar' in str(path) and 'unpolar' not in str(path)]
        else:
            standard_path = [path for path in standard_path if 'polar' not in str(path) or 'unpolar' in str(path)]

        standard = MergedHyperspectralImage(image_paths=standard_path, wavelengths=wavelengths, scalar=0.8)
        standard.normalize()

        # Load cycle subset for naive fitting to mask
        hs = HyperspectralImage(image_path=cycle, wavelengths=wavelengths, standard=standard, background=background)

        # Normalize (automatically normalizes to integration time then the standard/background
        hs.normalize()

        # Resize to 256x256
        hs.resize_to(256)

        # Get variables for naive fit
        sub_wl = np.arange(500, 610, 10)
        hs.subset_by_metadata('Wavelength', sub_wl)  # Subset to Hb dominated spectrum
        selected = np.isin(wl, sub_wl)
        e = eps[:, selected]

        # Fit the image with naive lsq and apply gabor bank
        naive_fit = naive_leastsq_reflectance(hs.image, e)
        hb_index = naive_fit[0] + naive_fit[1]
        gabor_response = apply_kernel_bank(hb_index, gabor_bank)

        # Create and mask
        mask = np.logical_and(
            cv2.adaptiveThreshold((255 * gabor_response / gabor_response.max()).astype(np.uint8), 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0).astype(bool),
            (gabor_response > 0.1).astype(bool)
        )
        mask = mask.astype(bool)
        hs.apply_mask(mask)
        iio.imwrite(out_path / 'gabor_response.tiff', gabor_response)
        iio.imwrite(out_path / 'hb_index.tiff', hb_index)

        # Fit image
        hs.superset()  # Reset to full spectrum
        param_image, chi_sq = hs.fit(
            model,
            x0=[1, 1, 1, 1],
            bounds=[(0, 0, 0, 0), (np.inf, np.inf, np.inf, 1)],
            n_workers=15,
            score_function=reduced_chi_squared,
            max_nfev=5000)
        a, b, thb, so2, = param_image
        # mask = np.logical_and(mask, chi_sq < 1.5)

        # Create color maps
        cmap = truncate_colormap('jet', cmin=0.13, cmax=0.88)
        for png_name, im in zip(['a', 'b', 'thb', 'so2'], [a, b, thb, so2]):
            im[~mask] = 0
            cmin, cmax = np.nanmean(im[mask]) + np.array([-2, 2]) * np.nanstd(im[mask])
            cmin = max(0, cmin)
            color_var, cmap = colorize(im, hb_index * mask, cmap=cmap, cmin=cmin, cmax=cmax)
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(color_var, cmap=cmap)
            ax = plt.gca()
            ax.set_title(f'{animal} {date} {oxygen} {fov}')
            ax.axis('off')
            sm = plt.cm.ScalarMappable(norm=Normalize(vmin=cmin, vmax=cmax), cmap=cmap)
            plt.colorbar(sm, ax=ax)
            plt.tight_layout()
            fig.savefig(out_path / f'hsdfm_color_{png_name}.png')
            plt.close(fig)

        # Update output
        output.append([animal, date, oxygen, fov, polarization,
                       np.mean(a[mask]), np.mean(b[mask]), np.mean(thb[mask]), np.mean(so2[mask]), str(cycle)
                       ])
        iio.imwrite(out_path / 'hsdfm_mask.tiff', mask)
        iio.imwrite(out_path / 'scatter_a.tiff', a)
        iio.imwrite(out_path / 'scatter_b.tiff', b)
        iio.imwrite(out_path / 'thb.tiff', thb)
        iio.imwrite(out_path / 'so2.tiff', so2)

        iio.imwrite(out_path / 'hsdfm_chi_sq.tiff', chi_sq)

    df = pd.DataFrame(output,
                      columns=['Animal', 'Date', 'Oxygen', 'FOV', 'Polarization', 'Mean Scatter A', 'Mean Scatter B',
                               'Mean THb', 'Mean sO2', 'Full data path'])
    df.to_csv(processed / 'hsdfm_output.csv')

if __name__ == '__main__':
    main()
