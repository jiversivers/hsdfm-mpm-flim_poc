import gc
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from hsdfmpm.hsdfm.utils import find_cycles, gabor_filter_bank, naive_leastsq_reflectance
from hsdfmpm.hsdfm.fit import reduced_chi_squared, fit_volume
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

# Dir stuff
root = Path(r'D:\Jesse\hsdfmpm_poc')
processed = root / 'Processed'
processed.mkdir(exist_ok=True)

# Choose fitting wavelengths
wavelengths = np.arange(500, 730, 10)

# Load normalization data
standard_paths = find_cycles(root / 'Standards')
standard = MergedHyperspectralImage(image_paths=standard_paths, wavelengths=wavelengths, scalar=0.8)
standard.normalize_integration_time()

background = MergedHyperspectralImage(image_paths=find_cycles(root / 'Background'), wavelengths=wavelengths)
background.normalize_integration_time()

# Detach and garbage collect big objects
standard = standard.image.copy()
background = background.image.copy()
gc.collect()

# Get MCLUT
lut = LUT(dimensions=['mu_s', 'mu_a'], scale=50000, extrapolate=True, simulation_id=88)

# Prep filter bank
f = np.geomspace(0.01, 1, 16)
gabor_bank = gabor_filter_bank(frequency=f, sigma_x=4 / f, sigma_y=1 / f)

def parse_categorical(cycle):
    # Prepare categorical data
    animal = cycle.parts[4]
    date = datetime.strptime(cycle.parts[5], '%m%d%Y')
    oxygen = cycle.parts[6]
    fov = cycle.parts[7]
    return animal, date, oxygen, fov


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
    mu_s /= 0.1
    r = lut(mu_s, mu_a, extrapolate=True)
    return r


def process_cycle(cycle, skip_today=False):
    try:
        # Parse categorical data
        animal, date, oxygen, fov = parse_categorical(cycle)
        out_path = processed / animal / datetime.strftime(date,'%m%d%Y') / oxygen / fov
        out_path.mkdir(exist_ok=True, parents=True)

        if skip_today and (out_path / 'scatter_a.tiff').exists() and datetime.fromtimestamp((out_path / 'scatter_a.tiff').stat().st_mtime).date() == datetime.today().date():
            print(f'Skipping cycle {cycle}')
            mask = cv2.imread(out_path / 'hsdfm_mask.tiff', cv2.IMREAD_UNCHANGED)
            a = cv2.imread(out_path / 'scatter_a.tiff', cv2.IMREAD_UNCHANGED)
            b = cv2.imread(out_path / 'scatter_b.tiff', cv2.IMREAD_UNCHANGED)
            thb = cv2.imread(out_path / 'thb.tiff', cv2.IMREAD_UNCHANGED)
            so2 = cv2.imread(out_path / 'so2.tiff', cv2.IMREAD_UNCHANGED)

            # Update output
            output = [
                animal, date, oxygen, fov,
                np.mean(a[mask]), np.mean(b[mask]), np.mean(thb[mask]), np.mean(so2[mask]), str(cycle)
            ]

            return output

        # Load cycle subset for naive fitting to mask
        hs = HyperspectralImage(image_path=cycle, wavelengths=wavelengths)

        # Normalize (automatically normalizes to integration time then the standard/background (manually, for memory)
        hs.normalize_integration_time()
        hs._active = (hs - background) / (standard - background)

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
        mask = cv2.adaptiveThreshold((255 * gabor_response / gabor_response.max()).astype(np.uint8), 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
        mask = mask.astype(bool)
        hs.apply_mask(mask)
        iio.imwrite(out_path / 'gabor_response.tiff', gabor_response)
        iio.imwrite(out_path / 'hb_index.tiff', hb_index)
        fig, ax = plt.subplots()
        ax.plot(np.nanmean(hs, axis=(1, 2)))

        # Fit image
        hs.superset()  # Reset to full spectrum
        img = hs.image.copy()  # Detach from object to save memory (no need to have standard and raw data persist)
        del hs
        gc.collect()
        param_image, chi_sq = fit_volume(
            volume=img,
            model=model,
            x0=[0.5, 0.5, 0.5, 0.5],
            bounds=[(0, 0, 0, 0), (np.inf, np.inf, np.inf, 1)],
            use_multiprocessing=False,
            score_function=reduced_chi_squared,
            max_nfev=5000)
        a, b, thb, so2, = param_image

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
        output = [
            animal, date, oxygen, fov,
            np.mean(a[mask]), np.mean(b[mask]), np.mean(thb[mask]), np.mean(so2[mask]), str(cycle)
                  ]
        iio.imwrite(out_path / 'hsdfm_mask.tiff', mask)
        iio.imwrite(out_path / 'scatter_a.tiff', a)
        iio.imwrite(out_path / 'scatter_b.tiff', b)
        iio.imwrite(out_path / 'thb.tiff', thb)
        iio.imwrite(out_path / 'so2.tiff', so2)

        iio.imwrite(out_path / 'hsdfm_chi_sq.tiff', chi_sq)

        return output
    except Exception as e:
        print(f'{e}\non\n{cycle}')
        gc.collect()
        return [
        animal, date, oxygen, fov,
        np.nan, np.nan, np.nan, np.nan, str(cycle)
        ]


if __name__ == '__main__':
    # Find raw data
    cycles = find_cycles(root / 'Animals')
    # pool = Pool(19)
    output = []
    try:
        for out in tqdm.tqdm(map(process_cycle, cycles), total=len(cycles)):
            output.append(out)
    finally:
        # pool.terminate()
        # pool.close()
        # pool.join()
        # del pool
        gc.collect()

    df = pd.DataFrame(output,
                      columns=['Animal', 'Date', 'Oxygen', 'FOV', 'Mean Scatter A', 'Mean Scatter B',
                               'Mean THb', 'Mean sO2', 'Full data path'])
    df.to_csv(processed / 'hsdfm_output.csv')
