import gc
from functools import partial
import tifffile as tf
from multiprocessing import Pool
from multiprocessing.spawn import freeze_support

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from hsdfmpm.hsdfm.utils import find_cycles, gabor_filter_bank, naive_leastsq_reflectance
from hsdfmpm.hsdfm.fit import reduced_chi_squared, fit_volume
from hsdfmpm.utils import apply_kernel_bank
import pandas as pd
from hsdfmpm.hsdfm import HyperspectralImage, MergedHyperspectralImage
from hsdfmpm.utils import bin
from pathlib import Path
from hsdfmpm.utils import truncate_colormap, colorize
from matplotlib.colors import Normalize
from photon_canon.contrib.bio import hemoglobin_mus
from photon_canon.lut import LUT
from photon_canon.contrib.bio import wl, eps
import cv2
from datetime import datetime, timedelta
import warnings


def parse_categorical(cycle):
    # Prepare categorical data
    animal = cycle.parts[6]
    date = datetime.strptime(cycle.parts[4], '%m%d%Y')
    oxygen = cycle.parts[7]
    fov = cycle.parts[8]
    root = Path(*cycle.parts[:4])
    return animal, date, oxygen, fov, root


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


def process_cycle(args):
    cycle, out_path, wavelengths, lut, sca_idx, abs_idx = args
    out_path = Path(out_path)
    animal, date, oxygen, fov, root = parse_categorical(cycle)

    # Define an offset only mode to fit in non-absorptive wavelengths
    mu_s_sca, _, _ = hemoglobin_mus(10, 1, 0, 0, wavelengths[sca_idx], force_feasible=False)
    r = lut(10 * mu_s_sca, 0)
    def offset_model(c):
        return r + c

    # Define the vascular model
    def abs_model(t, s):
        mu_s, mu_a, _ = hemoglobin_mus(10, 1, t, s, wavelengths[abs_idx], force_feasible=False)
        return lut(10 * mu_s, mu_a, extrapolate=True)

    try:
        # Load cycle subset for naive fitting to mask
        img = cv2.imread(out_path / 'hsdfm_norm.tiff', cv2.IMREAD_UNCHANGED)
        hb_index = cv2.imread(out_path / 'hb_index.tiff', cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(out_path / 'pre_mask.tiff', cv2.IMREAD_UNCHANGED)
        img = img[np.newaxis, ...]
        img = bin(img, bin_factor=8)

        # Fit the offset model in the scatter dominated wavelengths
        sca_img = img[sca_idx]
        param_image, chi_sq = fit_volume(
            volume=sca_img,
            model=offset_model,
            x0=[0],
            use_multiprocessing=False,
            score_function=reduced_chi_squared,
            max_nfev=5000,
            pbar=True
        )
        c = param_image


        # Fit the vascular-space with the absorption model in the absorption dominated wavelengths
        img[:, mask] = np.nan  # Mask the vasculature
        img = img[abs_idx] - c  # Apply the offset
        param_image, chi_sq = fit_volume(
            volume=img,
            model=abs_model,
            x0=[0.5, 0.5],
            bounds=[(0, 0), (np.inf, 1)],
            use_multiprocessing=False,
            score_function=reduced_chi_squared,
            max_nfev=5000,
            pbar=True
        )
        thb, so2 = param_image
        available_o2 = 4 * thb * so2

        # Create color maps
        cmap = truncate_colormap('jet', cmin=0.13, cmax=0.88)
        for png_name, im in zip(['thb', 'so2', 'offset', 'available_o2'], [thb, so2, c, available_o2]):
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
            np.mean(thb[mask]), np.mean(so2[mask]), np.mean(c), str(cycle)
                  ]
        tf.imwrite(out_path / 'hsdfm_mask.tiff', mask)
        tf.imwrite(out_path / 'thb.tiff', thb)
        tf.imwrite(out_path / 'so2.tiff', so2)
        tf.imwrite(out_path / 'c.tiff', c)
        tf.imwrite(out_path / 'hsdfm_chi_sq.tiff', chi_sq)

        return output
    except Exception as e:
        print(f'{e} on {cycle}')
        gc.collect()
        return [
        animal, date, oxygen, fov,
        np.nan, np.nan, np.nan, str(cycle)
        ]


def normalize_and_save(cycle, wavelengths, standard, background, out_path):
    # Load cycle subset for naive fitting to mask
    hs = HyperspectralImage(image_path=cycle, wavelengths=wavelengths, standard=standard, background=background)

    # Normalize
    hs.normalize_integration_time()
    hs.normalize_to_standard()

    metadata = {'axes':'TYX'}
    tf.imwrite(out_path / 'hsdfm_norm.tiff', hs.image.copy(), metadata=metadata)

    return hs


def preprocess_and_save(hs, out_path):
    # Resize to 256x256
    hs.resize_to(256)

    # Get variables for naive fit
    hs.subset_by_metadata('Wavelength', wavelengths)  # Subset to Hb dominated spectrum
    selected = np.isin(wl, wavelengths)
    e = eps[:, selected]

    # Fit the image with naive lsq and apply gabor bank
    naive_fit = naive_leastsq_reflectance(hs.image, e)
    hb_index = naive_fit[0] + naive_fit[1]
    gabor_response = apply_kernel_bank(hb_index, gabor_bank)

    # Create and apply mask
    blurred_gabor_response_map = cv2.GaussianBlur(gabor_response, (3, 3), 0.5)
    uint8_resp = (255 * (blurred_gabor_response_map / blurred_gabor_response_map.max())).astype(np.uint8)
    threshold = (blurred_gabor_response_map.max() / 255) * cv2.threshold(uint8_resp, 0, 255, cv2.THRESH_OTSU)[0]
    otsu_mask = blurred_gabor_response_map > threshold
    opened_adaptive_mask = cv2.morphologyEx(
        cv2.adaptiveThreshold(uint8_resp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0),
        cv2.MORPH_OPEN,
        np.ones((3, 3))
    ).astype(bool)
    mask = np.logical_or(otsu_mask, opened_adaptive_mask)
    hs.apply_mask(mask)

    # Save
    tf.imwrite(out_path / 'gabor_response.tiff', gabor_response)
    tf.imwrite(out_path / 'hb_index.tiff', hb_index)
    tf.imwrite(out_path / 'pre_mask.tiff', mask)


def preprocess_cycle_for_fitting(args, skip_exists=True):
    cycle, wavelengths, standard, background = args

    # Find raw data
    animal, date, oxygen, fov, root = parse_categorical(cycle)
    processed = root / 'Processed'
    processed.mkdir(exist_ok=True)
    out_path = processed / animal / datetime.strftime(date, '%m%d%Y') / oxygen / fov
    out_path.mkdir(exist_ok=True, parents=True)
    if skip_exists and (out_path / 'hsdfm_norm.tiff').exists():
            # and datetime.fromtimestamp((out_path / 'hsdfm_norm.tiff').stat().st_mtime).date() == datetime.today().date()):
        print("Skipping:", animal, date, oxygen, fov)
        return out_path

    hs = normalize_and_save(cycle, wavelengths, standard, background, out_path)
    preprocess_and_save(hs, out_path)

    print("Complete:", animal, date, oxygen, fov)
    return out_path


if __name__ == '__main__':
    freeze_support()

    # Dir stuff
    root = Path(r'D:\Jesse\Animal POC')
    cycles = find_cycles(root / "Animals")

    # Choose fitting wavelengths
    wavelengths = np.arange(500, 730, 10)
    absorption_dominated = np.arange(500, 610, 10)
    scatter_dominated = np.arange(610, 730, 10)
    abs_idx = np.array([wl in absorption_dominated for wl in wavelengths])
    sca_idx = np.array([wl in scatter_dominated for wl in wavelengths])

    # Load normalization data
    standard_paths = find_cycles(r'D:\Jesse\hsdfmpm_poc\Standards')
    standard = MergedHyperspectralImage(image_paths=standard_paths, wavelengths=wavelengths, scalar=0.8)
    standard.normalize_integration_time()

    background_paths = find_cycles(r'D:\Jesse\hsdfmpm_poc\Background')
    background = MergedHyperspectralImage(image_paths=background_paths, wavelengths=wavelengths)
    background.normalize_integration_time()

    # Get MCLUT
    smoothing_fn = partial(cv2.GaussianBlur, ksize=(3, 3), sigmaX=2)
    lut = LUT(dimensions=['mu_s', 'mu_a'], scale=50000, extrapolate=True, simulation_id=110, smoothing_fn=smoothing_fn)

    # Prep filter bank
    f = np.geomspace(0.01, 1, 16)
    gabor_bank = gabor_filter_bank(frequency=f, sigma_x=4 / f, sigma_y=1 / f)

    args = [[cyc, wavelengths, standard, background] for cyc in cycles]

    # Preproc
    pool = Pool(15)
    complete_check = []
    try:
        for cycle in tqdm.tqdm(map(preprocess_cycle_for_fitting, args), total=len(cycles)):
            complete_check.append(cycle)
    finally:
        pool.terminate()
        pool.close()
        pool.join()
        del pool
        gc.collect()
    pd.DataFrame(complete_check).to_csv("check.csv")

    # Fitting
    out_path = pd.read_csv(Path(__file__) / "../../check.csv")
    pool = Pool(15)
    output = []
    args = [[cyc, out, wavelengths, lut, sca_idx, abs_idx] for cyc, out in zip(cycles, out_path["0"])]
    try:
        for out in tqdm.tqdm(pool.imap(process_cycle, args), total=len(cycles)):
            output.append(out)
    finally:
        pool.terminate()
        pool.close()
        pool.join()
        del pool
        gc.collect()

    df = pd.DataFrame(output,
                      columns=['Animal', 'Date', 'Oxygen', 'FOV',
                               'Mean THb', 'Mean sO2', 'Mean Offset', 'Full data path'])
    df.to_csv(root / "Processed" / 'hsdfm_output.csv')
