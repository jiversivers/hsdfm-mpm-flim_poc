import sys
import traceback
import warnings
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pandas as pd
import skimage.filters
from hsdfmpm.fit import make_model_for_vascular_distance, distance_to_vasculature, image_reduced_chi_sqaured, pO2_of_sO2
import tifffile as tf
from hsdfmpm.hsdfm.utils import k_cluster, slice_clusters, find_cycles
from hsdfmpm.utils import truncate_colormap
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from skimage.restoration import estimate_sigma
from tqdm import tqdm

cmap = truncate_colormap("jet", cmin=0.13, cmax=0.88)


class ParamSet(NamedTuple):
    animal: str
    date: datetime
    oxygenation: int
    fov: str
    diff_limit: float
    A1: float
    p50: float
    dx: float
    chi_sq: float
    path: Path

    low_orr: float
    med_orr: float
    high_orr: float
    dtv_at_low_orr: float
    dtv_at_med_orr: float
    dtv_at_high_orr: float


def prepared_fov_data(fov_path):
    """Load, mask, and transform data"""
    a2 = tf.imread(fov_path / "a2_fraction.tiff")
    nadh = tf.imread(fov_path / "nadh.tiff")
    so2 = tf.imread(fov_path / "so2.tiff")
    orr = tf.imread(fov_path / "orr_map.tiff")

    a2_mask = make_a2_mask(a2)
    nadh = skimage.filters.median(nadh, np.ones((3, 3)))
    dtv = distance_to_vasculature(a2_mask, um_per_pixel=1.14415550203554)
    tf.imwrite(fov_path / "a2_mask.tiff", a2_mask)
    tf.imwrite(fov_path / "dtv.tiff", dtv)

    return dtv, orr, nadh, a2_mask, np.nanmean(so2), np.nanmedian(nadh[np.logical_and(dtv > 0, dtv < 10)])


def make_a2_mask(a2_fraction):
    """Mask A2 fraction with K-Clustering Algorithm"""
    bad_mask = np.isnan(a2_fraction)
    a2_fraction[bad_mask] = 1  # Make sure it gets ignored in clustering
    blurred = cv2.GaussianBlur(a2_fraction, (5, 5), 3)
    clusters = k_cluster(blurred, 2)
    a2_mask = slice_clusters(blurred, clusters, slice(0, 1, None))
    return a2_mask


def fit_dtv_model(dtv, nadh, a2_mask, so2, A2_est):
    model = make_model_for_vascular_distance("nadh", sO2_0=so2, A2=A2_est)
    x_data = dtv[np.logical_and(~a2_mask, dtv >= 10)]
    y_data = nadh[np.logical_and(~a2_mask, dtv >= 10)]
    p0 = [100, 2 * A2_est, 3.5, 5]  # diff_limit, A1, p50, dx
    try:
        params, cov = curve_fit(
            model,
            x_data,
            y_data,
            p0=p0,
            bounds=[
                (0, A2_est, 0, 0),
                (200, 3 * A2_est, 100, 10)
            ],
            maxfev=5000,
            sigma=estimate_sigma(nadh),
            absolute_sigma=True
        )
        y_fit = model(np.sort(x_data), *params)

    except RuntimeError as e:
        print(e, file=sys.stderr)
        params = [np.nan, np.nan, np.nan, np.nan]

    return params, x_data, y_data, y_fit


def get_ranked_clusters(orr, dtv, k=3, out_path=None):
    clusters = k_cluster(cv2.GaussianBlur(orr, (3, 3), 1), k=k)
    low_orr = slice_clusters(orr, clusters, slice(0, 1, None))
    med_orr = slice_clusters(orr, clusters, slice(1, 2, None))
    high_orr = slice_clusters(orr, clusters, slice(2, 3, None))

    dtv_at_low = np.nanmean(dtv[low_orr])
    dtv_at_med = np.nanmean(dtv[med_orr])
    dtv_at_high = np.nanmean(dtv[high_orr])

    orr_at_low = np.nanmean(orr[low_orr])
    orr_at_med = np.nanmean(orr[med_orr])
    orr_at_high = np.nanmean(orr[high_orr])

    fig, ax = plt.subplots(figsize=(8, 8))
    tf.imwrite(out_path / "orr_sorted_clusters.tiff", np.stack([low_orr, med_orr, high_orr]))
    ax.imshow(low_orr + 2 * med_orr + 3 * high_orr)
    plt.axis("off")
    fig.savefig(out_path / "orr_clusters.png")
    plt.close(fig)
    return orr_at_low, orr_at_med, orr_at_high, dtv_at_low, dtv_at_med, dtv_at_high


def make_plot(params, x_data, y_data, y_fit, out_path):

    # Make plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hexbin(x_data, y_data, label='Data Density', mincnt=1, cmap=cmap)

    if params is not None:
        # Get model outputs
        diff_limit, A1, p50, dx = params
        chi_sq = image_reduced_chi_sqaured(x_data, y_fit, len(params), 2)
        ax.plot(np.sort(x_data), y_fit, color="r", linewidth=2, label=r'$\mathrm{NAD(P)H}\left(d\right)$ Fitted Model')
    else:
        chi_sq = np.nan

    ax.set(xlabel=r'Distance from Vasculature $\mathrm{(\mu m)}$',
           ylabel=r'NADH Equivalent Concentration $\mathrm{(\mu M)}$')
    ax.text(x_data.max(), y_data.max(),
            fr"$d_\mathrm{'{limit}'} = {diff_limit:.2f}$"
            "\t"
            fr"$A_1 = {A1:.2f}$"
            "\n"
            fr"$p_{'{50}'} = {p50:.2f}$"
            "\t"
            fr"$dx = {dx:.2f}$"
            "\n"
            fr"$\chi^2_\nu = {chi_sq:.2f}$",
            ha="right", va="top", fontsize=14, bbox=dict(facecolor='white', alpha=0.7))

    ax.legend(loc="lower right")
    fig.savefig(out_path / "dtv_v_nadh_w_plot.png")
    plt.close(fig)

    return chi_sq


def parse_categorical(fov_path):
    animal, date, oxygen, fov = fov_path.parts[-4:]
    return animal, datetime.strptime(date, "%m%d%Y"), int(oxygen), fov


def worker(fov_path):
    dtv, orr, nadh, a2_mask, so2, A2_est = prepared_fov_data(fov_path)
    params, x_data, y_data, y_fit = fit_dtv_model(dtv, nadh, a2_mask, so2, A2_est)
    cluster_output = get_ranked_clusters(orr, dtv, k=3, out_path=fov_path)
    try:
        x2 = make_plot(params, x_data, y_data, y_fit, fov_path)
    except Exception as e:
        traceback.print_exc()
        print(f"Above traceback from {type(e).__name__}: {e}")
        print(f"Error processing {fov_path}")
    return ParamSet(*parse_categorical(fov_path), *params, x2, fov_path, *cluster_output)

if __name__ == "__main__":
    processed = Path(r"D:\Jesse\Animal POC\Animals\Processed")
    fovs = find_cycles(processed, "so2.tiff")
    df = pd.DataFrame(tqdm(map(worker, fovs), total=len(fovs)))

    df.to_csv(processed / "multimodal_process.csv")
