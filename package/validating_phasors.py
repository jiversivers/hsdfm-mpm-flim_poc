import gc
from multiprocessing import Pool

import numpy as np
import pandas as pd
from numpy import iterable
import itertools

from tqdm import tqdm

from hsdfmpm.mpm import InstrumentResponseFunction
from hsdfmpm.mpm.flim.utils import (
    get_phasor_coordinates,
    find_intersection_with_circle,
    project_to_line,
    lifetime_from_cartesian,
    get_endpoints_from_projection,
    fit_phasor
)

rng = np.random.default_rng(42)
omega = 2 * np.pi * 80e6
t = np.linspace(0.5, 10.5, 256) / 1e9
irf = InstrumentResponseFunction.load()

def generate_decay_histogram(tau1, tau2=None, alpha=1.0, n_photons=1e3, bin_count=256, t_max=10.0e-9):
    tau1  = np.asanyarray(tau1,  dtype=float)
    if tau2 is None:
        tau2  = tau1
        alpha = 1.0
    tau2  = np.asanyarray(tau2,  dtype=float)
    alpha = np.asanyarray(alpha, dtype=float)

    t_edges = (np.linspace(0, t_max, bin_count + 1))[np.newaxis, np.newaxis, ...]
    t0, t1 = t_edges[..., :-1], t_edges[..., 1:]

    p1 = np.exp(-t0 / tau1) - np.exp(-t1 / tau1)
    p2 = np.exp(-t0 / tau2) - np.exp(-t1 / tau2)
    probs = alpha * p1 + (1 - alpha) * p2

    probs /= np.sum(probs, axis=-1, keepdims=True)
    flat = probs.reshape(-1, bin_count)
    hists = np.asarray([rng.multinomial(n_photons, p).astype(np.uint32) for p in flat]).reshape(probs.shape)
    return hists


def convolve_with_irf(decay):
    while decay.ndim < irf.decay.ndim:
        decay = np.expand_dims(decay, axis=0)
    T = decay.shape[-1]
    conv_L = decay.shape[-1] + irf.shape[-1] -  1
    decay = np.fft.fft(decay, n=conv_L, axis=-1)
    irf_fft = np.fft.fft(irf.decay / irf.decay.sum(axis=-1, keepdims=-1), n=conv_L, axis=-1)
    decay *= irf_fft
    decay = np.fft.ifft(decay, axis=-1).real
    return decay[..., :T]


def bias(est, true):
    return np.nanmean(est - true)


def rmse(est, true):
    return np.sqrt(np.nanmean((est - true)**2))


def r2(est, true):
    return np.corrcoef(est[~np.isnan(est)], true[~np.isnan(est)])[0,1]**2


def score(est, true):
    return bias(est, true), rmse(est, true), r2(est, true)


def process_simulation(args):
    t1, t2, acntr, asprd, convolved = args
    tau1 = np.ones((64, 64, 1)) * t1
    tau2 = np.ones((64, 64, 1)) * t2

    alpha = rng.normal(loc=acntr, scale=asprd, size=(64, 64, 1)).clip(0, 1)
    tm = np.sum(t1 * alpha + t2 * (1 - alpha), axis=-1)
    decay = generate_decay_histogram(tau1=tau1, tau2=tau2, alpha=alpha)
    decay = decay[np.newaxis, ...]

    if convolved:
        decay = convolve_with_irf(decay)
    P, photons = get_phasor_coordinates(decay, as_complex=True)

    if convolved:
        P *= irf.correction
    g, s = P.real, P.imag

    # Fit a line
    output = fit_phasor(g, s)
    b, m = output["fit_y_intercept"], output["fit_slope"]
    ratio = output["aspect_ratio"]
    p_value, n, red_chi_squared = output["p_value"], output["n"], output["red_chi_squared"]
    xs, ys = find_intersection_with_circle(b, m)
    gp, sp = project_to_line(g, s, xs, ys)
    tau = lifetime_from_cartesian(xs, ys, omega)  # tau:
    alphas, tau_m = get_endpoints_from_projection(gp, sp, xs, ys, tau)

    # Calculate error metrics
    t1_score = (bias(tau[0], t1), rmse(tau[0], t1))
    t2_score = (bias(tau[1], t2), rmse(tau[1], t2))
    taus_score = (bias(tau, np.array([t1, t2])), rmse(tau, np.array([t1, t2])))
    a1_score = score(alphas[0].flatten(), alpha.flatten())
    a2_score = score(alphas[1].flatten(), (1 - alpha).flatten())
    tm_score = score(tau_m.flatten(), tm.flatten())

    out = [ratio, t1_score, t2_score, taus_score, a1_score, a2_score, tm_score, p_value, n, red_chi_squared]
    gc.collect()

    return out


if __name__ == '__main__':
    # Generate arrays of two lifetime species
    tau1_value_arr = np.arange(0.1, 1, 0.05) * 1e-9
    tau2_value_arr = np.arange(1.5, 8, 0.25) * 1e-9
    alpha_center = np.arange(0, 1, 0.05)
    alpha_spread = np.arange(0, 0.5, 0.05)
    convolved = [False, True]

    pool = Pool(processes=15)
    try:
        stats = pool.imap(
            process_simulation,
            itertools.product(tau1_value_arr, tau2_value_arr, alpha_center, alpha_spread, convolved),
            chunksize=256
        )
        unpacked_data = []
        for params, stat in tqdm(
                zip(itertools.product(tau1_value_arr, tau2_value_arr, alpha_center, alpha_spread, convolved), stats),
                desc='Processing',
                total=(len(tau1_value_arr) * len(tau2_value_arr) * len(alpha_center) * len(alpha_spread) * len(convolved))
        ):
            ud = [*params]
            for item in stat:
                if iterable(item):
                    for i in item:
                        ud.append(i.item())
                else:
                    try:
                        ud.append(item.item())
                    except AttributeError:
                        ud.append(item)
            unpacked_data.append(ud)


        df = pd.DataFrame(
            unpacked_data,
            columns=[
                'true_t1',
                'true_t2',
                'center_alpha',
                'spread_alpha',
                'convolved',
                'ratio',
                't1_bias',
                't1_rmse',
                't2_bias',
                't2_rmse',
                'taus_bias',
                'taus_rmse',
                'a1_bias',
                'a1_rmse',
                'a1_r2',
                'a2_bias',
                'a2_rmse',
                'a2_r2',
                'tm_bias',
                'tm_rmse',
                'tm_r2'
                'p_value',
                'n',
                'red_chi_squared'
            ]
        )
        df.to_csv('phasor_validation_sweep_stats.csv')
    except Exception as e:
        print(e)
        pool.terminate()
    finally:
        pool.close()
        pool.join()
