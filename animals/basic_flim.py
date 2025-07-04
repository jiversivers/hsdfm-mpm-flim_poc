import numpy as np
import re
from hsdfmpm.mpm import InstrumentResponseFunction, LifetimeImage
from hsdfmpm.hsdfm.utils import find_cycles
from hsdfmpm.mpm.flim.utils import plot_universal_circle
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from hsdfmpm.utils import truncate_colormap, colorize
from pathlib import Path
import tifffile as tf
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Dir stuff
root_dir = Path(r'D:\Jesse\Animal POC\Animals')
processed = root_dir / 'Processed'
processed.mkdir(exist_ok=True)

# Get paths
flim_paths = find_cycles(root_dir, search_term='.sdt')

# This creates the IRF model, then stores/updates it in the .hsdfm data to be reused.
irf = InstrumentResponseFunction.load(
    path=r'\\deckard\bmeg\Rajaram-Lab\Ivers,Jesse\Codes\matlab\Toolbox\ImageProcessing\MPM_Processing\FLIM Code\IRF Files\Raw\Upright_I_IRF.sdt',
    reference_lifetime=0, channels=0)
irf.store()

def main():
    output = []
    for path in tqdm(flim_paths):
        # Parse sample categories
        date, animal, oxygen, fov, fov_path = path.parts[4:]
        try:
            date = datetime.strptime(date, '%m%d%Y').date()
        except Exception as e:
            print(e)
        out_path = Path(processed, animal, datetime.strftime(date, '%m%d%Y'), oxygen, fov)
        out_path.mkdir(exist_ok=True, parents=True)

        # Load decay and irf
        decay = LifetimeImage(image_path=path, channels=0)
        decay.load_irf()

        # Calculate phasor coords and fit-line endpoints
        try:
            g, s = decay.phasor_coordinates(threshold=25, median_filter_count=1, k_size=5)

            # Get endpoints
            alphas, taum, tau = decay.fit_for_lifetime_approximations()
            a2_fraction = alphas[1] / (alphas[0] + alphas[1])
            a1_fraction = alphas[0] / (alphas[0] + alphas[1])

            # Make phasor plot with fit line
            cmap = truncate_colormap('jet', cmin=0.13, cmax=0.88)
            b, m = decay.fit_y_intercept, decay.fit_slope
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axline((0, b), slope=m, linestyle=':', color='black', alpha=0.5, label="Fit line")
            ax.hexbin(g, s, cmap=cmap, mincnt=1)
            _, pts = plot_universal_circle(decay.omega, harmonic=1, tau_labels=tau)
            ax.text(*(1.05 * pts[0]), f"{1e9 * tau[0]:.2f} ns", ha="center", va="center")
            ax.text(*(1.05 * pts[1]), f"{1e9 * tau[1]:.2f} ns", ha="center", va="center")
            ax.set(xlabel='G', xlim=[0, 1], ylabel="S", ylim=[0, 0.6], aspect=1)
            fig.savefig(out_path / "phasor_plot.png")
            plt.close(fig)

            for png_name, im in zip(['g', 's', 'a1_fraction', 'a2_fraction', 'taum'], [g, s, a1_fraction, a2_fraction, taum]):
                cmin, cmax = np.mean(im) + np.array([-2, 2]) * np.std(im)
                color_var, cmap = colorize(im.squeeze(), np.nansum(decay.decay, axis=-1).squeeze(), cmap=cmap, cmin=cmin, cmax=cmax)
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(color_var, cmap=cmap)
                ax = plt.gca()
                ax.set_title(f'{animal} {date} {oxygen} {fov}')
                ax.axis('off')
                sm = plt.cm.ScalarMappable(norm=Normalize(vmin=cmin, vmax=cmax), cmap=cmap)
                plt.colorbar(sm, ax=ax)
                plt.tight_layout()
                fig.savefig(out_path / f'color_{png_name}.png')
                plt.close(fig)

            # Save output statistics
            output.append([animal, date, oxygen, fov,
                           np.nanmean(taum), np.nanstd(taum), np.nanmean(a2_fraction), np.nanstd(a2_fraction),
                           decay.fit_y_intercept, decay.fit_slope, decay.aspect_ratio, decay.p_value, decay.red_chi_squared, decay.n,
                           str(path)])

            # Save output images
            tf.imwrite(out_path / 'taum.png', taum)
            tf.imwrite(out_path / 'a2_fraction.tiff', a2_fraction)
            tf.imwrite(out_path / 'g.tiff', g)
            tf.imwrite(out_path / 's.tiff', s)
        except Exception as e:
            print(e)
            print(out_path)
            continue

    return pd.DataFrame(output, columns=['Animal', 'Date', 'Oxygen', 'FOV',
                                         'Mean Tm', 'StDev Tm', 'Mean A2', 'StDev A2',
                                         'b', 'm', 'aspect', 'p_value', 'red_chi', 'n',
                                         'Full data path'])

if __name__ == '__main__':
    df = main()
    df.to_csv(processed / 'mpm_flim_output.csv')