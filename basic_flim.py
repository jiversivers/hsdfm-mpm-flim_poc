import numpy as np
from hsdfmpm.mpm import InstrumentResponseFunction, LifetimeImage
from hsdfmpm.hsdfm.utils import find_cycles
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from hsdfmpm.utils import truncate_colormap, colorize
from pathlib import Path
import imageio.v3 as iio
from datetime import datetime
import pandas as pd

# Dir stuff
root_dir = Path(r'E:\new df\POC Study')
processed = root_dir / 'Processed'
processed.mkdir(exist_ok=True)

# Get paths
flim_paths = find_cycles(root_dir / 'Animals', search_term='.sdt')

# This creates the IRF model, then stores/updates it in the .hsdfm data to be reused.
irf = InstrumentResponseFunction.load(
    path=r'\\deckard\bmeg\Rajaram-Lab\Ivers,Jesse\Codes\matlab\Toolbox\ImageProcessing\MPM_Processing\FLIM Code\IRF Files\Raw\Upright_I_IRF.sdt',
    reference_lifetime=0, channels=0)
irf.store()

def main():
    output = []
    for path in flim_paths:
        # Parse sample categories
        animal, date, oxygen, fov = path.parts[-5:-1]
        date = datetime.strptime(date, '%m%d%Y').date()
        out_path = Path(processed, animal, datetime.strftime(date, '%m%d%Y'), oxygen, fov)
        out_path.mkdir(exist_ok=True, parents=True)

        # Load decay and irf
        decay = LifetimeImage(image_path=path, channels=0)
        decay.load_irf()
        decay.resize_to(256)

        # Calculate phasor coords and fit-line endpoints
        g, s = decay.phasor_coordinates(threshold=25, median_filter_count=0, correction=True)
        alphas, taum, tau = decay.fit_for_lifetime_approximations(median_filter_count=1, k_size=5)
        a2_fraction = alphas[1] / (alphas[0] + alphas[1])
        a1_fraction = alphas[0] / (alphas[0] + alphas[1])

        cmap = truncate_colormap('jet', cmin=0.13, cmax=0.88)
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
        output.append([animal, date, oxygen, fov, np.nanmean(a2_fraction), np.nanstd(a2_fraction), str(path)])

        # Save output images
        iio.imwrite(out_path / 'a2_fraction.tiff', a2_fraction)


    df = pd.DataFrame(output, columns=['Animal', 'Date', 'Oxygen', 'FOV', 'Mean A2', 'StDev A2', 'Full data path'])

if __name__ == '__main__':
    df = main()
    df.to_csv(processed / 'mpm_flim_output.csv')