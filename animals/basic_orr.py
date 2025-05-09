import numpy as np
from hsdfmpm.hsdfm.utils import find_cycles
from hsdfmpm.utils import truncate_colormap
from hsdfmpm.mpm import AutofluorescenceImage
from hsdfmpm.mpm import OpticalRedoxRatio
import warnings
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
import imageio.v3 as iio
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Dir stuff
root_dir = Path(r'E:\new df\POC Study')
processed = root_dir / 'Processed'
processed.mkdir(exist_ok=True)

# Point to power files
power_file_dir = root_dir / 'LaserPower'

# Get paths
raw_paths = find_cycles(root_dir / 'Animals', search_term='.xml')
flim_paths = find_cycles(root_dir / 'Animals', search_term='.sdt')
fov_paths = np.unique([path.parent for path in raw_paths])

def main():

    output = []
    for path in tqdm(fov_paths):
        ex755_path = [p for p in path.glob('*755*') if not p in flim_paths]
        ex855_path = [p for p in path.glob('*855*') if not p in flim_paths]
        if not (ex755_path and ex855_path):
            warnings.warn(f'No 755 or 855 found for {path}', category=RuntimeWarning, stacklevel=2)
            continue
        if len(ex755_path) > 1 or len(ex855_path) > 1:
            warnings.warn(f'Multiple 755 or 855 found for {path}', category=RuntimeWarning, stacklevel=2)
            continue

        # Parse sample categories
        animal, date, oxygen, fov = path.parts[-4:]
        date = datetime.strptime(date, '%m%d%Y').date()
        out_path = Path(processed, animal, datetime.strftime(date, '%m%d%Y'), oxygen, fov)
        out_path.mkdir(exist_ok=True, parents=True)

        # Load images
        orr = OpticalRedoxRatio(ex755=ex755_path[0], ex855=ex855_path[0], power_file_path=power_file_dir)

        # Down sample for SNR and to match HSDFM
        orr.resize_to(256)

        # Create outputs
        cmap = truncate_colormap('jet', cmin=0.13, cmax=0.88)
        cmin, cmax = np.mean(orr.map) + np.array([-2, 2]) * np.std(orr.map)
        color_orr, cmap = orr.colorize(cmap=cmap, cmin=cmin, cmax=cmax)
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(color_orr, cmap=cmap)
        ax = plt.gca()
        ax.set_title(f'{animal} {date} {oxygen} {fov}')
        ax.axis('off')
        sm = plt.cm.ScalarMappable(norm=Normalize(vmin=cmin, vmax=cmax), cmap=cmap)
        plt.colorbar(sm, ax=ax)
        plt.tight_layout()

        # Save output statistics
        output.append([animal, date, oxygen, fov, np.nanmean(orr.map), np.nanstd(orr.map), str(path)])

        # Save output images
        iio.imwrite(out_path / 'orr_map.tiff', orr.map)
        iio.imwrite(out_path / 'af_intensity.tiff', (orr.fad + orr.nadh) / 2 )
        iio.imwrite(out_path / 'nadh_intensity.tiff', orr.nadh)
        iio.imwrite(out_path / 'fad_intensity.tiff', orr.fad)
        fig.savefig(out_path / 'color_orr.png')
        plt.close(fig)

    return pd.DataFrame(output, columns=['Animal', 'Date', 'Oxygen', 'FOV', 'Mean ORR', 'StDev ORR', 'Full data path'])

if __name__ == '__main__':
    df = main()
    df.to_csv(processed / 'mpm_af_output.csv')