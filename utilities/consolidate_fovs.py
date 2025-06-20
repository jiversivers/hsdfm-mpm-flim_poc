import shutil
import re

def consolidate_fovs(root):
    for animal_dir in root.iterdir():
        if not animal_dir.is_dir():
            continue
        for date_dir in animal_dir.iterdir():
            if not date_dir.is_dir():
                continue
            for group_dir in date_dir.iterdir():
                if not group_dir.is_dir():
                    continue

                fovs = {}
                # Group all related folders by fov prefix (e.g., 'fov1')
                for item in group_dir.iterdir():
                    if item.is_dir():
                        match = re.match(r'(fov\d+)', item.name)
                        if match:
                            fov_name = match.group(1)
                            fovs.setdefault(fov_name, []).append(item)

                for fov, folders in fovs.items():
                    target_fov_folder = group_dir / fov
                    target_fov_folder.mkdir(exist_ok=True)

                    for folder in folders:
                        if folder == target_fov_folder:
                            continue  # Skip already consolidated folder

                        name = folder.name
                        if name.startswith(fov + "_redox_755"):
                            dest = target_fov_folder / "redox_755"
                        elif name.startswith(fov + "_redox_855"):
                            dest = target_fov_folder / "redox_855"
                        elif name.startswith(fov + "_flim_755"):
                            dest = target_fov_folder / "flim"
                        elif name == fov:
                            # Copy subfolders like cycle1, cycle2 directly
                            for subfolder in folder.iterdir():
                                if subfolder.is_dir():
                                    shutil.move(str(subfolder), target_fov_folder / subfolder.name)
                            shutil.rmtree(folder)
                            continue
                        else:
                            continue  # skip unrelated folders

                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.move(str(folder), dest)

if __name__ == "__main__":
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser(description="Consolidate fov folders into single directories.")
    parser.add_argument("root", type=str, help="Path to the root Animals directory (e.g. D:/Animals)")
    args = parser.parse_args()

    root_path = Path(args.root).resolve()
    if not root_path.exists():
        print(f"Error: Provided path '{root_path}' does not exist.")
    else:
        consolidate_fovs(root_path)
        print(f"Finished consolidating FOV folders under '{root_path}'")