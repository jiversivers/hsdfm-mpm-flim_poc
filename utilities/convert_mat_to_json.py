import scipy.io
import json
def convert_mat_to_json(file_path):
    # Load .mat file
    mat = scipy.io.loadmat(file_path, squeeze_me=True)

    data_cell = mat['imgInfo']

    # Build JSON-compatible list
    json_data = []
    for item in data_cell:
        entry = {
            "AbsTime": item['AbsTime'].tolist().tolist(),
            "ExpTime": float(item['ExpTime']),
            "Filter": int(item['Filter']),
            "Wavelength": int(item['Filter'])
        }
        json_data.append(entry)

    # Write to JSON file
    output_path = file_path.parent / (file_path.stem + ".json")
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    return output_path

if __name__ == '__main__':
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser(description='Convert mat file to json')
    parser.add_argument('mat', help='mat file')
    args = parser.parse_args()

    mat_path = Path(args.mat)
    if not mat_path.exists():
        raise FileNotFoundError(f'Could not find MAT file: {mat_path}')
    else:
        out = convert_mat_to_json(mat_path)
        print(f'Saved to {out}')