import urllib.request
import zipfile
import os.path

datasets = ['lm']
SRC = "https://bop.felk.cvut.cz/media/data/bop_datasets"
DST = r"D:\bop_toolkit\data"

for dataset in datasets:
    for subset in ["base", "models", "test_all", "train_pbr"]:
        dest_zip = f"{DST}/{dataset}_{subset}.zip"
        if not os.path.isfile(dest_zip):
            print(f"Downloading {dataset}, {subset}...")
            urllib.request.urlretrieve(SRC + f'/{dataset}_{subset}.zip', dest_zip)
        if subset != "base":
            dest_unzip = DST + "/lm/"
        else:
            dest_unzip = DST
        print(f"Unzipping {dataset}, {subset}...")
        with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
            zip_ref.extractall(dest_unzip)
        print("Done")
