import urllib.request
import zipfile
import os.path

datasets = ['lm', 'tless', 'ycbv', 'tyol']
SRC = "https://bop.felk.cvut.cz/media/data/bop_datasets"
DST = r"D:\bop_toolkit\data"

for dataset in datasets:
    if dataset in ['lm', 'ycbv']:
        subsets = ["base", "models"] #, "test_all", "train_pbr"]
    if dataset == 'tless':
        subsets = ["base", "models"] #, "test_primesense_all", "train_pbr"]
    for subset in subsets:
        dest_zip = f"{DST}/{dataset}_{subset}.zip"
        if not os.path.isfile(dest_zip):
            print(f"Downloading {dataset}, {subset}...")
            urllib.request.urlretrieve(SRC + f'/{dataset}_{subset}.zip', dest_zip)
        if subset != "base":
            dest_unzip = DST + f"/{dataset}/"
        else:
            dest_unzip = DST
        print(f"Unzipping {dataset}, {subset}...")
        with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
            zip_ref.extractall(dest_unzip)
        print("Done")
