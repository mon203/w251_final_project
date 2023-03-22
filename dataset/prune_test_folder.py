import json
from pathlib import Path


def clean_scene_gt(path):
    with open(path, 'r') as f:
        scene_gt = json.load(f)
    scene_gt_clean = {}
    for key, values in scene_gt.items():
        for value in values:
            if value['obj_id'] == 16:
                scene_gt_clean[key] = [value]
    return scene_gt_clean


def rename_test_folder(dataset_dir):
    test_dir = dataset_dir.joinpath('lm', 'test_pbr', '000016')
    try:
        dataset_dir.joinpath('lm', 'test_pbr', '000000').rename(test_dir)
    except FileNotFoundError:
        print(f"Test folder already exists: {test_dir}")
    return test_dir


def main(dataset_dir):
    # rename test folder
    # test_dir = rename_test_folder(dataset_dir)
    test_dir = dataset_dir.joinpath('lm', 'test_pbr', '000016')

    # modify scene_gt.json to remove instances of objects other than 16
    scene_gt_path = test_dir / "scene_gt.json"
    scene_gt = clean_scene_gt(scene_gt_path)
    with open(scene_gt_path, 'w') as f:
        json.dump(scene_gt, f) #, ensure_ascii=False, indent=4)

    print("Done")


if __name__ == '__main__':
    dataset_dir = Path(r"D:\bop_toolkit\data\lm_251")
    main(dataset_dir)
