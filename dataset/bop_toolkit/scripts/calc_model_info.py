# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates the 3D bounding box and the diameter of 3D object models."""
from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('datasets_path', help="Path to where the dataset is")
args = parser.parse_args()
print(f"args: {args}")

# PARAMETERS.
################################################################################
p = {
  # See dataset_params.py for options.
  'dataset': 'lm',

  # Type of input object models.
  'model_type': 'pbr',

  # Folder containing the BOP datasets.
  'datasets_path': args.datasets_path,
}
################################################################################

# TODO: rename object ids to `obj_00000`

# Load dataset parameters.
dp_model = dataset_params.get_model_params(
  p['datasets_path'], p['dataset'], p['model_type'])

models_info = {}
for obj_id in dp_model['obj_ids']:
    misc.log('Processing model of object {}...'.format(obj_id))
    model_dir = dp_model['model_tpath'].format(obj_id=obj_id).replace("_pbr", "")
    model = inout.load_ply(model_dir)

    # Calculate 3D bounding box.
    ref_pt = np.array(list(map(float, model['pts'].min(axis=0)))).flatten()
    size = np.array(list(map(float, (model['pts'].max(axis=0) - ref_pt)))).flatten()

    # Calculated diameter.
    diameter = misc.calc_pts_diameter(model['pts'])

    models_info[obj_id] = {
        'min_x': ref_pt[0], 'min_y': ref_pt[1], 'min_z': ref_pt[2],
        'size_x': size[0], 'size_y': size[1], 'size_z': size[2],
        'diameter': diameter
    }

# Save the calculated info about the object models.
inout.save_json(dp_model['models_info_path'].replace("_pbr", ""), models_info)
