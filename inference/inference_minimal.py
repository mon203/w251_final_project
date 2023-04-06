import os
import sys

import json
import numpy as np
import tensorflow as tf

sys.path.extend([".", ".."])  # adds the folder from which you call the script
os.environ["CASAPOSE_INFERENCE"] = "True"

from casapose.pose_estimation.pose_evaluation import poses_pnp
from casapose.pose_estimation.voting_layers_2d import CoordLSVotingWeighted
from casapose.pose_models.tfkeras import Classifiers
from casapose.utils.config_parser import parse_config


def inference_on_image(image):
    # Create necessary variables
    objectsofinterest = [x.strip() for x in opt.object.split(",")]
    no_objects = len(objectsofinterest)

    separated_vectorfields = opt.modelname == "pvnet"
    no_points = opt.no_points

    height = opt.imagesize_test[0]
    width = opt.imagesize_test[1]

    input_segmentation_shape = None

    checkpoint_path = opt.outf + "/" + opt.net

    # Load camera intrinsic matrix json
    with open(r"C:\Users\grandid\source\repos\w251_final_project\inference\calib_camera2_2__facing_back_640.json") as f:
        data = json.load(f)
    camera_matrix = tf.convert_to_tensor([data['camera_matrix']])

    # Create model
    CASAPose = Classifiers.get(opt.modelname)
    ver_dim = opt.no_points * 2
    if opt.modelname == "pvnet":
        ver_dim = ver_dim * no_objects

    if opt.estimate_confidence:
        assert separated_vectorfields is not None, "confidence not compaitble with this model"
        ver_dim += opt.no_points

    net = CASAPose(
        ver_dim=ver_dim,
        seg_dim=1 + no_objects,
        input_shape=(height, width, 3),
        input_segmentation_shape=input_segmentation_shape,
        weights="imagenet",
        base_model=opt.backbonename,
    )

    # Load model checkpoint
    checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")
    checkpoint = tf.train.Checkpoint(network=net)  # , optimizer=optimizer)

    if opt.load_h5_weights is True:
        net.load_weights(opt.load_h5_filename + ".h5", by_name=True, skip_mismatch=True)

    elif opt.net != "":
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

    for layer in net.layers:
        layer.trainable = False

    net.summary()

    # Instantiate model and run `image` through model to infer estimated poses
    output_net = net([tf.expand_dims(image, 0)], training=False)
    output_seg, output_dirs, confidence = tf.split(output_net, [no_objects, no_points * 2, -1], 3)
    coordLSV_in = [output_seg, output_dirs, confidence]
    coords = CoordLSVotingWeighted(
        name="coords_ls_voting",
        num_classes=no_objects,
        num_points=no_points,
        filter_estimates=True,
    )(coordLSV_in)
    poses_est = poses_pnp(
        coords, output_seg, keypoints, camera_matrix, no_objects - 1, min_num=opt.min_object_size_test
    )
    tf.print(poses_est[0, 0, 0, 0, 0])
    return poses_est


if __name__ == '__main__':
    # -c config / config_8_16.ini - -load_h5_weights
    # 1 - -load_h5_filename.. / data / pretrained_models / result_w - -datatest.. /../../ data / datasets / lm_251 / test - -datameshes
    # D:\\bop_toolkit\\data\\lm_251\\lm\\models - -train_vectors_with_ground_truth 0 --save_eval_batches 1

    opt = parse_config()
    opt.modelname = 'casapose_c_gcu5'
    opt.load_h5_weights = True
    opt.load_h5_filename = '../data/pretrained_models/result_w'
    opt.datatest = '../../../data/datasets/lm_251/test'
    opt.datameshes = 'D:\\bop_toolkit\\data\\lm_251\\lm\\models'
    opt.train_vectors_with_ground_truth = False
    opt.save_eval_batches = True


    random_image = tf.random.uniform([480, 640, 3])

    poses_est = inference_on_image(random_image)
