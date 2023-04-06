import os
import sys

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

    keypoints_array = np.array([[[[[-9.7732497e-03, 3.6659201e-03, -1.4534000e-03],
                                   [3.6046398e+01, -1.4680500e+01, -4.5020599e+01],
                                   [-3.0289101e+01, -7.2402501e+00, -4.2632900e+01],
                                   [7.4425201e+00, 2.3966400e+01, -3.9362701e+01],
                                   [-4.3485899e+00, 3.6769500e+00, 4.5835800e+01],
                                   [6.2882501e-01, -3.6412800e+01, -2.7732599e+01],
                                   [-2.6754901e-01, 3.7588799e+01, -4.7640200e+00],
                                   [3.0029400e+01, -2.3939800e+01, -8.1097898e+00],
                                   [-2.8789900e+01, -1.9449200e+01, -9.0417604e+00]]],

                                 [[[1.6300200e-02, -2.3040799e-03, -1.1291500e-02],
                                   [5.5248199e+00, 5.4157101e+01, -9.6322701e+01],
                                   [-4.1018100e+00, 1.2732400e+01, 9.6678497e+01],
                                   [-9.1580000e+00, -4.1244202e+01, -8.7472397e+01],
                                   [7.3375401e+00, 9.0886101e+01, -1.1365300e+01],
                                   [-1.0262200e+01, -9.0547600e+01, -3.7563899e-01],
                                   [-4.7794201e+01, 1.6508699e+01, -5.6376900e+01],
                                   [4.8287998e+01, 2.4022501e+00, -6.2877899e+01],
                                   [4.6154099e+01, 1.1302400e+01, 4.9851101e+01]]],

                                 [[[1.7128000e-02, -4.5700101e-03, -5.3901700e-03],
                                   [2.0947300e+01, -6.1587502e+01, -5.4198200e+01],
                                   [-2.0933701e+01, 6.3563000e+01, 2.6130899e+01],
                                   [2.8901501e+01, 2.7392700e+01, -5.7568199e+01],
                                   [1.4403200e+00, -5.8665901e+01, 2.2473900e+01],
                                   [1.2946500e+01, 1.4082400e+01, 5.8292999e+01],
                                   [-2.8743299e+01, 1.6301001e+01, -5.2558300e+01],
                                   [-3.3441200e+01, -4.1310501e+01, -5.4232101e+01],
                                   [2.3869900e+01, 4.1699699e+01, 1.6587299e+01]]],

                                 [[[-2.4108901e-03, -6.2332200e-03, -6.3247699e-03],
                                   [1.1291000e+02, -3.4727199e+00, 9.2172699e+01],
                                   [-1.1182900e+02, 3.1709600e-02, 6.1154400e+01],
                                   [-6.2377201e+01, 1.0970700e+01, -1.0025700e+02],
                                   [4.2661201e+01, -2.4666700e+01, -9.9452499e+01],
                                   [1.0724100e+01, -3.5357201e+00, 1.0133300e+02],
                                   [-4.1970699e+01, -3.1155399e+01, 5.4645599e+01],
                                   [4.9310899e+00, 3.6434399e+01, -9.7123596e+01],
                                   [5.6840302e+01, -4.2665200e+00, 4.8058399e+01]]],

                                 [[[-3.4179699e-03, -9.8838797e-03, 3.9329501e-03],
                                   [4.9320702e+01, 6.2302999e+00, -4.0302898e+01],
                                   [-4.6246700e+01, 2.3396499e+00, -3.7502899e+01],
                                   [1.2448000e+01, -3.3365299e+01, -4.0734501e+01],
                                   [3.9640200e+00, 3.4297600e+01, -4.0923302e+01],
                                   [4.5272598e+01, -1.0067500e+00, 2.1399401e+01],
                                   [6.6833901e+00, -3.1548400e+00, 4.2783199e+01],
                                   [-2.3509399e+01, -2.7834400e+01, -1.9335600e+01],
                                   [-4.1355202e+01, 1.3988900e-01, 1.3391900e+00]]],

                                 [[[-1.7417900e-02, -4.2999300e-01, -1.3252300e-02],
                                   [-7.0443398e+01, 4.3526299e+01, 4.2999201e+00],
                                   [7.3233902e+01, 3.5586300e+01, 4.8644700e+00],
                                   [6.7131897e+01, -4.4466202e+01, -2.7725799e+00],
                                   [-7.0990898e+01, -3.6974701e+01, -1.3353300e+00],
                                   [-4.7924999e+01, 5.5036702e+00, -3.2836399e+01],
                                   [2.2584101e+01, 4.1242500e+01, 3.2724400e+01],
                                   [-2.4753901e+01, -4.0470100e+01, 3.2213699e+01],
                                   [4.7744598e+01, 4.2735401e-01, -3.1653799e+01]]],

                                 [[[9.9391900e-03, -1.1459400e-02, 6.8359398e-03],
                                   [-9.1147299e+00, -3.1402399e+01, -8.5777802e+01],
                                   [9.7676700e-01, 2.9348700e+00, 8.6390404e+01],
                                   [6.4356799e+00, 3.7870701e+01, -6.3978802e+01],
                                   [9.7071304e+00, -3.6640800e+01, -3.6885799e+01],
                                   [-1.5302700e+01, 1.4431200e+00, -4.7971500e+01],
                                   [-6.0784298e-01, -1.2160700e+01, 4.3689098e+01],
                                   [1.7079800e+01, 1.9666600e+00, -8.3763802e+01],
                                   [-4.1084499e+00, 3.5197800e+01, -2.3239799e+01]]],

                                 [[[-6.8052673e-01, 4.3509445e+00, -1.5487452e+00],
                                   [5.8407185e+01, -8.2160988e+01, -1.2327696e+01],
                                   [-9.1078697e+01, -4.6658367e+01, -7.3403926e+00],
                                   [9.6684196e+01, 1.5651445e+01, 2.6858237e+00],
                                   [-7.8112083e+01, 6.0135139e+01, -6.0886226e+00],
                                   [3.1481721e+00, 9.0496506e+01, -1.1837043e+01],
                                   [-1.9831656e+01, -8.0089989e+01, -2.5694869e+00],
                                   [5.0045956e+01, 5.1196590e+01, -1.4792501e+01],
                                   [2.3459833e+01, -4.9363476e+01, 1.9859182e+01]]]]],
                               )

    keypoints = tf.convert_to_tensor(keypoints_array, dtype=tf.float32)

    camera_matrix = [[345.5395354181145, 0, 319.4688241083385],
                     [0, 345.25337576116874, 237.47917860129158],
                     [0, 0, 1]],
    camera_matrix = tf.convert_to_tensor(camera_matrix, dtype=tf.float32)

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

    # net.summary()

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
    # tf.print(poses_est[0, 0, 0, 0, 0])
    return poses_est


if __name__ == '__main__':
    opt = parse_config()
    opt.modelname = 'casapose_c_gcu5'
    opt.load_h5_weights = True
    opt.load_h5_filename = '../casapose/data/pretrained_models/result_w'
    opt.datatest = '../../../data/datasets/lm_251/test'
    opt.datameshes = 'D:\\bop_toolkit\\data\\lm_251\\lm\\models'
    opt.train_vectors_with_ground_truth = False
    opt.save_eval_batches = True
    opt.object = 'obj_000001,obj_000005,obj_000006,obj_000008,obj_000009,obj_000010,obj_000011,obj_000016'

    random_image = tf.random.uniform([448, 448, 3])

    poses_est = inference_on_image(random_image)

    print(f"The pose estimates for this image are: {poses_est}")
