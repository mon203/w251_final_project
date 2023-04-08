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

from casapose.utils.geometry_utils import apply_offsets, project
from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()
from casapose.utils.draw_utils import draw_bb

from PIL import Image


def input_parameters(h5_path, meshes_dir, synthetic_image):
    class opt:
        modelname = 'casapose_c_gcu5'
        estimate_confidence = 1
        estimate_coords = 1
        confidence_regularization = 1
        object = 'obj_000001,obj_000005,obj_000006,obj_000008,obj_000009,obj_000010,obj_000011,obj_000016'

        no_points = 9
        save_debug_batch = 0

        imagesize = (448, 448)
        imagesize_test = (480, 640)
        crop_factor = 0.933333333
        pretrained = 1
        manualseed = 1237

        # losses
        mask_loss_weight = 1.0
        vertex_loss_weight = 0.5
        proxy_loss_weight = 0.015
        keypoint_loss_weight = 0.007
        filter_vertex_with_segmentation = 1
        filter_high_proxy_errors = 0
        use_bpnp_reprojection_loss = 0
        max_keypoint_pixel_error = 12.5

        # learning rate
        lr = 0.001
        lr_decay = 0.5
        lr_epochs_steps = 50, 75, 90

        # general
        gpuids = 0, 1
        loginterval = 10
        epochs = 100
        batchsize = 4
        saveinterval = 5
        validationinterval = 1

        # data preprocessing
        workers = 0
        prefetch = 10

        # augmentation
        translation = 0
        rotation = 0
        noise = 0.0001
        brightness = 0.001
        contrast = 0.001
        saturation = 0.001
        hue = 0.001
        use_imgaug = 1

        # test
        min_object_size_test = 200
        write_poses = 0
        save_eval_batches = 0

        # output
        net = 'training_checkpoints'
        outf = 'train_casapose_8_16_objects'

        # config
        train_vectors_with_ground_truth = 1
        load_h5_weights = 0
        copy_weights_from_backup_network = 0
        copy_weights_add_confidence_maps = 0
        objects_in_input_network = 8
        objects_to_copy = 1
        objects_to_copy_list = 'config/objects_to_copy.csv'

        confidence_filter_estimates = 1
        confidence_choose_second = 0
        train_vectors_with_ground_truth = 0
        datatest_wxyz_quaterion = 0
        filter_test_with_gt = 0

        evalf = 'output'
        datatest = 'import_data/test/test'
        datameshes = 'import_data/test/models'
        # datatest = '/workspace/CASAPose/import_data/test/test'
        # datameshes = '/workspace/CASAPose/import_data/test/models'
        data = ''
        datatest_path_filter = None
        color_dataset = 1
        train_validation_split = None  # 0.9
        backbonename = 'resnet18'
        load_h5_weights = 1
        # load_h5_filename = '../../../data/pretrained_models/result_w'
        load_h5_filename = 'data/pretrained_models' + '/result_w.h5'

    # opt = parse_config()
    # opt.modelname = 'casapose_c_gcu5'
    # opt.load_h5_weights = True
    opt.load_h5_filename = h5_path
    opt.datameshes = meshes_dir
    # opt.train_vectors_with_ground_truth = False
    # opt.save_eval_batches = True
    # opt.object = 'obj_000001,obj_000005,obj_000006,obj_000008,obj_000009,obj_000010,obj_000011,obj_000016'

    objectsofinterest = [x.strip() for x in opt.object.split(",")]
    no_objects = len(objectsofinterest)
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
    cuboids = np.array([[[[-37.92094, -38.788555, -45.88129],
                          [-37.92094, -38.788555, 45.87838],
                          [-37.92094, 38.795883, -45.88129],
                          [-37.92094, 38.795883, 45.87838],
                          [37.901394, -38.788555, -45.88129],
                          [37.901394, -38.788555, 45.87838],
                          [37.901394, 38.795883, -45.88129],
                          [37.901394, 38.795883, 45.87838]]],

                        [[[-50.35713, -90.89071, -96.8516],
                          [-50.35713, -90.89071, 96.82902],
                          [-50.35713, 90.8861, -96.8516],
                          [-50.35713, 90.8861, 96.82902],
                          [50.38973, -90.89071, -96.8516],
                          [50.38973, -90.89071, 96.82902],
                          [50.38973, 90.8861, -96.8516],
                          [50.38973, 90.8861, 96.82902]]],

                        [[[-33.44303, -63.791, -58.71809],
                          [-33.44303, -63.791, 58.707314],
                          [-33.44303, 63.781857, -58.71809],
                          [-33.44303, 63.781857, 58.707314],
                          [33.47729, -63.791, -58.71809],
                          [33.47729, -63.791, 58.707314],
                          [33.47729, 63.781857, -58.71809],
                          [33.47729, 63.781857, 58.707314]]],

                        [[[-114.72308, -37.718895, -103.983604],
                          [-114.72308, -37.718895, 103.97095],
                          [-114.72308, 37.706425, -103.983604],
                          [-114.72308, 37.706425, 103.97095],
                          [114.71827, -37.718895, -103.983604],
                          [114.71827, -37.718895, 103.97095],
                          [114.71827, 37.706425, -103.983604],
                          [114.71827, 37.706425, 103.97095]]],

                        [[[-52.200897, -38.71081, -42.8214],
                          [-52.200897, -38.71081, 42.82927],
                          [-52.200897, 38.691044, -42.8214],
                          [-52.200897, 38.691044, 42.82927],
                          [52.194057, -38.71081, -42.8214],
                          [52.194057, -38.71081, 42.82927],
                          [52.194057, 38.691044, -42.8214],
                          [52.194057, 38.691044, 42.82927]]],

                        [[[-75.0917, -54.39756, -34.629425],
                          [-75.0917, -54.39756, 34.602924],
                          [-75.0917, 53.53758, -34.629425],
                          [-75.0917, 53.53758, 34.602924],
                          [75.05686, -54.39756, -34.629425],
                          [75.05686, -54.39756, 34.602924],
                          [75.05686, 53.53758, -34.629425],
                          [75.05686, 53.53758, 34.602924]]],

                        [[[-18.320473, -38.923126, -86.376724],
                          [-18.320473, -38.923126, 86.3904],
                          [-18.320473, 38.900208, -86.376724],
                          [-18.320473, 38.900208, 86.3904],
                          [18.340351, -38.923126, -86.376724],
                          [18.340351, -38.923126, 86.3904],
                          [18.340351, 38.900208, -86.376724],
                          [18.340351, 38.900208, 86.3904]]],

                        [[[-98.16961, -85.56149, -32.21377],
                          [-98.16961, -85.56149, 29.116282],
                          [-98.16961, 94.26338, -32.21377],
                          [-98.16961, 94.26338, 29.116282],
                          [96.808556, -85.56149, -32.21377],
                          [96.808556, -85.56149, 29.116282],
                          [96.808556, 94.26338, -32.21377],
                          [96.808556, 94.26338, 29.116282]]]])
    cuboids = tf.convert_to_tensor(cuboids, dtype=tf.float32)
    if synthetic_image:
        camera_matrix = np.array([[[572.4114, 0., 325.2611],
                                   [0., 573.57043, 242.049],
                                   [0., 0., 1.]]])
    else:  # Samsung S22
        camera_matrix = [[[345.5395354181145, 0, 319.4688241083385],
                         [0, 345.25337576116874, 237.47917860129158],
                         [0, 0, 1]]]
    camera_matrix = tf.convert_to_tensor(camera_matrix, dtype=tf.float32)

    return opt, no_objects, height, width, input_segmentation_shape, checkpoint_path, \
           no_points, keypoints, camera_matrix, cuboids


def load_casapose(opt, no_objects, height, width, input_segmentation_shape, checkpoint_path):
    # Create model
    CASAPose = Classifiers.get(opt.modelname)
    ver_dim = opt.no_points * 2
    if opt.estimate_confidence:
        ver_dim += opt.no_points
    if opt.modelname == "pvnet":
        ver_dim = ver_dim * no_objects

    net = CASAPose(
        ver_dim=ver_dim,
        seg_dim=1 + no_objects,
        input_shape=(height, width, 3),
        input_segmentation_shape=input_segmentation_shape,
        weights="imagenet",
        base_model=opt.backbonename,
    )

    # Load model checkpoint
    checkpoint = tf.train.Checkpoint(network=net)  # , optimizer=optimizer)
    if opt.load_h5_weights is True:
        net.load_weights(opt.load_h5_filename + ".h5", by_name=True, skip_mismatch=True)
    elif opt.net != "":
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
    for layer in net.layers:
        layer.trainable = False
    # net.summary()
    return net


def inference_on_image(image, net, no_objects, no_points, keypoints, camera_matrix, opt):
    no_objects += 1
    # Instantiate model and run `image` through model to infer estimated poses
    net_input = [tf.expand_dims(image, 0)]
    output_net = net(net_input, training=False)
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


def draw_bb_inference(
        img,  # [x, x, 3]
        estimated_poses,  # [8, 3, 4]
        cuboids,  # [8, 1, 8, 3]
        camera_matrix,  # [1, 3, 3]
        path,
        file_prefix='251_output',
        gt_pose=None,  # [8, 1, 3, 4]
        normal=[0.5, 0.5],
):

    estimated_poses = tf.reshape(estimated_poses, [8, 3, 4])

    # image
    img_keypoints = tf.cast(((img * normal[1]) + normal[0]) * 255, dtype=tf.uint8).numpy()
    img_cuboids = img_keypoints.copy()

    eps = 1e-4

    for obj_idx, obj_pose in enumerate(estimated_poses):

        inst_idx = 0
        obj_pose_est = estimated_poses[obj_idx].numpy()
        instance_cuboids = cuboids[obj_idx][inst_idx].numpy()

        # draw bb - ESTIMATED
        valid_est = np.abs(np.sum(obj_pose_est)) > eps
        if valid_est:
            # print('est')
            # print(obj_pose_est.shape)
            transformed_cuboid_points2d, _ = project(instance_cuboids, camera_matrix.numpy(), obj_pose_est)
            transformed_cuboid_points2d = np.reshape(transformed_cuboid_points2d, (8, 2))
            draw_bb(transformed_cuboid_points2d, img_cuboids, (0, 255, 0))
        else:
            print(obj_idx)
            print('skipped obj est')

        # GT
        if gt_pose is not None:
            instance_pose_gt = gt_pose[obj_idx][inst_idx].numpy()
            valid_gt = np.abs(np.sum(instance_pose_gt)) > eps
            if valid_gt:
                print('gt')
                gt_pose_est = gt_pose[obj_idx].numpy()
                print(gt_pose_est[0].shape)
                transformed_cuboid_points2d_gt, _ = project(instance_cuboids, camera_matrix.numpy(),
                                                            gt_pose_est[0])
                print(gt_pose_est[0])
                draw_bb(transformed_cuboid_points2d_gt, img_cuboids, (255, 0, 0))
            else:
                print('skipped obj')

    # save image
    os.makedirs(path, exist_ok=True)
    img_cuboids = Image.fromarray((img_cuboids).astype("uint8"))
    img_cuboids.save(path + "/" + str(file_prefix) + "_cuboids_all.png")

    return img_cuboids


def load_image(path):
    # tensor = tf.random.uniform([448, 448, 3])

    img = tf.io.read_file(path)
    tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)
    return tensor


def image_transformation(img, cam_matrix):
    # Crop image to 448, 448, 3
    target_height, target_width = 448, 448
    offset_height = int((img.shape[-3] - target_height) / 2)
    offset_width = int((img.shape[-2] - target_width) / 2)
    img = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)

    # adjust the camera matrix according to https://stackoverflow.com/questions/74749690/how-will-the-camera-intrinsics-change-if-an-image-is-cropped-resized
    cam_matrix = cam_matrix.numpy()
    cam_matrix[0, 0, 2] = cam_matrix[0, 0, 2] - offset_width
    cam_matrix[0, 1, 2] = cam_matrix[0, 1, 2] - offset_height
    cam_matrix = tf.convert_to_tensor(cam_matrix, dtype=tf.float32)
    return img, cam_matrix


if __name__ == '__main__':
    h5_path = 'data/pretrained_models/result_w'
    meshes_dir = 'D:\\bop_toolkit\\data\\lm_251\\lm\\models'
    output_path = "output_251"
    synthetic_images = True  # True if inference on synthetic image, False if on real image (Samsung S22)

    # Input parameters
    opt, no_objects, height, width, input_segmentation_shape, checkpoint_path, \
    no_points, keypoints, camera_matrix, cuboids = input_parameters(h5_path, meshes_dir, synthetic_images)

    # Load Casapose model
    model = load_casapose(opt, no_objects, height, width, input_segmentation_shape, checkpoint_path)

    # Load RGB image, or read from webcam
    image = load_image("import_data/test/test/000001/000001.jpg")

    # Crop image
    # image, camera_matrix = image_transformation(image, camera_matrix)

    # Infer poses from image
    poses_est = inference_on_image(image, model, no_objects, no_points, keypoints, camera_matrix, opt)

    # Bounding box
    image_with_predictions = draw_bb_inference(image, poses_est, cuboids, camera_matrix, output_path)
