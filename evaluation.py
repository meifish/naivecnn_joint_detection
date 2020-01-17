import tensorflow as tf

def accu_rate(y_true, y_pred):
    """
    For a given pixel radius normalized by the torso height of each sample, we count the number of images in
    the test set for which the distance between predicted position and ground truth is less than this radius.
    It was introduced in [MODEC: Multimodal Decomposable Models for Human Pose Estimation]
    (https://homes.cs.washington.edu/~taskar/pubs/modec_cvpr13.pdf).
    y_pred: tensor of size [n_images, height, width, n_joints]
    y_true: tensor of size [n_images, 1, 2, n_joints]
    normalized_radius: pixel radius normalized by the torso height of each sample
    """

    # TODO: Move this chunk to dataloader, so we don't need to generate the ground truth heatmap twice in
    #      loss function as well as accuracy mstrices.
    ########################################################################################################
    # Our y_true is (nExample, 1, 14, 2) generated from dataloader, where the second one is just a dummy dimension.
    sigma = 50

    im_height = y_pred.shape[1]
    im_width = y_pred.shape[2]

    # set up meshgrid: (height, width, 2)
    meshgrid = tf.meshgrid(tf.range(im_height), tf.range(im_width))
    meshgrid = tf.cast(tf.transpose(tf.stack(meshgrid)), tf.float32)

    # set up broadcast shape: (batch_size, height, width, num_joints, 2)
    meshgrid_broadcast = tf.expand_dims(tf.expand_dims(meshgrid, 0), -2)
    y_true_broadcast = tf.expand_dims(y_true, 1)

    diff = meshgrid_broadcast - y_true_broadcast

    ground = tf.exp(-0.5 * tf.reduce_sum(tf.square(diff), axis=-1) / sigma ** 2)

    ########################################################################################################
    normalized_radius = 25  # If the value is smaller, the stricter the evaluation is.
                            # The `normalized_dist` needs to be smaller than this value to be considered accurate.

    joints = 'all'

    def get_joints_coords(hm):
        hm_height, hm_width, n_joints = int(hm.shape[1]), int(hm.shape[2]), 14
        # we want to circumvent the fact that we can't take argmax over 2 axes
        hm = tf.reshape(hm, [-1, hm_height * hm_width, n_joints])
        coords_raw = tf.argmax(hm, axis=1)  # [n_images, n_joints]
        # Now we obtain real spatial coordinates for each image and for each joint
        coords_x = coords_raw // hm_width
        coords_y = coords_raw - coords_x * hm_width
        coords_xy = tf.stack([coords_x, coords_y], axis=1)  # [n_images, [coords_x, coords_y], jts]
        return tf.cast(coords_xy, tf.float32)

    lhip_idx, rsho_idx = 3, 8  # index of the joints
    pred_coords, true_coords = get_joints_coords(y_pred), get_joints_coords(ground)

    torso_distance = tf.norm(true_coords[:, :, lhip_idx] - true_coords[:, :, rsho_idx], axis=1,
                             keep_dims=True)  # [n_images]
    normalized_dist = tf.norm(pred_coords - true_coords, axis=1) * 100 / torso_distance  # [n_images, n_joints]
    if joints != 'all':
        norm_dist_list = []
        for joint in joints:
            norm_dist_list.append(normalized_dist[:, joint])
            normalized_dist = tf.stack(norm_dist_list, axis=1)
    detection_rate = 100 * tf.reduce_mean(tf.cast(tf.less_equal(normalized_dist, normalized_radius), tf.float32))
    return detection_rate