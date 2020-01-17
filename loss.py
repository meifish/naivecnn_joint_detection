import tensorflow as tf

def l2_loss(y_true, y_pred):
    """
    The loss function generates ground truth heatmap on the fly.
    And then performs Least Square Error between ground truth and prediction.
    """

    # Generate Ground Truth Heatmap
    #########################################################################

    sigma = 50   # Should be able to customized through argument, but since
                 # the Keras custom loss function only allows two arguments
                 # to be `y_true` and `y_pred`, and therefore it couldn't be
                 # shown in arguments.
                 # TODO: Move the Heatmap GT generator to dataloader, so we
                 #       don't need to generate twice here and in evaluation.

    im_height = y_pred.shape[1]
    im_width = y_pred.shape[2]

    # set up meshgrid: (height, width, 2)
    meshgrid = tf.meshgrid(tf.range(im_height), tf.range(im_width))
    meshgrid = tf.cast(tf.transpose(tf.stack(meshgrid)), tf.float32)

    # set up broadcast shape: (batch_size, height, width, num_joints, 2)
    meshgrid_broadcast = tf.expand_dims(tf.expand_dims(meshgrid, 0), -2)
    y_true_broadcast = tf.expand_dims(y_true, 1)
    print("y_true shape", y_true_broadcast.shape)
    print("meshgrid_broadcast shape:", meshgrid_broadcast.shape)

    # L2 Loss (Least Square Error)
    #########################################################################
    diff = meshgrid_broadcast - y_true_broadcast
    print("diff shape:", diff.shape)

    ground = tf.exp(-0.5 * tf.reduce_sum(tf.square(diff), axis=-1) / sigma ** 2)
    print("ground shape:", ground.shape)

    # compute loss: first sum over (height, width), then take average over num_joints
    loss = tf.reduce_sum(tf.square(ground - y_pred), axis=[0, 1, 2])
    print("loss shape:", loss.shape)

    return tf.reduce_mean(loss)