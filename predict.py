import os
import cv2
import mat4py
import numpy as np
import scipy
import matplotlib.pyplot as plt
#from .models.jtNet import vgg16jtNet
from .train import find_latest_checkpoint
from .data_utils.data_loader import lsp_load_data

"""
def model_from_checkpoint_path( checkpoints_path ):

    latest_weights = find_latest_checkpoint( checkpoints_path )
    assert ( not latest_weights is None ) , "Checkpoint not found."

    #TODO: Change this to read from configuration
    model = vgg16jtNet(nclasses=14, input_height=224, input_width=224)

    print("loaded weights " , latest_weights )
    model.load_weights(latest_weights)
    return model


def model_from_weight( weight_path ):

    #TODO: Change this to read from configuration
    model = vgg16jtNet(nclasses=14, input_height=224, input_width=224)
    model.load_weights(weight_path)
    return model
"""

def predict_multiple(model, images_path, image_fnames, jts_map_path, checkpoints_path=None, load_weight_path=None):

    """
    if model is None and ( not checkpoints_path is None ):
        model = model_from_checkpoint_path(checkpoints_path)

    if model is None and ( not load_weight_path is None):
        model = model_from_weight(load_weight_path)
    """

    # Prepare Joint map
    mat = scipy.io.loadmat(jts_map_path)
    jts_map = mat.copy()
    jts_map = jts_map['joints']

    # jts_map needs to have order (nExamples, 2, nJoints) where 2 refers to `X` and `Y` coordinate of the joints
    jts_map = np.rollaxis(jts_map, 2)             # Note that 'jts_map' has shape (channel, jts, total_examples)
    jts_map = np.delete(jts_map, np.s_[-1::], 1)  # Remove the visibility channel # note: lsp original data use this one

    sample_generator = lsp_load_data(
        images_path,
        image_fnames,  # Change this text path to read from different file names set.
        jts_map,
        batch_size=20,
        n_classes=14,
        person_centered_cropped=True,
        resize_height=224,
        resize_width=224,
        return_originals=True
    )

    X, Y, ori_img, ori_jts = next(sample_generator)

    predict_result = []
    cropped_imgs = []

    for i in len(X):
        pr = model.predict( np.array([X[i]]) )[0]

        # Rotate the axis to be able to print by matplotlib
        pr = np.rollaxis(pr, 2, 0)
        pr = np.rollaxis(pr, 1, 3)
        predict_result.append(pr)
        cropped_imgs.append(Y[i])

    return predict_result, cropped_imgs


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap



def predict_plot(predict_result, cropped_imgs, idx=0):
    #TODO: make plot display multiple predictions altogether

    mycmap = transparent_cmap(plt.cm.rainbow)

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(cropped_imgs[idx], cv2.COLOR_BGR2RGB))

    for i in range(14):
        ax.imshow(predict_result[i], cmap=mycmap)

    plt.show()
