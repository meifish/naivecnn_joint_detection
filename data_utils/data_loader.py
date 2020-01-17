# Global library
import sys
import os
import numpy as np
import cv2
import glob
import itertools
import random
import pickle
from scipy.stats import multivariate_normal
from tqdm import tqdm
from .augmentation import augment_seg


IMAGE_ORDERING = 'channels_last'
#random.seed(0)
#class_colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(5000)  ]


def joint_dict(joint_dic = None):

    if not joint_dic:
        joint_dict  = {
            "r_ankel"   : 0,
            "r_knee"    : 1,
            "r_hip"     : 2,
            "l_hip"     : 3,
            "l_knee"    : 4,
            "l_ankle"   : 5,
            "r_wrist"   : 6,
            "r_elbow"   : 7,
            "r_shoulder" : 8,
            "l_shoulder" : 9,
            "l_elbow"    : 10,
            "l_wrist"    : 11,
            "neck"       : 12,
            "head_top"   : 13
        }
    
    assert (type(joint_dic) == "dict"), "The input joint_dic has to be a dictionary."
    
    return joint_dic



# Deprecated
def generate_heatmap_array( joint_map, img_path, heatmap_path, channel_last = True ):
    
    """
    joint_map: (2, nClass, nExample) ndarray, which has the GT information of X position, Y position.
               The ndarray should has ground truth information for the N images, and stored in the shape of (3, nClass, nExample) 
               
               - 2: index 0 for X-position (float); index 1 for Y-position (float).
               - nClass: Number of joints
               - nExample: Number of examples.
               
    img_path: The directory of the original images. Note: The images in the directory needs to have indecies starting from '1' instead of '0'.
    """
    
    # Step 1: Read in the path of the images
    images = glob.glob( os.path.join(img_path,"*.jpg")  ) + glob.glob( os.path.join(img_path,"*.png")  ) +  glob.glob( os.path.join(img_path,"*.jpeg")  )
    
    # Step 2: Generate a dict of Ground Truth heatmap ndarray for corresponding image.
    
    gt_dict = {}
    
    for img in tqdm(images):
        im_bnme = os.path.basename(img)
        im = cv2.imread(img)
        im_height = im.shape[0]
        im_width = im.shape[1]
        im_idx = int(im_bnme.lstrip('im').replace('.jpg', '')) - 1

        num_joints = 14
        X = 0
        Y = 1
        
        # Place-holder x, y grid (same size as its original image) for Gaussian distribtion
        x, y = np.mgrid[0: im_height, 0:im_width]        
        gaussian_ndarray = None
        
        for jt in range(num_joints):
            # Read in Ground Truth joints location.
            joint_x = joint_map[X][jt][im_idx]
            joint_y = joint_map[Y][jt][im_idx]        
    
            # Create Gaussian distribution on the location (joint_x, joint_y)
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x; 
            pos[:, :, 1] = y
            # The argument ([center_y, center_x], [cov], [cov]), where cov is the 'covariance of [x-x, x-y], [y-x, y-y] respectively.
            # Change the cov will change the size of the Gaussian bulb.
            gaussian = multivariate_normal([joint_y, joint_x], [[0.1 * y.max(), 0.0], [0.0, 0.1 * x.max()]])
  
            # Stack `nClass` heatmap image into a ndarray of shape: (nClass, height, width)
            if type(gaussian_ndarray) == type(None):
                gaussian_ndarray = np.array([gaussian.pdf(pos)])
            else:
                gaussian_ndarray = np.concatenate((gaussian_ndarray, np.array([gaussian.pdf(pos)])), axis = 0)

        if channel_last:
            gaussian_ndarray = np.rollaxis(gaussian_ndarray, 0, 3)
        
        gt_dict = { im_bnme: gaussian_ndarray }
        
        pickle_filename = "gt.pickle"
        pickle_out = open(os.path.join(heatmap_path, pickle_filename), "wb")
        pickle.dump(gt_dict, pickle_out)
        pickle_out.close()
    
    return gt_dict


def get_filenames_from_paths( image_fnames_txt ):

    f = open(image_fnames_txt, "r")
    images = [i.strip() for i in f.readlines()]
    print("Reading", len(images), "files from", os.path.basename(image_fnames_txt), "...")
    f.close()

    ret = []

    for im in images:
        ret.append(im)
    print("Reading...", ret)

    return ret

# Deprecated
def get_pairs_from_paths( images_path , segs_path ):
    images = glob.glob( os.path.join(images_path,"*.jpg")  ) + glob.glob( os.path.join(images_path,"*.png")  ) +  glob.glob( os.path.join(images_path,"*.jpeg")  )
    segmentations  =  glob.glob( os.path.join(segs_path,"*.png")  )

    segmentations_d = dict( zip(segmentations,segmentations ))

    ret = []

    for im in images:
        seg_bnme = os.path.basename(im).replace(".jpg" , ".png").replace(".jpeg" , ".png")
        seg = os.path.join( segs_path , seg_bnme  )
        assert ( seg in segmentations_d ),  (im + " is present in "+images_path +" but "+seg_bnme+" is not found in "+segs_path + " . Make sure annotation image are in .png"  )
        ret.append((im , seg) )

    return ret

# Deprecated
def get_image_arr( path , width , height , imgNorm="sub_mean" , odering='channels_first' ):


    if type( path ) is np.ndarray:
        img = path
    else:
        img = cv2.imread(path, 1)

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, ( width , height ))
        img = img.astype(np.float32)
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
    elif imgNorm == "divide":
        img = cv2.resize(img, ( width , height ))
        img = img.astype(np.float32)
        img = img/255.0

    if odering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


# Deprecated
def get_segmentation_arr( path , nClasses ,  width , height , no_reshape=False ):

    seg_labels = np.zeros((  height , width  , nClasses ))

    if type( path ) is np.ndarray:
        img = path
    else:
        img = cv2.imread(path, 1)

    img = cv2.resize(img, ( width , height ) , interpolation=cv2.INTER_NEAREST )
    img = img[:, : , 0]

    for c in range(nClasses):
        # One-hot encoding
        seg_labels[: , : , c ] = (img == c ).astype(int)

    if no_reshape:
        return seg_labels

    seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
    return seg_labels


# Deprecated
def verify_segmentation_dataset( images_path , segs_path , n_classes ):

    img_seg_pairs = get_pairs_from_paths( images_path , segs_path )

    assert len(img_seg_pairs)>0 , "Dataset looks empty or path is wrong "

    for im_fn , seg_fn in tqdm(img_seg_pairs) :
        img = cv2.imread( im_fn )
        seg = cv2.imread( seg_fn )

        assert ( img.shape[0]==seg.shape[0] and img.shape[1]==seg.shape[1] ) , "The size of image and the annotation does not match or they are corrupt "+ im_fn + " " + seg_fn
        assert ( np.max(seg[:,:,0]) < n_classes) , "The pixel values of seg image should be from 0 to "+str(n_classes-1) + " . Found pixel value "+str(np.max(seg[:,:,0]))

    print("Dataset verified! ")
    
    
# Deprecated
def image_segmentation_generator( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width  , do_augment=False ):

    img_seg_pairs = get_pairs_from_paths( images_path , segs_path )
    random.shuffle( img_seg_pairs )
    zipped = itertools.cycle( img_seg_pairs  )

    while True:
        X = []
        Y = []
        for _ in range( batch_size) :
            im , seg = next(zipped) 

            im = cv2.imread(im , 1 )
            seg = cv2.imread(seg , 1 )

            if do_augment:
                img , seg[:,:,0] = augment_seg( img , seg[:,:,0] )

            X.append( get_image_arr(im , input_width , input_height ,odering=IMAGE_ORDERING )  )
            Y.append( get_segmentation_arr( seg , n_classes , output_width , output_height )  )

        yield np.array(X) , np.array(Y)


def crop_image_jt(img, jt_map, center_x, center_y, crop_height, crop_width):
    """
    Given the image and the (X, Y) location, this function crops the image centered from the given (X, Y) with the given size specified by the
    crop_width and crop_height. 
    
    Note: 1/ If the image is smaller than the speicified cropping width and height, then the it returns None
          2/ If the desired center (X, Y) is too close to the image edge so that the cropped region would exceed the image's boundary, i.e.,
          then the boundary of the original image will be used. For example, if a given image has size (100, 100), and with the following args:
          crop_width = 60, crop_height = 60, (X, Y) = (80, 80). Then the image will be cropped as: img[40:100, 40:100], and (X, Y) would no longer
          being the center of the image.
    """

    _h = img.shape[0]
    _w = img.shape[1]
    
    _jt_map = jt_map.copy()
    
    if (_h <= crop_height) or (_w <= crop_width):
        print("image too small.")
        print("image h:", _h, "image w:", _w)
        print("cropped height:", crop_height, "cropped width:", crop_width)
        return None

    # Try cropping the image into square box and keep the original ratio.
    crop_size = int(max(crop_width, crop_height))
    if (_h >= crop_size) and (_w >= crop_size):
        crop_height = crop_size
        crop_width = crop_size

    center = ( center_y , center_x )
    bound = ( crop_height, crop_width )

    start = np.fromiter(map(lambda a, da: round(a-da//2), center, bound), dtype=np.int)

    offset = start<0                       # (False, True) if start is (10, -10)
    offset = start * offset * (-1)         # (10, -10) * (0, 1) * (-1)  -> (0, 10)
    
    start = start + offset                 # (10, -10) + (0, 10) = (10, 0)
    end = start + bound
    
    slices = tuple(map(slice, start, end))


    _jt_map[1] -= start[0]  
    _jt_map[0] -= start[1]   

    return (img, center, slices, img[slices], _jt_map)


def lsp_load_data( images_path,
                   image_fnames_txt,
                   jts_map,  
                   batch_size,  
                   n_classes , 
                   person_centered_cropped = True ,
                   resize_height = None, resize_width = None,
                   return_originals = False):
    """
    @images_path:         The folder path of the images
    @images_fnames_txt:   A text file in which every line is the base name of one image, i.e., im0001.jpg
    @jts_map:             The ndarray of joints location (X-Y coordinates) in the image. The size should be (nExample, 2, nJoint)
    @batch_size:          The number of images for a batch. It affects how many ndarray the generator will yield each time.
    @n_classes:           The number of prediction class. For the LSP dataset, it's 14 for the number of joints.
    @person_centered_cropped:     Cropped the image into a square shape, with the center of the bounding box of the target person. 
                          The bounding box information did not come with the LSP dataset, and therefore it is estimated by the joints location.
                          For example, the boundary is formed by "top-leftmost", "top-rightmost", "bottom-leftmost", and "bottom-rightmost".
                          The center is then "1/2 * (bottom - top, right - left)". The cropped size is 384 x 384 pixels.
    @resize_height:       Resize/ scale the final image height. Using 384 pixels if not otherwise specified.
    @resize_width:        Resize/ scale the final image width. Using 384 pixels if not otherwise specified.
    
    Note: 1/ If one would like to use uncropped images from LSP, the person_centered_cropped can be set to False. However, in this way, one needs to 
          specify `resize_height` and `resize_width` to unify the training images size in order to fit the model.
          2/ `person_centered_cropped` and `resize` can work together. It will implement the central cropped first, and then apply resizing.
    """

    f = open(image_fnames_txt, "r")
    images = [i.strip() for i in f.readlines()]
    print("Reading", len(images), "files from", os.path.basename(image_fnames_txt), "...")
    f.close()

    random.shuffle( images )
    images = itertools.cycle( images )

    while True:
        X = []
        Y = []
        I = []
        J = []
            
        for i in range( batch_size) :
            im_fname = os.path.join(images_path, next(images))
            im_bname = os.path.basename(im_fname)
            im_index = int(im_bname.lstrip('im').replace('.jpg', '')) - 1

            img = cv2.imread(im_fname , 1 )      # (height, width, 3)  -> image: channel-last
            jt_map = jts_map[im_index]           # (2, 14)             -> The (x, y) coordinates for the 14 joints

            im_height = img.shape[0]
            im_width = img.shape[1]


            # Person centered crop
            if person_centered_cropped:
                min_x = np.amin(jt_map[0])   
                max_x = np.amax(jt_map[0])   
                min_y = np.amin(jt_map[1])  
                max_y = np.amax(jt_map[1])   
                center_x = round(min_x + 1/2 * (max_x - min_x))
                center_y = round(min_y + 1/2 * (max_y - min_y))
                
                person_width = (center_x - min_x) * 2
                person_height = (center_y - min_y) * 2
                padding = 10

                #crop_size = int(max(person_width, person_height) + padding)
                crop_height = int(person_height + person_height * 0.1)
                crop_width = int(person_width + person_width * 0.1)

                if (im_height <= crop_height) or (im_width <= crop_width):
                    print("image smaller than crop size. passed")
                    print("file name:", im_bname)
                    print("image h:", im_height, "image w:", im_width)
                    continue
                elif crop_image_jt(img, jt_map, center_x, center_y, crop_height, crop_width):
                    img, center, slices, img_crop, jt_map_crop = crop_image_jt(img, jt_map, center_x, center_y, crop_height, crop_width)
                else:
                    print("image " + im_bname + " return None in cropping with height " + str(im_height) + " and width " + str(im_width))
                    print("crop_height:" + str(crop_height) + " crop_width: " + str(crop_width))
                    print("jts:")
                    print(jt_map)
                    continue


            else:
                img_crop = img
                jt_map_crop = jt_map.copy()


            if (img_crop.shape[0] == 0) or (img_crop.shape[1] == 0):
                continue

            # Resize
            if (resize_width != None) and (resize_height != None):
                crop_width = img_crop.shape[1]
                crop_height = img_crop.shape[0]

                img_crop = cv2.resize(img_crop, ( resize_width , resize_height ))
                jt_map_crop[0] = jt_map_crop[0] * ( resize_width / crop_width)
                jt_map_crop[1] = jt_map_crop[1] * ( resize_height / crop_height)
                
            # TODO: Add more image normalization options here
            img = img / 255
            
            jt_map_crop = np.rollaxis(jt_map_crop, 0, 2)   #(14,2)
            jt_map      = np.rollaxis(jt_map, 0, 2)        #(14,2)



            X.append( img_crop )
            Y.append( jt_map_crop )     #(nExamples, 14, 2)
    
            I.append( img )
            J.append( jt_map )          #(nExamples, 14, 2)

            #print("file name:", im_fname)


        # Add dumpy dimension: since Keras requires the y_pred and y_true to have the same dimension
        # here our y_true is a (nExamples, 14, 2) array, where the y_pred would be a (nExamples, height, width, 14) array.
        # so y_true has dimension 3 while y_pred has 4. Therefore, we add a dumpy dimension and it becomes (nExamples, 1, 14, 2)
        Y = np.expand_dims(Y, axis=1)   #(nExamples, 1, 14, 2)
        J = np.expand_dims(J, axis=1)   #(nExamples, 1, 14, 2)

        #print("Y dims:", Y.shape)
        #print("J dims:", J.shape)
        #print("iter:", i)


        if return_originals == False:
            yield np.array(X) , np.array(Y)
            
        else:
            yield np.array(X) , np.array(Y), np.array(I), np.array(J)