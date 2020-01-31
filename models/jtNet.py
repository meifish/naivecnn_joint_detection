import keras
from keras.models import *
from keras.layers import *
from types import MethodType
from ..train import train
from ..predict import predict_multiple, predict_plot
from ..activation import channel_softmax_2d


# VGG pre-trained weights for 'channel-last' images
pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

def get_vgg_encoder( input_height=224 ,  input_width=224 , pretrained='imagenet'):

    assert input_height%32 == 0
    assert input_width%32 == 0


    img_input = Input(shape=(input_height, input_width , 3 ))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1' )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool' )(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1' )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool' )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1' )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2' )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool' )(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool' )(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool' )(x)
    f5 = x


    if pretrained == 'imagenet':
        VGG_Weights_path = keras.utils.get_file( pretrained_url.split("/")[-1] , pretrained_url  )
        Model(  img_input , x  ).load_weights(VGG_Weights_path)

    return img_input , [f1 , f2 , f3 , f4 , f5 ]


def jtNet_decoder(  f , n_classes , n_up=3 ):

    assert n_up >= 2

    o = f
    o = ( ZeroPadding2D( (1,1) ))(o)
    o = ( Conv2D(512, (3, 3), padding='valid' ))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D( (2,2) ))(o)
    o = ( ZeroPadding2D( (1,1) ))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid' ))(o)
    o = ( BatchNormalization())(o)

    for _ in range(n_up-2):
        o = ( UpSampling2D( (2,2) ) )(o)
        o = ( ZeroPadding2D( (1,1) ))(o)
        o = ( Conv2D( 128 , (3, 3), padding='valid' ))(o)
        o = ( BatchNormalization())(o)

    o = ( UpSampling2D((2,2) ))(o)
    o = ( ZeroPadding2D((1,1) ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid' ))(o)
    o = ( BatchNormalization())(o)

    o =  Conv2D( n_classes , (3, 3) , padding='same' )( o )

    return o


def get_joint_regression_model(input, output):
    img_input = input   # Input is a tensor - before encoder
    o = output          # Output is also a tensor - from decoder

    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    # 'channels_last':
    output_height = o_shape[1]
    output_width = o_shape[2]
    input_height = i_shape[1]
    input_width = i_shape[2]
    n_classes = o_shape[3]

    o = (Activation(channel_softmax_2d))(o)  # apply softmax across the N class dimension for every pixel.
    model = Model(img_input, o)  # Consist of input and the softmax output
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.model_name = ""

    model.train = MethodType(train, model)  # Customize train function
    model.predict_joint = MethodType( predict_multiple , model )      # Customize predict function
    model.predict_plot = MethodType( predict_plot , model )
    # model.evaluate_joint = MethodType( evaluate , model )    # Customize evaluate function

    return model



def vgg16jtNet(nclasses, input_height=224, input_width=224 , encoder_level=3):

    encoder = get_vgg_encoder
    img_input , levels = encoder( input_height=input_height ,  input_width=input_width )

    feat = levels[ encoder_level ]
    o = jtNet_decoder(  feat , nclasses , n_up=4 )
    model = get_joint_regression_model(img_input , o)

    return model