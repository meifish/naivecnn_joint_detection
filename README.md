# naivecnn-joint-detection
Implementation of Human Joints Detection, using Keras to perform naive CNN model with 2D softmax

## Models
| model_name       | Encoder           | Prediction Model   | Prediction Form    |
|------------------|-------------------|--------------------|--------------------|
| vgg_      | VGG 16            | Softmax 2D         | Heatmap            |


## Getting Started

### Prerequisites
* Python 3
* Keras 2.0
* opencv for python

```shell
sudo apt-get install python-opencv
sudo pip install --upgrade keras
```

### Installing

Install the module
```shell
git clone https://github.com/meifish/naivecnn-joint-detection
python setup.py install
```

### Dataset
#### LSP (Leeds Sports Pose Dataset)

The dataset for training comes with this project. Images are in `dataset\lsp_dataset_original` folder. <br>
One can find more dataset information on LSP website: https://sam.johnson.io/research/lsp.html


## Use Python Module
