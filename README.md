# ResPerfNet

ResPerfNet is a ML-Based model to predict inference time of various neural networks on diverse deep learning accelerators.


## Usage

The commands for verify full model prection are as follows:

For TensorFlow:

    $ python3 verify_model.py --model_name lenet --batch_size 1 --network_name resperfnet --feature_transform boxcox --model_dirname ./model_tf --output_model_predict_dirname model_predict_tf --train_path tensorflow_data

For TensorRT:

    $ python3 verify_model.py --model_name lenet --batch_size 1 --network_name resperfnet --feature_transform boxcox --model_dirname ./model_trt --output_model_predict_dirname model_predict_trt --train_path tensorRT_data


Required arguments:
  ```
  -h, --help            show the help message and exit
  --network_name NETWORK_NAME, -n NETWORK_NAME
                        network name for training or testing
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        batch size you want to run
  --feature_transform {,boxcox}, -ft {,boxcox}
                        transofrmation for features
  --model_dirname MODEL_DIRNAME, -md MODEL_DIRNAME
                        model dirname
  --model_name {lenet,alexnet,vgg16}, -mn {lenet,alexnet,vgg16}
                        Neural networks models
  --train_path {tensorflow_data,tensorRT_data}, -tp {tensorflow_data,tensorRT_data}
                        The input main train path
  --output_model_predict_dirname OUTPUT_MODEL_PREDICT_DIRNAME, -ompdd OUTPUT_MODEL_PREDICT_DIRNAME
                        The dirname of the output model csv filename in
                        generation model data step
  ```

## Installation

ResPerfNet mainly uses the following dependencies:

- numpy
- pandas
- scikit
- scipy
- sklearn
- tensorflow
- termcolor

Use pip3 to install the depenencies with the specific versions:

```
pip3 install -r requirements.txt
```

Tested with Python 3.6.9 on Ubuntu 18.04.

## Model Architectures

ResPerfNet uses a specialized csv format for model architecture specification (See `data_full_model/model_csv` for details.), and it could be easily extend more neural network architectures by following the csv format.

Our experimentsal neural network architectures are as follows:

- LeNet
- AlexNet
- VGG-16

