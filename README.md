# Brain Tumor Classification DLMI 2020

## Description
This project, conducted at Tel Aviv University as part of the DLMI course (0553-5542) under the guidance of Prof. Hayit Greenspan in July 2020, focuses on the classification of brain tumors from MRI images.

### Methodology
The primary challenge in classifying MRI images lies in the scarcity of labeled data. Our solution involves a convolutional capsule architecture for classification coupled with data synthesis using a conditional generative model.


![Synthetic Meningioma MRI images](https://github.com/nuniz/brain_tumor_classification_dlmi_2020/blob/master/FakeAndMask1.gif)

![Synthetic Tumors](https://github.com/nuniz/brain_tumor_classification_dlmi_2020/blob/master/GAN_example.gif)

## Dataset
For all experiments, we utilized the 'Brain Tumor' dataset proposed by Cheng Jun et al. The dataset can be accessed here: https://figshare.com/articles/brain_tumor_dataset/1512427.

## Acknowledgments
We extend our gratitude to Cher Bass for publishing *Image Synthesis with a Convolutional Capsule Generative Adversarial Network*, which greatly inspired our work. Additionally, we draw inspiration from M. Bada's paper *Classification of Brain Tumors from MRI Images Using a Convolutional Neural Network*.

Our code, implemented in PyTorch, encompasses various components:

- Data preprocessing in Matlab.
- Modified conditional GAN with a latent vector.
- Capsule classifier model.
- t-SNE for visualization.
- Dataloader, cross-validation, and training/evaluation scripts for different classifiers.
- ResNet50 classifier, with an option to freeze layers.

We incorporated and modified parts of existing code:

- Conditional GAN based on Pix2Pix PyTorch implementation ([https://github.com/phillipi/pix2pix](https://github.com/phillipi/pix2pix)).
- Data augmentation (rotate) for classifiers ([https://github.com/aksh-ai/neuralBlack](https://github.com/aksh-ai/neuralBlack)).
- Convolutional capsule building block and dynamic routing PyTorch implementation from CapsPix2Pix paper ([https://github.com/CherBass/CapsPix2Pix](https://github.com/CherBass/CapsPix2Pix)).

### Modification of the Pix2Pix Code
We made several modifications to the Pix2Pix architecture:

- Changed the input mask to 1D channel (from 3D).
- Modified the network to handle image sizes of (512x512) instead of (256x256).
- Ensured compatibility with 16-bit depth.
- Added a latent vector component processed through a fully connected layer and concatenated with the input mask image.
- Saved generated images to a pickle file.

We modified the following files in the original Pix2Pix implementation:
- [network.py](generative_model/models/network.py).
- [test.py](generative_model/test.py).
- [base_dataset.py](generative_model/data/base_dataset.py).

## Code Instructions

### Generative Model

#### Preprocess
1. Run [process.m](preprocess/process.m) to create three directories for each label inside the 'brain_tumor' folder, making the data compatible for the Pix2Pix network.
2. Follow the instructions in [Datasets](generative_model/docs/datasets.md) for further processing to ensure compatibility.
3. Run [augmentation.m](preprocess/augmentation.m) to control the number of desired augmented images per label, performing tumor augmentation for later stages.

### Training the models

For each label train a GAN with [train.py](generative_model/train.py). as follows:
* Start the visdom server before the training:
```bash
python -m visdom.server
```
* Train a model:
```bash
python train.py --dataroot ./datasets/AB --name pix2pix_i --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --preprocess none --crop_size 512
```

### Image generation to pkl file

* Generate synthetic image (test the model): Data directory is of the augmented data from step 2 in Data Proprocessing.
* Run [test.py](generative_model/test.py). makes the required pickle file for the classifier. 
* It can be applied to the original data as well.

```bash
python test.py --dataroot ./datasets/AB --name pix2pix_i --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --preprocess none --crop_size 512 --dataset_mode single 
```

# Capsule Classifier

## Pickle
Generate the pickle file for the dataset as described in the generative model. Each element in the pickle consists of two fields:

- Image (size of 512x512) and a label (0-2).
- Append these elements into a numpy array and save them into a pickle file. Create pickles for both the training and testing datasets.

## Flags
Adjust the following variables in the `flags.py` file:

- `Train`: Binary flag
- `Test`: Binary flag
- `check_name`: The name of the loaded model
- `LoadSavedModel`: Binary flag to load a pretrained model
- `FreezeLayer`: Binary flag for ResNet50
- `Model`: "resnet" or "capsule"
- `epochs`: Number of epochs
- `model_dir`: The folder to save the model to
- `data_path`: Path to the pickle file of the training data
- `data_test_path`: Path to the pickle file of the test data

Keep the values of other variables unchanged.

## Train and Evaluate
Modify the flags in the [flags.py](capsule_classifier/flags.py) file, then run the [train.py](capsule_classifier/train.py) script.

## t-SNE
Execute [tsne.py](capsule_classifier/tsne.py) after you have a trained model.


 ## Capsule network architecture
 
         Defined model: capsule  Number of parameters 3827660\3827660 
         ----------------------------------------------------------------
                 Layer (type)               Output Shape         Param #
         ================================================================
                     Conv2d-1          [-1, 1, 512, 512]              26
                  LeakyReLU-2          [-1, 1, 512, 512]               0
                     Conv2d-3         [-1, 64, 256, 256]           1,088
                BatchNorm2d-4         [-1, 64, 256, 256]             128
             conv_cap_layer-5      [-1, 1, 64, 256, 256]               0
                     Conv2d-6        [-1, 128, 128, 128]         131,200
                BatchNorm2d-7        [-1, 128, 128, 128]             256
             conv_cap_layer-8     [-1, 1, 128, 128, 128]               0
                     Conv2d-9          [-1, 256, 64, 64]         524,544
               BatchNorm2d-10          [-1, 256, 64, 64]             512
            conv_cap_layer-11       [-1, 1, 256, 64, 64]               0
                    Conv2d-12          [-1, 512, 32, 32]       2,097,664
               BatchNorm2d-13          [-1, 512, 32, 32]           1,024
            conv_cap_layer-14       [-1, 1, 512, 32, 32]               0
                    Conv2d-15            [-1, 1, 29, 29]           8,193
                 LeakyReLU-16            [-1, 1, 29, 29]               0
                    Linear-17                  [-1, 841]         708,122
                 LeakyReLU-18                  [-1, 841]               0
                   Dropout-19                  [-1, 841]               0
                    Linear-20                  [-1, 420]         353,640
                 LeakyReLU-21                  [-1, 420]               0
                   Dropout-22                  [-1, 420]               0
                    Linear-23                    [-1, 3]           1,263
                LogSoftmax-24                    [-1, 3]               0
         ================================================================
         Total params: 3,827,660
         Trainable params: 3,827,660
         Non-trainable params: 0
         ----------------------------------------------------------------
         Input size (MB): 1.00
         Forward/backward pass size (MB): 184.04
         Params size (MB): 14.60
         Estimated Total Size (MB): 199.64
         ----------------------------------------------------------------
