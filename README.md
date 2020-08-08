# brain_tumor_classification_dlmi_2020

## Description
Classification of Brain Tumors from MRI Images Using a Capsule Classifier and Image Synthesis.

Tel Aviv University. DLMI course (0553-5542), under the supervision of Prof. Hayit Greenspan.  July 2020.

## Abstract
The biggest problem for classifying magnetic resonance images (MRI) with deep learning techniques lies in the number of labelled data. 
Although recent works in the field of neural network have shown promising abilities of classifying brain tumors from brain MRI images, 
they used very deep and complex network architectures. 
Simpler network architecture requires fewer resources, and enable to use real time applications on mobile platforms. 
To this end, we introduce a new deep learning method to classify brain tumor of three tumor types. 
Our solution combines synthesise of additional data using a conditional generative model, with a new convolutional capsule architecture for the classification.

We evaluated the performance both qualitatively and quantitatively. 
Qualitatively, the conditional GAN was capable at synthesising images of different appearance, for the same underlying skull geometry. 
Moreover,  the features learned by the conditional GAN are often semantically meaningful groups, covering structures such as skull and tumor. 
The classificaiton performance was evaluated quantitatively using accuracy and F1 measures. 
The best results of 93\% accuracy and F1 score of 92\%, were obtained when the classifier pre-trained 
on 7,000 synthesised images and then trained on the original data using a 7-fold cross-validation. 
Our method performs as well as the Resnet50 state-of-the-art deep network, with 9x less parameters.

## Dataset
For all our experiments, we used the 'Brain Tumor' dataset, proposed by Cheng Jun et al.

The dataset can be found in this link: https://figshare.com/articles/brain_tumor_dataset/1512427

# How to run the code:

* Use the generative model to create train & test dataset pickle files.

    * Our modified pix2pix model creates syntethic data in addition to the original data. 

    * The code and instruction are given in the generative_model folder.

* Use the classification models (Resnet50/ Our capsule classifier) as described in the capsule classifier folder.

## Generative Model:
### Preprocess

### Training the models

### Evaluate and make the pickle file

## Capsule classifier:

### Pickle
* Make the pickle file of the dataset (as described in the generative model).
* Each element of the pickle is build from 2 fields:
   * Image (size of 512x512) and a label (0-2).
   * The elements are append into numpy array and than save into pickle file.
* You have to make a pickle of the train and the test datasets.

### Flags
* Change the following variables:

        Train - binary flag
        Test - binary flag
        check_name - The name of the loaded model
        LoadSavedModel - binary flag if you want to load pretrained model
        FreezeLayer - for the Resnet50 - binary flag.
        Model - "resnet" or "capsule"
        epochs - number of epochs
        model_dir - The folder you want to save the model to
        data_path - A path to the pkl file of the traninig data
        data_test_path - A path to the pkl of the test data
        
 * Keep the other variables values.

### Training/ Testing
* Run the script train.py.

### t-SNE
* Run the script tsne.py after you have a trained model.

## ACKNOWLEDGMENTS
Special thank to Cher Bass for publishing ’Image Synthesis with a Convolutional Capsule Generative Adversarial
Network’ [7] and inspired our work. 

We learned a lot from
the paper about capsule networks, convolutional capsules and
conditional GAN.

We also thank M. Bada for publishing the paper ’Classification of Brain Tumors from MRI Images Using a Convolutional
Neural Network’[3] that taught us about the biological background and helped us to define the problem better.

## Code
The code that we written [@pytorch] includes:

      • Data preprocessing [@Matlab].
      • Modified cGAN with a latent vector.
      • Capsule classifier model.
      • t-SNE.
      • Dataloader.
      • Cross-validation.
      • Train and evaluate different classifiers.
      • Resnet50 1D classifier, with an option to freeze layers.

We used the following code parts:

      • We based our cGAN on Pix2Pix pytorch implementation [https://github.com/phillipi/pix2pix].
      • Data augmentation (rotate) for the classifiers [https://github.com/aksh-ai/neuralBlack].
      • Convolutional capsule building block and dynamic routing pytorch implementation from CapsPix2Pix paper [https://github.com/CherBass/CapsPix2Pix].
      
 ## Capsule network architecture
![The suggested architecture for the classifier](https://raw.githubusercontent.com/nuniz/brain_tumor_classification_dlmi_2020/master/classifier.png)
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


## Generated images @rotate
![Syntethic Meningioma MRI images](https://raw.githubusercontent.com/nuniz/brain_tumor_classification_dlmi_2020/master/angle.png)
