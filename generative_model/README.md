
#### Modiciation of the pix2pix code
We made the following modification to the pix2pix architecture:
* We based on the pix2pix architecture for the cGAN [8].
* We changed the input mask to 1D channel (instead of 3D
channel) and modified the network so it could handle image
size of (512x512) instead of (256x256). 
* We also made thenetwork compatible to depth of 16-bit. 
* Furthermore, we added
a latent vector component that was processed through a fully
connected layer and then concatenated with the input mask
image.
* Save the result to pickle file.

We relied on the original pix2pix implementation and modified the following files:
* network.py
* test.py
* base_dataset.py


### Training the models

For each label train a GAN with train.py as follows:
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
* Run test.py makes the required pickle file for the classifier. 
* It can be applied to the original data as well.

```bash
python test.py --dataroot ./datasets/AB --name pix2pix_i --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --preprocess none --crop_size 512 --dataset_mode single 
```
