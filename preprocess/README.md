### Preprocess
The brain MRI dataset has grayscale images inside brain_tumor folder where each datum is a .mat file.

* Run preprocess.m 
    * it will make 3 directories for each label. 
    * In order to make the above data compatible for Pix2Pix network, further processing is needed - follow the [Datasets](docs/datasets.md).
* Run augmentation.m
    * Control the number of desired augmented image per label.
    * Perform a tumor augmentation for later stage by running augmentation.m
