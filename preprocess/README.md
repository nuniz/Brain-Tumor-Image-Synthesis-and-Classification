### Preprocess
The brain MRI dataset has grayscale images inside brain_tumor folder where each datum is a .mat file.

* Run [process.m](preprocess/process.m). 
    * it will make 3 directories for each label. 
    * In order to make the above data compatible for Pix2Pix network, further processing is needed - follow the [Datasets](generative_model/docs/datasets.md).
* Run [augmentation.m](preprocess/augmentation.m)
    * Control the number of desired augmented image per label.
    * Perform a tumor augmentation for later stage by running augmentation.m
