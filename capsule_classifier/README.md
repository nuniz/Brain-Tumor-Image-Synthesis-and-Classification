## Capsule classifier:

### Pickle
* Make the pickle file of the dataset (as described in the generative model).
* Each element of the pickle is build from 2 fields:
   * Image (size of 512x512) and a label (0-2).
   * The elements are append into numpy array and than save into pickle file.
* You have to make a pickle of the traninig and the testing datasets.

### Flags
* Change the following variables:

        Train - binary flag
        Test - binary flag
        LoadSavedModel - binary flag if you want to load pretrained model
        check_name - The name of the loaded model
        FreezeLayer - for the Resnet50 - binary flag.
        Model - "resnet" or "capsule"
        epochs - number of epochs
        model_dir - The folder you want to save the model to
        data_path - A path to the pkl file of the traninig data
        data_test_path - A path to the pkl of the test data
        
 * Keep the other variables for the original value

### Training/ Testing
* Run the script train.py.

### t-SNE
* Run the script tsne.py after you have a trained model.
