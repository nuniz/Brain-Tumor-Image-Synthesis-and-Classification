from utils import *

class Flags():
    def __init__(self):
        self.Train = False
        self.Test = True
        self.CrossValidation = True
        self.LoadSavedModel = True
        self.FreezeLayers = False
        self.batch_norm = True
        self.SaveFig = True
        self.LoadTestSaparete = True
        self.OneDataSet = False
        self.Augmentation = True
        self.LoadImageNet = False

        self.model_name = "resnet"
        self.epochs = 20
        self.train_phase = ['train', 'train_valid', 'valid']
        self.batch_size = 4
        self.test_batch_size = 10
        self.test_size = 0.15
        self.train_size = 1-self.test_size
        self.target_label = ["meningioma", "glioma", "pituitary"]
        self.class_number = len(self.target_label)
        self.model_dir = F"C:\\Users\\zorea\\PycharmProjects\\DLMI\\Checkpoints"

        if self.LoadTestSaparete:
            self.data_path = ["C:\\Users\\zorea\\PycharmProjects\\DLMI\\Data\\TrainData.pkl"]
        else:
            self.data_path = ["C:\\Users\\zorea\\PycharmProjects\\DLMI\\Data\\classifier_data.pickle"]

        # self.data_test_path = F"C:\\Users\\zorea\\PycharmProjects\\DLMI\\Data\\classifier_data.pickle"
        self.data_test_path = F"C:\\Users\\zorea\\PycharmProjects\\DLMI\\Data\\TestData.pkl"

        if self.LoadSavedModel:
            self.LoadImageNet = False

        if self.model_name == "capsule":
            # self.check_name = r'capsule_epoch25_score83.56164383561644.pt'  # r
            # self.check_name = r'capsule_epoch27_score90.55555555555556.pt'  # r
            # self.check_name = r'capsule_epoch25_score89.34948979591837.pt'  # r
            # self.check_name = r"capsule_epoch27_score95.36082474226805.pt"
            self.check_name = r"capsule_epoch14_score94.48992443324937.pt"
            self.Rgb = False
            self.FreezeLayers = False
        elif self.model_name == "resnet":
            # self.check_name = r'resnet_epoch11_score99.04109589041096.pt'  # r
            self.check_name = r'resnet_epoch21_score92.7295918367347.pt'  # r
            self.Rgb = True
        else:
            print("Error: Don't have model.\n")
            sys.exit(1)

        if not self.Train and not self.OneDataSet:
            self.LoadSavedModel = True

        self.num_of_freeze_layers = 8
        self.k_cross_validation = 7
        self.lr = 3e-4
        self.lr = 5e-5
        self.momentum = 0.9
        self.dropout = 0.4
        self.leaky_relu_slope = 0.2
