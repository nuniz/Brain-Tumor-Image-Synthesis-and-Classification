from utils import *

class BrainTumorDataset(Dataset):
    """
    Make the dataloader class

    """
    def __init__(self, images, labels):
        # images
        self.X = images
        # labels
        self.y = labels

        # Transformation for converting original image array to an image and then convert it to a tensor
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.ToTensor()
                                             ])

    def __len__(self):
        # return length of image samples
        return len(self.X)

    def __getitem__(self, idx):
        # perform transformations on one instance of X
        # Original image as a tensor
        data = self.transform(self.X[idx])

        # convert labels to the format output by our classifier
        # 1, 2, 3 = 0, 1, 2
        # store the network's understandable label as a tensor
        labels = torch.tensor((self.y[idx] - 1))

        # return the label and list of augmented images a tuple
        # 8 augmented images per sample will be returned
        return labels, data


class BrainTumorAugmentedDataset(Dataset):
    """
    expand the data with augmentation

    """
    def __init__(self, images, labels):
        # images
        self.X = images
        # labels
        self.y = labels

        # Transformation for converting original image array to an image and then convert it to a tensor
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.ToTensor()
                                             ])

        # Transformation for converting original image array to an image, rotate it randomly between -45 degrees and 45 degrees, and then convert it to a tensor
        self.transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(45),
            transforms.ToTensor()
        ])

        # Transformation for converting original image array to an image, rotate it randomly between -90 degrees and 90 degrees, and then convert it to a tensor
        self.transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(90),
            transforms.ToTensor()
        ])

        # Transformation for converting original image array to an image, rotate it randomly between -120 degrees and 120 degrees, and then convert it to a tensor
        self.transform3 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(120),
            transforms.ToTensor()
        ])

        # Transformation for converting original image array to an image, rotate it randomly between -180 degrees and 180 degrees, and then convert it to a tensor
        self.transform4 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(180),
            transforms.ToTensor()
        ])

        # Transformation for converting original image array to an image, rotate it randomly between -270 degrees and 270 degrees, and then convert it to a tensor
        self.transform5 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(270),
            transforms.ToTensor()
        ])

        # Transformation for converting original image array to an image, rotate it randomly between -300 degrees and 300 degrees, and then convert it to a tensor
        self.transform6 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(300),
            transforms.ToTensor()
        ])

        # Transformation for converting original image array to an image, rotate it randomly between -330 degrees and 330 degrees, and then convert it to a tensor
        self.transform7 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(330),
            transforms.ToTensor()
        ])

    def __len__(self):
        # return length of image samples
        return len(self.X)

    def __getitem__(self, idx):
        # perform transformations on one instance of X
        # Original image as a tensor
        data = self.transform(self.X[idx])

        # Augmented image at 45 degrees as a tensor
        aug45 = self.transform1(self.X[idx])

        # Augmented image at 90 degrees as a tensor
        aug90 = self.transform2(self.X[idx])

        # Augmented image at 120 degrees as a tensor
        aug120 = self.transform3(self.X[idx])

        # Augmented image at 180 degrees as a tensor
        aug180 = self.transform4(self.X[idx])

        # Augmented image at 270 degrees as a tensor
        aug270 = self.transform5(self.X[idx])

        # Augmented image at 300 degrees as a tensor
        aug300 = self.transform6(self.X[idx])

        # Augmented image at 330 degrees as a tensor
        aug330 = self.transform7(self.X[idx])

        # store the transformed images in a list
        # new_batch = [data]
        new_batch = [data, aug45, aug90, aug120, aug180, aug270, aug300, aug330]

        # convert labels to the format output by our classifier
        # 1, 2, 3 = 0, 1, 2
        # store the network's understandable label as a tensor
        labels = torch.tensor((self.y[idx] - 1))

        # return the label and list of augmented images a tuple
        # 8 augmented images per sample will be returned
        return labels, new_batch


def load_data_set(flags):
    """
    load the dataset

    Arguments
    ---------
    flag

    Outputs
    ---------
    train and validation set
    test set

    """
    Xt = []
    yt = []
    for paths in flags.data_path:
        data = pickle.load(open(paths, 'rb'))

        for features, labels in data:
            Xt.append(features)
            yt.append(labels)
    random.seed(51)
    if flags.OneDataSet:
        Xt, _, yt, __ = train_test_split(Xt, yt, test_size=0.001,shuffle=True, random_state=33)
        return BrainTumorDataset(Xt, yt)

    if flags.LoadTestSaparete:
        X_train_valid,  y_train_valid = Xt, yt
        del Xt, yt, data

        data_test = pickle.load(open(flags.data_test_path, 'rb'))
        Xt = []
        yt = []

        for features, labels in data_test:
            Xt.append(features)
            yt.append(labels)
        # random.seed(51)
        X_test,  y_test = Xt, yt
        del Xt, yt

    else:
        X_train_valid, X_test, y_train_valid, y_test = train_test_split(Xt, yt, test_size=flags.test_size,
                                                                        shuffle=True, random_state=33)
    print(f"Number of training samples: {len(X_train_valid)}")
    print(f"Number of testing samples: {len(X_test)} \n")

    if flags.Augmentation:
        train_valid_set = BrainTumorAugmentedDataset(X_train_valid, y_train_valid)
    else:
        train_valid_set = BrainTumorDataset(X_train_valid, y_train_valid)

    test_set = BrainTumorDataset(X_test, y_test)

    return train_valid_set, test_set
