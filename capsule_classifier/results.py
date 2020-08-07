class Results():
    def __init__(self):
        self.train_loss = []
        self.valid_loss = []
        self.train_accuracy = []
        self.valid_accuracy = []
        self.epoch = []
        self.cm = []
        self.report = []
        self.best_score = 0
        self.test_accuracy = None
        self.confusion_matrix = None
        self.total_params = 0
        self.total_trainable_params = 0

    def update(self, train_loss, valid_loss,
               train_accuracy, valid_accuracy, epoch):
        self.train_loss.append(train_loss)
        self.valid_loss.append(valid_loss)
        self.train_accuracy.append(train_accuracy)
        self.valid_accuracy.append(valid_accuracy)
        self.epoch.append(epoch)