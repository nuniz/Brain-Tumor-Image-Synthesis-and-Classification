from utils import *
from flags import *
from data import *
from network import *
from results import *


if __name__ == '__main__':
    ###############
    # Define the network    #
    ###############
    device_name, device = check_cuda()
    flags = Flags()
    results = Results()

    train_valid_set, test_set = load_data_set(flags)
    net, results.total_params, results.total_trainable_params = make_model(flags)
    print("Defined model: " + flags.model_name + "  Number of parameters {}\{} \n"
          .format(results.total_trainable_params, results.total_params))


    net.to(device)
    net.cuda()

    if flags.model_name == "capsule":
        summary(net,(1,512,512))
    else:
        summary(net, (3, 512, 512))

    loss_fun = nn.CrossEntropyLoss().cuda()
    if flags.FreezeLayers:
        print ("Freeze layers, num of layers: {}".format(flags.num_of_freeze_layers))
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                    momentum=flags.momentum, lr=flags.lr)
    else:
        optimizer = torch.optim.SGD(net.parameters(), momentum=flags.momentum, lr=flags.lr)
    start_time = time.time()

    ###############
    # Training    #
    ###############
    if flags.Train:
        train_valid_len = len(train_valid_set)
        train_valid_idx = np.arange(train_valid_len)
        np.random.seed(0)
        np.random.shuffle(train_valid_idx)
        train_accuracy, train_loss, current_k_fold = 0,0,0
        train_valid_loader = None
        shuffle_fold_flag = True
        for epoch in range(flags.epochs):
            if shuffle_fold_flag:
                train_idx, valid_idx, current_k_fold = k_vold_index_divide(train_valid_idx, train_valid_len,
                                                                           flags.k_cross_validation, current_k_fold)
                shuffle_fold_flag = flags.CrossValidation
            for phase in flags.train_phase:
                if train_valid_loader is not None:
                    del train_valid_loader
                train_valid_loader = [torch.utils.data.DataLoader(train_valid_set, pin_memory=True,
                                                                 batch_size=flags.batch_size,
                                                                 sampler=SubsetRandomSampler(train_idx)),
                                      torch.utils.data.DataLoader(train_valid_set, pin_memory=True,
                                                           batch_size=flags.test_batch_size,
                                                           sampler=SubsetRandomSampler(valid_idx))]
                train_batch = int(int(train_valid_loader[0].sampler.indices.size) / flags.batch_size)
                valid_batch = int(int(train_valid_loader[1].sampler.indices.size) / flags.batch_size)
                if phase == 'train':
                    print("{}/{} Fold, Batch size {} ,Train batches {}, Valid batches {}"
                          .format(current_k_fold, flags.k_cross_validation,
                                  flags.batch_size, train_batch, valid_batch))

                loss_arr = []
                total_samples, correct_samples = 0, 0
                if phase == 'train':
                    num_of_batch = train_batch
                    torch.set_grad_enabled(True)
                    net.train()  # Update weights for
                    loader_idx = 0
                else:
                    torch.set_grad_enabled(False)
                    net.eval()  # Skips dropout
                    if phase == 'train_valid':
                        num_of_batch = train_batch
                        loader_idx = 0
                    else:
                        num_of_batch = valid_batch
                        loader_idx = 1

                for i, (labels, aug_inputs) in enumerate(train_valid_loader[loader_idx]):
                    labels = labels.cuda()
                    for inputs in aug_inputs:
                        inputs = inputs.cuda()
                        if flags.Rgb:
                            inputs = inputs.repeat(1, 3, 1, 1)
                        # Forward prop
                        output = net(inputs)
                        labels = torch.tensor(labels, dtype=torch.long, device=device).cuda()
                        loss = loss_fun(output, labels)

                        if phase == 'train':
                            # Back prop
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        if phase == 'valid' or phase == 'train_valid':
                            loss_arr.append(loss.item())
                            predicts = torch.max(output, 1)[1]
                            correct_samples += (predicts == labels).sum().item()
                            total_samples += len(labels)

                    if i % (int(num_of_batch/4)) == 0 and i > 0:
                        msg = "eopch:{} / {}, batch {} / {} , " \
                              "time is {} min, current loss {}".format(epoch + 1, flags.epochs,
                                                               i, num_of_batch,
                                                               int((time.time() - start_time) / 60),
                                                               loss.item())
                        print(phase + ".. " + msg)

                if phase == 'train_valid':
                    train_accuracy, train_loss = correct_samples / total_samples * 100, np.average(loss_arr)
                if phase == 'valid':
                    valid_accuracy, valid_loss = correct_samples / total_samples * 100, np.average(loss_arr)
                    results.update(train_loss, valid_loss, train_accuracy, valid_accuracy,
                                   epoch + 1)
            msg = "Finished epoch :{} at {}. valid accuracy: {}" \
                  " train accuracy: {} \n".format(epoch + 1, datetime.datetime.now(), valid_accuracy, train_accuracy)
            print(msg)

            if results.best_score < valid_accuracy:
                if epoch > 0:
                    remove_file(model_path)
                results.best_score = valid_accuracy
                model_path = os.path.join(flags.model_dir,
                                        flags.model_name + "_epoch{}_score{}.pt".format(
                                            epoch + 1, results.best_score))
                flags.check_name = model_path
                print ("Save model: " + model_path + "\n")
                torch.save(net.state_dict(), model_path)

        # Plot Accuracy during training
        plot_loss_accuracy(flags, results)
        # del net

    ###############
    # Test    #
    ###############
    if flags.Test:
        test_loader = torch.utils.data.DataLoader(test_set, pin_memory=True,
                                                  batch_size=flags.test_batch_size)#, num_workers = 4)
        torch.set_grad_enabled(False)
        net.eval()  # Skips dropout
        record_flag = False
        for i, (labels, inputs) in enumerate(test_loader):
            labels = labels.cuda()
            inputs = inputs.cuda()
            if flags.Rgb:
                inputs = inputs.repeat(1, 3, 1, 1)
            # Forward prop
            output = net(inputs)
            labels = torch.tensor(labels, dtype=torch.long, device=device).cuda()
            predicts = torch.max(output, 1)[1]
            if record_flag is False:
                labels_arr = labels
                predicts_arr = predicts
                record_flag = True
            else:
                labels_arr = torch.cat((labels_arr, labels), dim = 0)
                predicts_arr = torch.cat((predicts_arr, predicts), dim = 0)

        cm, report = evaluate_metrics(labels_arr.cpu().data.numpy(),
                                      predicts_arr.cpu().data.numpy(),
                                      flags.target_label, flags)

    plt.show()
    x=2