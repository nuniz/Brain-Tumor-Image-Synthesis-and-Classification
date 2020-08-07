from utils import *
from flags import *
from data import *
from network import *

if __name__ == '__main__':
    device_name, device = check_cuda()
    flags = Flags()
    flags.OneDataSet = True

    # Load the dataset
    image_set = load_data_set(flags)
    net, total_params, total_trainable_params = make_model(flags)
    print("Defined model: " + flags.model_name + "  Number of parameters {}\{} \n"
          .format(total_trainable_params, total_params))

    # Remove the classification layer of the network
    net.fc = nn.Sequential()
    net.to(device)
    net.cuda()
    start_time = time.time()

    loader = torch.utils.data.DataLoader(image_set, pin_memory=True,
                                              batch_size=flags.test_batch_size)  # , num_workers = 4)
    torch.set_grad_enabled(False)
    net.eval()  # Skips dropout
    features = []
    labels_vector = []

    # Compute t-SNE
    for i, (labels, inputs) in enumerate(loader):
          labels = labels.cuda()
          inputs = inputs.cuda()
          if flags.Rgb:
                inputs = inputs.repeat(1, 3, 1, 1)
          # Forward prop
          output = net(inputs)
          features = smart_vstack(features,output.cpu().numpy())

          labels = torch.tensor(labels, dtype=torch.long, device=device).cuda()
          labels_vector = smart_concatenate(labels_vector, labels.cpu().numpy())

    tsne = TSNE(n_components=2).fit_transform(features)
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    labels_vector = [flags.target_label[int(label)] for label in labels_vector]

    tsne_df = pd.DataFrame({'X':tx,
                            'Y':ty,
                            'Class':labels_vector})

    plt.figure(figsize=(7,7))
    g= sns.scatterplot(x="X", y="Y",
                  hue="Class",
                  palette=['red','green','blue'],
                  legend='full',
                  data=tsne_df)

    if flags.LoadSavedModel:
        plt.title('t-SNE, for Resnet50 Custome trained')
    else:
        plt.title('t-SNE, for Resnet50 ImageNet pretrained')
    plt.show(g)

    x=2