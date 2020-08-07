from utils import *
from flags import *
from data import *
from network import *

device_name, device = check_cuda()
flags = Flags()
flags.LoadSavedModel = True
flags.OneDataSet = True

image_set = load_data_set(flags)
net, total_params, total_trainable_params = make_model(flags)
net.to(device)
net.cuda()
net.eval()
torch.set_grad_enabled(False)


def batch_predict(inputs, net = net):
    inputs = inputs.astype('float32')
    inputs = torch.tensor(inputs)
    inputs = torch.transpose(inputs, 1,3)
    if not flags.Rgb:
        inputs = inputs[:,1,:,:]
        inputs = torch.unsqueeze(inputs,1)
        # inputs = torch.transpose(inputs, 0, 1)
    inputs = 2*(inputs -1)
    output = net(inputs.cuda())
    probs = torch.exp(output)
    return probs.detach().cpu().numpy()


def plot_images_lime(inputs, labels, flags, label_to_explain = 1, num_features=1, num_samples=20, columns=4, rows=5):
    class_names = flags.target_label
    fig = plt.figure(figsize=(15, 15))
    end = int(flags.test_batch_size + 1)
    for i in range(1, end):
        explanation = calculate_lime(0.5*(1+inputs[i-1]), label_to_explain, num_features, num_samples)
        # explanation = calculate_lime(127.5*(1+inputs[i-1]), label_to_explain, num_features, num_samples)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=num_features, hide_rest=True)
        gigapixels = mark_boundaries(np.zeros((explanation.image.shape)), explanation.segments)
        img_boundry = mark_boundaries(temp/255.0, mask)
        exp_img =  cv2.normalize(explanation.image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = 0.4*(exp_img) + 1 * img_boundry + 0.1 * gigapixels
        pred_class = class_names[explanation.top_labels[0]]
        ax = fig.add_subplot(rows, columns, i)
        ax.title.set_text("Predicted: {}, True Label: {}".format(pred_class, flags.target_label[int(labels[i-1])]))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img)
    plt.show()


def calculate_lime(img, label_to_explain, num_features, num_samples):
    explainer = lime_image.LimeImageExplainer()
    img = torch.Tensor.cpu(img).detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img.astype('float')
    return explainer.explain_instance(img,
                                      batch_predict,
                                      labels=np.array([label_to_explain]),
                                      num_features=num_features,
                                      hide_color=0,
                                      num_samples=num_samples,
                                      random_seed=42)


if __name__ == '__main__':
    device_name, device = check_cuda()
    flags = Flags()
    flags.LoadSavedModel = True
    flags.OneDataSet = True
    flags.test_batch_size = 4

    image_set = load_data_set(flags)
    net, total_params, total_trainable_params = make_model(flags)
    print("Defined model: " + flags.model_name + "  Number of parameters {}\{} \n"
          .format(total_trainable_params, total_params))


    start_time = time.time()

    loader = torch.utils.data.DataLoader(image_set, pin_memory=True,
                                              batch_size=flags.test_batch_size)  # , num_workers = 4)


    for i, (labels, inputs) in enumerate(loader):
        labels = labels.cuda()
        inputs = inputs.cuda()
        inputs = inputs.repeat(1, 3, 1, 1)
            # inputs = inputs.repeat(1, 3, 1, 1)

        plot_images_lime(inputs, labels, flags,columns=flags.test_batch_size/2, rows=flags.test_batch_size/2)
        break


    x=2