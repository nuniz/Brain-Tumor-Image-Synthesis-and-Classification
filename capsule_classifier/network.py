from utils import *

USE_CUDA = torch.cuda.is_available()


class conv_cap_layer(nn.Module):
    """
    Capsule convolutional building block

    """
    def __init__(self, in_capsules, out_capsules, in_channels, out_channels, stride=1, padding=2,
                 kernel=5, num_routes=3, nonlinearity='sqaush', batch_norm=False, cuda=USE_CUDA):
        super(conv_cap_layer, self).__init__()
        self.num_routes = num_routes
        self.in_channels = in_channels
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.out_channels = out_channels
        self.nonlinearity = nonlinearity
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(in_capsules*out_capsules*out_channels)
        self.conv2d = nn.Conv2d(kernel_size=(kernel, kernel), stride=stride, padding=padding,
                                in_channels=in_channels, out_channels=out_channels*out_capsules)
        self.cuda = cuda

    def forward(self, x):
        batch_size = x.size(0)
        in_width, in_height = x.size(3), x.size(4)
        x = x.view(batch_size*self.in_capsules, self.in_channels, in_width, in_height)
        u_hat = self.conv2d(x)

        out_width, out_height = u_hat.size(2), u_hat.size(3)

        # batch norm layer
        if self.batch_norm:
            u_hat = u_hat.view(batch_size, self.in_capsules, self.out_capsules * self.out_channels, out_width, out_height)
            u_hat = u_hat.view(batch_size, self.in_capsules * self.out_capsules * self.out_channels, out_width, out_height)
            u_hat = self.bn(u_hat)
            u_hat = u_hat.view(batch_size, self.in_capsules, self.out_capsules*self.out_channels, out_width, out_height)
            u_hat = u_hat.permute(0,1,3,4,2).contiguous()
            u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, self.out_capsules, self.out_channels)

        else:
            u_hat = u_hat.permute(0,2,3,1).contiguous()
            u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, self.out_capsules*self.out_channels)
            u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, self.out_capsules, self.out_channels)


        b_ij = Variable(torch.zeros(1, self.in_capsules, out_width, out_height, self.out_capsules))
        if self.cuda:
            b_ij = b_ij.cuda()
        for iteration in range(self.num_routes):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(5)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)


            if (self.nonlinearity == 'relu') and (iteration == self.num_routes - 1):
                v_j = F.relu(s_j)
            elif (self.nonlinearity == 'leakyRelu') and (iteration == self.num_routes - 1):
                v_j = F.leaky_relu(s_j)
            else:
                v_j = self.squash(s_j)

            v_j = v_j.squeeze(1)

            if iteration < self.num_routes - 1:
                temp = u_hat.permute(0, 2, 3, 4, 1, 5)
                temp2 = v_j.unsqueeze(5)
                a_ij = torch.matmul(temp, temp2).squeeze(5) # dot product here
                a_ij = a_ij.permute(0, 4, 1, 2, 3)
                b_ij = b_ij + a_ij.mean(dim=0)

        v_j = v_j.permute(0, 3, 4, 1, 2).contiguous()

        return v_j

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class capsule_classifier(nn.Module):
    """
    capsule classifier

    """
    def __init__(self, flags):
        super(capsule_classifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2, stride=1)
        self.leakurelu = nn.LeakyReLU()
        self.convcaps1 = conv_cap_layer(in_capsules=1, out_capsules=1, in_channels=1, out_channels=64,
                                        stride=2, padding=1, kernel=4,
                                        nonlinearity="squash", batch_norm=flags.batch_norm, cuda=USE_CUDA)
        self.convcaps2 = conv_cap_layer(in_capsules=1, out_capsules=1, in_channels=64, out_channels=128,
                                        stride=2, padding=1, kernel=4,
                                        nonlinearity="squash", batch_norm=flags.batch_norm, cuda=USE_CUDA)
        self.convcaps3 = conv_cap_layer(in_capsules=1, out_capsules=1, in_channels=128, out_channels=256,
                                        stride=2, padding=1, kernel=4,
                                        nonlinearity="squash", batch_norm=flags.batch_norm, cuda=USE_CUDA)
        self.convcaps4 = conv_cap_layer(in_capsules=1, out_capsules=1, in_channels=256, out_channels=512,
                                        stride=2, padding=1, kernel=4,
                                        nonlinearity="squash", batch_norm=flags.batch_norm, cuda=USE_CUDA)
        # self.convcaps5 = convolutionalCapsule(in_capsules=1, out_capsules=1, in_channels=512, out_channels=1024,
        #                                       stride=2, padding=1, kernel=4,
        #                                       nonlinearity="squash", batch_norm=flags.batch_norm, cuda=USE_CUDA)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=0, stride=1)



        self.fc = nn.Sequential(nn.Linear(841, 841), #169
                               nn.LeakyReLU(negative_slope=flags.leaky_relu_slope),
                               nn.Dropout(p=flags.dropout),
                               nn.Linear(841, 420),
                               nn.LeakyReLU(negative_slope=flags.leaky_relu_slope),
                               nn.Dropout(p=flags.dropout),
                               nn.Linear(420, flags.class_number),
                               nn.LogSoftmax(dim=1))


    def forward(self, x):
        batch_size = x.size(0)

        x = self.leakurelu(self.conv1(x))
        x = x.view(batch_size, 1, x.size(1), x.size(2), x.size(3))

        x = self.convcaps1(x)
        x = self.convcaps2(x)
        x = self.convcaps3(x)
        x = self.convcaps4(x)
        # x = self.convcaps5(x)

        x = x.view(batch_size, x.size(2), x.size(3), x.size(4))
        x = self.leakurelu(self.conv2(x))
        x = x.view(batch_size, x.size(2) * x.size(3))
        x = self.fc(x)

        return x #x.squeeze

def make_model(flags):
    """
    MAkes the model with flags defenition

    Arguments
    ---------
    flags

    Outputs
    ---------
    model
    number of trainable params
    number of total params

    """
    if flags.model_name == "resnet":
        net = models.resnet50(pretrained=(flags.LoadImageNet))
        for param in net.parameters():
            param.requires_grad = True
        # get input of fc layer
        n_inputs = net.fc.in_features

        # redefine fc layer / top layer/ head for our classification problem
        net.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                        nn.LeakyReLU(negative_slope=flags.leaky_relu_slope),
                                        nn.Dropout(p=flags.dropout),
                                        nn.Linear(2048, 2048),
                                        nn.LeakyReLU(negative_slope=flags.leaky_relu_slope),
                                        nn.Dropout(p=flags.dropout),
                                        nn.Linear(2048, flags.class_number),
                                        nn.LogSoftmax(dim=1))
        for name, child in net.named_children():
            for name2, params in child.named_parameters():
                params.requires_grad = True

        if flags.FreezeLayers:
            ct = 0
            for child in net.children():
                ct += 1
                if ct < flags.num_of_freeze_layers:
                    for param in child.parameters():
                        param.requires_grad = False

    elif flags.model_name == "capsule":
        net = capsule_classifier(flags)
        for param in net.parameters():
            param.requires_grad = True
        for name, child in net.named_children():
            for name2, params in child.named_parameters():
                params.requires_grad = True
    else:
        print("Error: Don't have model.\n")
        sys.exit(1)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print the trasnfer learning NN model's architecture
    # net
    if flags.LoadSavedModel:
        net.load_state_dict(torch.load(os.path.join(flags.model_dir, flags.check_name)))
        print("Load model from: " + os.path.join(flags.model_dir, flags.check_name))

    return net, pytorch_total_params, pytorch_total_trainable_params

