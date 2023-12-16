from tqdm import tqdm
import numpy as np
import time
import argparse
import os.path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import json
import torchvision
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
# from pytorch_apis import mm, sum_two_tensors, transpose
from gp_apis import gp_mm, gp_sum_two_tensors, gp_transpose
import math
import os
import random 


seed = 42
torch.cuda.manual_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda")
input_size = 784

class LinearFunction(torch.autograd.Function):
    """
    References:
    - PyTorch Extending PyTorch: https://pytorch.org/docs/stable/notes/extending.html
    - Custom C++ and CUDA Extensions: https://pytorch.org/tutorials/advanced/cpp_extension.html#integrating-a-c-cuda-operation-with-pytorch
    """
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        dim0, dim1 = input.size(0), weight.size(1)
        output = gp_mm(input, weight, dim0, dim1, device)
        if bias is not None:
            dim0, dim1 = output.size(0), output.size(1)
            output = gp_sum_two_tensors(output, bias.unsqueeze(0).expand_as(output), dim0, dim1, device)
        return output

    @staticmethod
    def backward(ctx, dZ):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            dim_0, dim_1 = input.shape
            weightT = gp_transpose(weight, weight.size(1), weight.size(0), device)
            grad_input = gp_mm(dZ, weightT, dim_0, dim_1, device)
            
        if ctx.needs_input_grad[1]:
            dim_0, dim_1 = weight.shape
            inputT = gp_transpose(input, input.size(1), input.size(0), device)
            grad_weight = gp_mm(inputT, dZ, dim_0, dim_1, device)
            
        if bias is not None and ctx.needs_input_grad[2]:
            dim0, dim1 = bias.size(0), dZ.size(0)
            ones = torch.ones(1, dim1).to(device)
            grad_bias = gp_mm(ones, dZ, 1, dim0, device).squeeze()

        return grad_input, grad_weight, grad_bias


class CustomLinear(nn.Module):
    """
    References:
    - PyTorch Extending PyTorch: https://pytorch.org/docs/stable/notes/extending.html
    - Custom C++ and CUDA Extensions: https://pytorch.org/tutorials/advanced/cpp_extension.html#integrating-a-c-cuda-operation-with-pytorch
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.input_features = in_features
        self.output_features = out_features

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

class LeNet300_100(nn.Module):
    def __init__(self, custom=True):
        super(LeNet300_100, self).__init__()
        
        if custom:
            linearLayer = CustomLinear
        else:
            linearLayer = nn.Linear
            
        self.fc1 = linearLayer(in_features = input_size, out_features = 300)
        self.fc2 = linearLayer(in_features = 300, out_features = 100)
        self.output = linearLayer(in_features = 100, out_features = 10)    
    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        return self.output(out)


def _load_data(DATA_PATH='./contents/', batch_size=512):
    print("data_path: ", DATA_PATH)
    train_trans = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])

    train_dataset = torchvision.datasets.MNIST(root=DATA_PATH,
                                              download=True,
                                              train=True,
                                              transform=train_trans)
    train_loader = DataLoader(dataset=train_dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=8)

    ## for testing
    test_trans = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])

    test_dataset = torchvision.datasets.MNIST(root=DATA_PATH,
                                             download=True,
                                             train=False,
                                             transform=test_trans)
    test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=8)

    return train_loader, test_loader


def adjust_learning_rate(lr, optimizer, epoch, decay, threshold):
    if (epoch+1) % threshold == 0:
        lr = lr * decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print("learning_rate: ", lr)
    return lr

def main(args):
    time_bf = time.time()
    num_epoches = args.num_epoches
    decay = args.decay
    threshold = args.threshold
    learning_rate = args.learning_rate
    custom = eval(args.custom)

    ## Load data
    DATA_PATH = "./data/"
    train_loader, test_loader=_load_data()

    model = LeNet300_100(custom)
    model.to(device)
    print("Begin training")
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    loss_fun = nn.CrossEntropyLoss()

    model = model.train()
    epoch_losses = []
    for epoch in range(num_epoches):
        learning_rate = adjust_learning_rate(learning_rate, optimizer, epoch, decay, threshold)
        print("EPOCH", epoch + 1)
        y_pred_all = np.array([])
        all_labels = np.array([])
        total_loss = 0.0

        for (x_batch, y_labels) in tqdm(train_loader):
            x_batch, y_labels = x_batch.reshape(-1, 28 * 28 * 1).to(device), y_labels.to(device)

            output_y = model(x_batch)
            loss = loss_fun(output_y, y_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, y_pred = torch.max(output_y.data, 1)
            y_pred_all = np.append(y_pred_all, y_pred.cpu().numpy())
            all_labels = np.append(all_labels, y_labels.cpu().numpy())

        # Compute average loss for the epoch
        avg_loss = total_loss / len(train_loader)

        # Append the average loss to the list of losses
        epoch_losses.append(avg_loss)

        # Compute accuracy on the training dataset
        train_accy = accuracy_score(y_pred_all, all_labels)
        train_f1 = f1_score(y_pred_all, all_labels, average='macro')
        train_precision = precision_score(y_pred_all, all_labels, average='macro')
        train_recall = recall_score(y_pred_all, all_labels, average='macro')


        print("Train accuracy for epoch {}: {:.4f}".format(epoch + 1, train_accy))
        print("Train F1 for epoch {}: {:.4f}".format(epoch + 1, train_f1))
        print("Average loss for epoch {}: {:.4f}".format(epoch + 1, avg_loss))

    y_pred_all = np.array([])
    y_test_all = np.array([])
    model.eval()
    with torch.no_grad():
      for (x_batch,y_labels) in tqdm((test_loader)):
        x_batch, y_labels = (x_batch.reshape(-1, 28*28)).to(device), (y_labels).to(device)
        output_y_test = model(x_batch)

        _, y_pred = torch.max(output_y_test.data, 1)
        y_pred_all=np.append(y_pred_all, y_pred.cpu().numpy())
        y_test_all=np.append(y_test_all, y_labels.cpu().numpy())

    test_accy = accuracy_score(y_pred_all, y_test_all)
    test_f1 = f1_score(y_pred_all, y_test_all, average='macro')
    test_precision = precision_score(y_test_all, y_test_all, average='macro')
    test_recall = recall_score(y_pred_all, y_test_all, average='macro')

    print()
    print("Test accuracy: {:.4f}".format(test_accy))
    print("Test F1: {:.4f}".format(test_f1))
    print("Time taken:", time.time()-time_bf)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LeNet-300-100 Training')

    # Define command-line arguments
    parser.add_argument('--num-epoches', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--decay', type=float, default=0.1,
                        help='Decay value')
    parser.add_argument('--threshold', type=int, default=15,
                        help='Threshold value')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--custom', type=str, default='False',
                        help='Use custom option')

    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
