from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import networkx as nx

import scipy.sparse as sp
from scipy.spatial.distance import cdist

import sys
import argparse
import subprocess

try:
    import networkx as nx

except ImportError:
    import platform
    if platform.system() == 'Windows':
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'networkx'])
        
    else:
        subprocess.check_call([sys.executable, "-m", "pip3", "install", 'networkx'])

finally:
    import networkx as nx

class BorisNet(nn.Module):
    def __init__(self):
        super(BorisNet, self).__init__()
        self.fc = nn.Linear(784, 10, bias=False)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class BorisConvNet(nn.Module):
    def __init__(self):
        super(BorisConvNet, self).__init__()
        self.conv = nn.Conv2d(1, 10, 28, stride=1, padding=14)
        self.fc = nn.Linear(4 * 4 * 10, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.max_pool2d(x, 7)
        return self.fc(x.view(x.size(0), -1))

class BorisGraphNet(nn.Module):
    def __init__(self, img_size=28, pred_edge=False, original_guassian_adj_mtx = False):
        super(BorisGraphNet, self).__init__()
        self.pred_edge = pred_edge
        N = img_size ** 2
        self.fc = nn.Linear(N, 10, bias=False)

        if pred_edge:
            col, row = np.meshgrid(np.arange(img_size), np.arange(img_size))
            coord = np.stack((col, row), axis=2).reshape(-1, 2)
            coord = (coord - np.mean(coord, axis=0)) / (np.std(coord, axis=0) + 1e-5)
            coord = torch.from_numpy(coord).float()  # 784,2
            coord = torch.cat((coord.unsqueeze(0).repeat(N, 1,  1),
                                    coord.unsqueeze(1).repeat(1, N, 1)), dim=2)
            #coord = torch.abs(coord[:, :, [0, 1]] - coord[:, :, [2, 3]])
            self.pred_edge_fc = nn.Sequential(nn.Linear(4, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 1),
                                              nn.Tanh())
            self.register_buffer('coord', coord)

        elif original_guassian_adj_mtx:
            # Gaussian Adj Matrix (Author's implementation)
            A = self.guassian_precompute_adjacency_images(img_size)
            self.register_buffer('A', A)
        
        else:
            # Precompute My Own Custom Adj Matrix
            A = self.precompute_adjacency_images(img_size)
            self.register_buffer('A', A)

    # ---------- Adjacency Matrix construction ----------

    # HMLAA Seminar Exercise Task -> My implementation
    @staticmethod
    def precompute_adjacency_images(img_size):
        # Input graph of N nodes and unit length edges between them
        graph = nx.grid_2d_graph(img_size, img_size)

        # Unit length adj matrix
        adj_mtx = nx.adjacency_matrix(graph)

        # Input adjacency matrix is added to a sparse matrix which has ones on diagonal
        adj_mtx = adj_mtx + sp.eye(adj_mtx.shape[0])

        D1 = np.array(adj_mtx.sum(axis=0))**(0.01)  # along columns
        D2 = np.array(adj_mtx.sum(axis=1))**(0.01)  # along rows

        # Create sparse matrices from corresponding diagonals of row vectors (transform into matrix)
        D1 = sp.diags(D1[0,:])
        D2 = sp.diags(D2[:,0])

        # Dot Product with both the Diagonal Sparse Matrices
        adj_mtx = adj_mtx.dot(D1)
        adj_mtx = D2.dot(adj_mtx)

        # Converting sparse matrix to numpy 2d array and then to tensor
        adj_mtx = sp.csr_matrix.toarray(adj_mtx)
        adj_mtx = torch.from_numpy(adj_mtx).float()

        #print(adj_mtx[:10, :10])

        # Return extremely sparse input adjacency matrix
        return adj_mtx


    # Original Guassian Adjacency Matrix -> Implemented by the author
    @staticmethod
    def guassian_precompute_adjacency_images(img_size):
        col, row = np.meshgrid(np.arange(img_size), np.arange(img_size))
        coord = np.stack((col, row), axis=2).reshape(-1, 2) / img_size
        dist = cdist(coord, coord)  
        sigma = 0.05 * np.pi
        
        # Below, I forgot to square dist to make it a Gaussian (not sure how important it can be for final results)
        A = np.exp(- dist / sigma ** 2)
        print('WARNING: try squaring the dist to make it a Gaussian')
            
        A[A < 0.01] = 0
        A = torch.from_numpy(A).float()

        # Normalization as per (Kipf & Welling, ICLR 2017)
        D = A.sum(1)  # nodes degree (N,)
        D_hat = (D + 1e-5) ** (-0.5)
        A_hat = D_hat.view(-1, 1) * A * D_hat.view(1, -1)  # N,N

        # Some additional trick I found to be useful
        A_hat[A_hat > 0.0001] = A_hat[A_hat > 0.0001] - 0.2

        #print(A_hat[:10, :10])
        return A_hat

# ---------------------------------------------------

    def forward(self, x):
        B = x.size(0)
        if self.pred_edge:
            self.A = self.pred_edge_fc(self.coord).squeeze()

        avg_neighbor_features = (torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1),
                                 x.view(B, -1, 1)).view(B, -1))
        return self.fc(avg_neighbor_features)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    print("\nAvailable Models:\n 1. FC\n 2. Graph (with my Custom Adj Matrix - default)\n 3. Convolution\n 4. Graph (with Gaussian Adj Matrix)\n")

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='graph', choices=['fc', 'graph', 'conv', 'gaussian_graph'],
                        help='model to use for training (default: graph)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--pred_edge', action='store_true', default=False,
                        help='predict edges instead of using predefined ones')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    #use_cuda = True

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning the model on: {device}\n")

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, 
                        transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                        ])), batch_size=args.test_batch_size, shuffle=False, **kwargs)


    if args.model == 'fc':
        assert not args.pred_edge, "this flag is meant for graphs"
        model = BorisNet()
        print("\n---------- Running FC Model ----------\n")
        print("\n*************************************\n")

    elif args.model == 'gaussian_graph':
        model = BorisGraphNet(pred_edge=args.pred_edge, original_guassian_adj_mtx = True)
        print(f"\n---------- Running Graph Model with Gaussian Mtx (Original implementation)\n & Predict Edge Flag set to {args.pred_edge} ----------\n")
        print("\n*************************************\n")

    elif args.model == 'conv':
        model = BorisConvNet()
        print("\n---------- Running Conv Model ----------\n")
        print("\n*************************************\n")

    elif args.model == 'graph':
        model = BorisGraphNet(pred_edge=False, original_guassian_adj_mtx = False)
        print(f"\n---------- Running Graph Model with Custom Adjacency Mtx (My implementation)\n & Predict Edge Flag set to {args.pred_edge} ----------\n")
        print("\n*************************************\n")

    else:
        raise NotImplementedError(args.model)

    model.to(device)
    print(f"\nModel Configuration: \n {model}")
    print("\n*************************************\n")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-1 if args.model == 'conv' else 1e-4)

    print('\nNumber of trainable parameters: %d\n' %
          np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()]))

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)



# Driver program
if __name__ == '__main__':
    main()
    # Examples:
    # python fc_vs_graph_train.py --model fc
    # python fc_vs_graph_train.py --model conv
    # python fc_vs_graph_train.py --model gaussian_graph -> Originial implementation
    # python fc_vs_graph_train.py --model graph --pred_edge
    # python fc_vs_graph_train.py --model graph -> My custom adjacency matrix
