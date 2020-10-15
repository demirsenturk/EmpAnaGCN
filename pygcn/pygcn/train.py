from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_cora, accuracy, load_SBM, load_real_world_dataset
from pygcn.models import GCN

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return acc_train, loss_train.item(), loss_val.item()


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test, loss_test.item()

run = 0
number_of_runs = 15
while run < number_of_runs:
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Different data loaders
    "For Cora dataset"
    # adj, features, labels, idx_train, idx_val, idx_test = load_cora()
    "SBM Graph Loader"
    adj, features, true_labeling, idx_train, idx_val, idx_test, labels = load_SBM()
    "Real-world Dataset Loader"
    # adj, features, labels, idx_train, idx_val, idx_test = load_real_world_dataset()

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid= 64, #args.hidden,
                nclass=labels.max().item() + 1,
                dropout= args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr= args.lr,
                           weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Train model
    t_total = time.time()
    # for epoch in range(args.epochs):
    #     train(epoch)

    results = []
    train_losses = []
    validation_losses = []
    acc, loss = test()
    results.append(acc)

    for epoch in range(200):
        acc_train, train_loss, val_loss = train(epoch)
        acc, loss = test()
        results.append(acc)
        train_losses.append(train_loss)
        validation_losses.append(val_loss)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    str_i = 0
    if run == 0:
        str_i = "1"
    elif run == 1:
        str_i = "2"
    elif run == 2:
        str_i = "3"
    elif run == 3:
        str_i = "4"
    elif run == 4:
        str_i = "5"
    elif run == 5:
        str_i = "6"
    elif run == 6:
        str_i = "7"
    elif run == 7:
        str_i = "8"
    elif run == 8:
        str_i = "9"
    elif run == 9:
        str_i = "10"
    elif run == 10:
        str_i = "11"

    np.save('GCN' + str_i, results)
    np.save('GCN' + str_i + '_val_loss', validation_losses)
    np.save('GCN' + str_i + '_train_loss', train_losses)
    # Testing
    test()
    print("Finished Run: " + str(run))
    run = run + 1
