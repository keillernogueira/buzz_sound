import sys
import datetime

from dataloader import BuzzDataLoader
from dataloader_audio import BuzzDataLoaderAudio
from config import *
from utils import *

from sklearn.metrics import accuracy_score, confusion_matrix

import torch
from torch import optim
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F


def audio_test(test_loader, net):
    # Setting network for evaluation mode.
    net.eval()

    non_background_preds = []
    non_background_seconds = []
    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):

            # Obtaining images, labels and paths for batch.
            inps, seconds = data
            inps = inps.squeeze()

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()

            # Forwarding.
            outs = net(inps_c)
            # Computing probabilities.
            soft_outs = F.softmax(outs, dim=1)

            # Obtaining prior predictions.
            prds = soft_outs.cpu().data.numpy().argmax(axis=1)
            non_background = np.where(np.logical_or(prds == 1, prds == 2))
            non_background_preds.append(non_background)
            non_background_seconds.append(seconds[non_background])
    print(non_background_preds)
    print(non_background_seconds)


def test(test_loader, net, epoch):
    # Setting network for evaluation mode.
    net.eval()

    all_labels = None
    all_preds = None
    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):

            # Obtaining images, labels and paths for batch.
            inps, labs = data

            inps = inps.squeeze()
            labs = labs.squeeze()

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            labs_c = Variable(labs).cuda()

            # Forwarding.
            outs = net(inps_c)
            # Computing probabilities.
            soft_outs = F.softmax(outs, dim=1)

            # Obtaining prior predictions.
            prds = soft_outs.cpu().data.numpy().argmax(axis=1)

            if all_labels is None:
                all_labels = labs
                all_preds = prds
            else:
                all_labels = np.concatenate((all_labels, labs))
                all_preds = np.concatenate((all_preds, prds))

        acc = accuracy_score(all_labels, all_preds)
        conf_m = confusion_matrix(all_labels, all_preds)

        _sum = 0.0
        for k in range(len(conf_m)):
            _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

        print("---- Validation/Test -- Epoch " + str(epoch) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Overall Accuracy= " + "{:.4f}".format(acc) +
              " Normalized Accuracy= " + "{:.4f}".format(_sum / float(outs.shape[1])) +
              " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
              )

        sys.stdout.flush()

    return acc, _sum / float(outs.shape[1]), conf_m


def train(train_loader, net, criterion, optimizer, epoch):
    # Setting network for training mode.
    net.train()

    # Average Meter for batch loss.
    train_loss = list()

    # Iterating over batches.
    for i, data in enumerate(train_loader):
        # Obtaining buzz sounds and labels
        buzzes, labels = data[0], data[1]
        # print(buzz_s.shape, label)

        # Casting tensors to cuda.
        buzzes_c, labels_c = buzzes.cuda(), labels.cuda()
        buzzes_c.squeeze_(0)
        labels_c.squeeze_(0)

        # Casting to cuda variables.
        inps = Variable(buzzes_c).cuda()
        labs = Variable(labels_c).cuda()

        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        outs = net(inps)
        soft_outs = F.softmax(outs, dim=1)

        # Obtaining predictions.
        prds = soft_outs.cpu().data.numpy().argmax(axis=1)
        # print(soft_outs.data, prds, type(prds))

        # Computing loss.
        loss = criterion(outs, labs)

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())

        # Printing.
        if (i + 1) % DISPLAY_STEP == 0:
            acc = accuracy_score(labels, prds)
            conf_m = confusion_matrix(labels, prds)

            _sum = 0.0
            for k in range(len(conf_m)):
                _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

            print("Training -- Epoch " + str(epoch) + " -- Iter " + str(i+1) + "/" + str(len(train_loader)) +
                  " -- Time " + str(datetime.datetime.now().time()) +
                  " -- Training Minibatch: Loss= " + "{:.6f}".format(train_loss[-1]) +
                  " Overall Accuracy= " + "{:.4f}".format(acc) +
                  " Normalized Accuracy= " + "{:.4f}".format(_sum / float(outs.shape[1])) +
                  " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
                  )

    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')

    # general options
    parser.add_argument('--operation', type=str, required=True, help='Operation. Options: [Train | Test]')
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save outcomes (such as images and trained models) of the algorithm.')

    # model options
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Path to a trained model to be used during the inference.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epoch_num', type=int, default=50, help='Number of epochs')
    args = parser.parse_args()
    print(args)

    # data loaders
    if args.operation == 'Train':
        train_dataset = BuzzDataLoader('Train', args.dataset_path)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=NUM_WORKERS, drop_last=False)

        test_dataset = BuzzDataLoader('Validation', args.dataset_path)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # network
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, train_dataset.num_classes)
        model.cuda()

        # loss
        criterion = nn.CrossEntropyLoss().cuda()

        optimizer = optim.Adam([
            {'params': list(model.parameters())[:-1]},
            {'params': list(model.parameters())[-1], 'lr': args.learning_rate, 'weight_decay': args.weight_decay}],  # 1e-4
            lr=args.learning_rate/10, weight_decay=args.weight_decay, betas=(0.9, 0.99)
        )

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        curr_epoch = 1
        best_records = []
        for epoch in range(curr_epoch, args.epoch_num + 1):
            train(train_dataloader, model, criterion, optimizer, epoch)
            if epoch % VAL_INTERVAL == 0:
                # Computing test.
                acc, nacc, cm = test(test_dataloader, model, epoch)

                save_best_models(model, optimizer, args.output_path, best_records, epoch, acc, nacc, cm)

            scheduler.step()
    elif args.operation == 'Test':
        assert args.model_path is not None, "For inference, flag --model_path should be set."

        # network
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, 3)
        model.load_state_dict(torch.load(args.model_path))
        model.cuda()

        dataset = BuzzDataLoaderAudio('Train', args.dataset_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
        audio_test(dataloader, model)
    else:
        raise NotImplementedError("Operation " + args.operation + " not implemented")
