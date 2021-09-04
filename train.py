from utils import preprocess_data
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD, MultiBoxLoss
from datasets import GroceryDataset
from utils import *
import argparse, traceback

parser = argparse.ArgumentParser()

parser.add_argument('-data', required=True)
parser.add_argument('-object_json', default='./All_objects.json')
parser.add_argument('-checkpoint', default=None, help='model checkpoint path')
parser.add_argument('-batch_size', default=32, type=int, help='batch_size')
parser.add_argument('-iterations', default=2000, type=int, help='number of iterations to train')
parser.add_argument('-num_workers', default=3, type=int, help='number of worker to fetch data from dataset')
parser.add_argument('-print_freq', default=2, type=int, help='printing frequency')
parser.add_argument('-lr', default=0.001, type=float, help='learning rate' )
parser.add_argument('-momentum', default=0.9, type=float, help='momentum')
parser.add_argument('-grad_clip', default=None, type=int, help='if gradients explode, we can use it to clip gradients')
parser.add_argument('-use_gpu', default=True, type=bool, help='Enable gpu use')
parser.add_argument('-model_save', default='checkpoint.pth.tar')



global start_epoch, label_map, epoch, decay_lr_at, device
num_class = len(label_map)  # number of different types of objects
cudnn.benchmark = True


def main(args):

    # cuda settings
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    # adding seed for reproduceability
    torch.manual_seed(5)

    #learning rate parameters
    decay_lr_at = [1000, 2000]  # After these iteration decay the learning rate
    decay_lr_to = 0.1  # decay lr by this fraction

    # Initialize model or load checkpoint
    if args.checkpoint is None:
        start_epoch = 0
        model = SSD(num_class=num_class)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * args.lr}, {'params': not_biases}],
                                    lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = GroceryDataset(args.object_json, split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=args.num_workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    epochs = args.iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    # Epochs
    print('Starting Training')
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(args, train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, args.model_save)


def train(args, model, train_loader, criterion, optimizer, epoch):
    """
    Here we initiate the training for one epoch on the dataset.

    :param args: arguments for the training
    :param model: model
    :param train_loader: DataLoader for training data
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    # Moving the model to training mode
    model.train()

    avg_batch_losses = AverageMeter()  # Average loss calculator
    # Batches
    for i, (images, boxes, labels) in enumerate(train_loader):

        # Move to default device
        images = images.to(device)  # (N, 3, 300, 300)
        boxes = [b.to(device) for b in boxes] # A list of size == batch size

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 1930, 4), (N, 1930, num_class)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes)  # scalar

        # Zero out the gradient buffer and initiate gradient computation
        optimizer.zero_grad()
        loss.backward()

        # Clip the gradients if grad_clip is provided
        if args.grad_clip is not None:
            clip_gradient(optimizer, args.grad_clip)

        # Apply Gradients
        optimizer.step()
        avg_batch_losses.update(loss.item(), images.size(0))

        # Print status
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'. \
                  format(epoch, i, len(train_loader), loss=avg_batch_losses), flush=True)
    # freeing some memory here
    del predicted_locs, predicted_scores, images, boxes


if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.isfile(args.object_json):
        print('Data Json file does not exist. Creating one.')
        try:
            preprocess_data(dset_path=args.data, output_folder='./')
        except:
            traceback.print_exc()
            exit('An exception occured while creating Data File')
    main(args)
