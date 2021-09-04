import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from utils import transform


class GroceryDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_file, split):
        """
        :param data_file: folder where data files are stored
        :param split: split, one of 'train' or 'test'
        """
        self.split = split.lower()
        assert self.split in {'train', 'test'}
        # Read data files
        with open(data_file, 'r') as j:
            self.all_data = json.load(j)
        self.data = self.all_data[self.split]

    def convert_boxes(self, boxes):
        new_boxes = []
        for b in boxes:
            b[2] = b[0] + b[2]
            b[3] = b[1] + b[3]
            new_boxes.append(b)
        return new_boxes

    def __getitem__(self, i):
        # Read image
        item = self.data[i]
        image = Image.open(item['path'], mode='r').convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        boxes = self.convert_boxes(item['boxes'])
        boxes = torch.FloatTensor(boxes)  # (n_objects, 4) (boxes in format (xmin, ymin, xmax, ymax)
        labels = torch.LongTensor(item['label'])  # (n_objects)

        # Apply transformations
        image, boxes, labels = transform(image, boxes, labels, split=self.split)
        return image, boxes, labels

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each
