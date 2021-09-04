from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGGBase(nn.Module):
    """
    VGG base is used to produce lower level feature maps
    """

    def __init__(self):
        super(VGGBase, self).__init__()

        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)

        # Replacing fully connected layers of FC6 and FC7 with atrous conv layers
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=(1, 1))

        # Load pretrained layers
        self.load_pretrained_layers()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """
        out = F.relu(self.conv1_1(image))  # (N, 64, 300, 300)
        out = F.relu(self.conv1_2(out))  # (N, 64, 300, 300)
        out = self.pool1(out)  # (N, 64, 150, 150)

        out = F.relu(self.conv2_1(out))  # (N, 128, 150, 150)
        out = F.relu(self.conv2_2(out))  # (N, 128, 150, 150)
        out = self.pool2(out)  # (N, 128, 75, 75)

        out = F.relu(self.conv3_1(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_2(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_3(out))  # (N, 256, 75, 75)
        out = self.pool3(out)  # (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

        out = F.relu(self.conv4_1(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_2(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_3(out))  # (N, 512, 38, 38)
        conv4_3_feats = out  # (N, 512, 38, 38)
        out = self.pool4(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv5_1(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_2(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_3(out))  # (N, 512, 19, 19)
        out = self.pool5(out)  # (N, 512, 19, 19), pool5 does not reduce dimensions

        out = F.relu(self.conv6(out))  # (N, 1024, 19, 19)

        conv7_feats = F.relu(self.conv7(out))  # (N, 1024, 19, 19)

        # Lower-level feature maps
        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):
        """
        Loading VGG-16 weights trained on ImageNet.
        Pytorch library has set of pretrained weights for VGG-16. It can be find in below link
        https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16

        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation.
        See 'decimate' in utils.py.
        """
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # VGG base with pretrained weights
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Copy pretrained weights to our current VGG model base
        for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        # Auxiliary/additional convolutions on top of the VGG base
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=(1, 1), padding=0)  # stride = 1, by default
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2),
                                 padding=1)  # dim. reduction because stride > 1

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=(1, 1), padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2),
                                 padding=1)  # dim. reduction because stride > 1

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        """

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2
        """
        out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 19, 19)
        out = F.relu(self.conv8_2(out))  # (N, 512, 10, 10)
        conv8_2_feats = out  # (N, 512, 10, 10)

        out = F.relu(self.conv9_1(out))  # (N, 128, 10, 10)
        out = F.relu(self.conv9_2(out))  # (N, 256, 5, 5)
        conv9_2_feats = out  # (N, 256, 5, 5)

        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats


class PredictionConvolutions(nn.Module):
    """
    Convolution to predict bounding boxes and classes with features maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 1930 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores are the representation of confidence in prediction for each object class
    for all 1930 bounding boxes,.
    In this application we have two classes only. One is an object and other is background(no object).
    """

    def __init__(self, num_class):
        """
        :param num_class: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.num_class = num_class

        # Here n_boxes represent number of boxes to predict for each feature map cell
        # As per the instruction we have used 1 box for each feature map cell
        n_boxes = {'conv4_3': 1, 'conv7': 1, 'conv8_2': 1, 'conv9_2': 1}

        # Convolution for bounding box predictions
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=(3, 3), padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=(3, 3), padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=(3, 3), padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=(3, 3), padding=1)

        # convolutions for class score predictions
        self.cls_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * num_class, kernel_size=(3, 3), padding=1)
        self.cls_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * num_class, kernel_size=(3, 3), padding=1)
        self.cls_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * num_class, kernel_size=(3, 3), padding=1)
        self.cls_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * num_class, kernel_size=(3, 3), padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats):
        """
        Forward propagation of prediction convolutions which will predict classes and bounding boxes
        for each feature map cell.

        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 38, 38)
        :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 10, 10)
        :param conv9_2_feats: conv9_2 feature map, a tensor of dimensions (N, 256, 5, 5)
        :return: 1930 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv4_3_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 16)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 1444, 4), total 1444 boxes on this feature map

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 361, 4), total 361 boxes

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 100, 4) total 100 boxes

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 25, 4) total 25 boxes

        ########################################################################
        # Predicting the classes with localized boxes
        c_conv4_3 = self.cls_conv4_3(conv4_3_feats)  # (N, 1 * num_class, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 1 * num_class)
        # total 1444 boxes for this feature map
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.num_class)  # (N, 1444, num_class)

        c_conv7 = self.cls_conv7(conv7_feats)  # (N, 1 * num_class, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 1 * num_class)
        # total 361 boxes for this feature map
        c_conv7 = c_conv7.view(batch_size, -1, self.num_class)  # (N, 361, num_class)

        c_conv8_2 = self.cls_conv8_2(conv8_2_feats)  # (N, 1 * num_class, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 1 * num_class)
        # total of 100 boxes for this feature map
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.num_class)  # (N, 100, num_class)

        c_conv9_2 = self.cls_conv9_2(conv9_2_feats)  # (N, 1 * num_class, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 1 * num_class)
        # total of 25 boxes for this feature map
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.num_class)  # (N, 25, num_class)

        # A total of 1930 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locations = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2], dim=1)  # (N, 1930, 4)
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2], dim=1)  # (N, 1930, num_class)

        return locations, classes_scores


class SSD(nn.Module):
    """
    The SSD network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    """

    def __init__(self, num_class):
        super(SSD, self).__init__()

        self.num_class = num_class
        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(num_class)

        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 1930 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Rescale conv4_3 after L2 norm
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)

        # Run auxiliary convolutions (higher level feature map generators)
        conv8_2_feats, conv9_2_feats = self.aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        # (N, 1930, 4), (N, 1930, 2)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        In SSD technique, the better the prior boxes the better the detection.
        We map the ground truth boxes to prior boxes based on a scoring method like IOU.
        Best

        Create the 1930 prior (default) boxes. One box for each feature map cell for corresponding
        feature map dimensions.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (1930, 4)
        """
        # value of k for each feature map to create k^2 boxes for each feature map
        feature_map_dims = {'conv4_3': 38, 'conv7': 19, 'conv8_2': 10, 'conv9_2': 5}

        # scale for boxes across different feature maps. boxes for inner feature maps
        # are scaled much lower to detect small objects
        obj_scales = {'conv4_3': 0.1, 'conv7': 0.21, 'conv8_2': 0.255, 'conv9_2': 0.30}

        # Defined aspect ratio calculated from mean of (w/h) across all bounding boxes
        # from the dataset. The mean is 0.66 with deviation of 0.07. So aspect ratio is kept
        # at 0.66 for all feature maps
        aspect_ratios = {'conv4_3': [0.5], 'conv7': [0.55], 'conv8_2': [0.6], 'conv9_2': [.66]}

        fmaps = list(feature_map_dims.keys())
        prior_boxes = []
        for k, fmap in enumerate(fmaps):
            # for each feature map, create k*k boxes
            for i in range(feature_map_dims[fmap]):
                for j in range(feature_map_dims[fmap]):
                    # calculate center coordinates of boxes
                    cx = (j + 0.5) / feature_map_dims[fmap]
                    cy = (i + 0.5) / feature_map_dims[fmap]

                    # For each
                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (1930, 4)
        prior_boxes.clamp_(0, 1)  # (1930, 4)

        return prior_boxes

    def detect_objects(self, pred_locs, pred_scores, min_score, max_overlap, top_k):
        """
        Decode 1930 bounding box locations and class scores to detect objects.

        For each class (in this case single class), perform Non-Maximum Suppression (NMS) on boxes
        that are above a minimum threshold.

        :param pred_locs: predicted boxes w.r.t 1930 prior boxes, a tensor of dimensions (N, 1930, 4)
        :param pred_scores: class scores for each box, a tensor of dimensions (N, 1930, 2)
        :param min_score: minimum threshold for a box to be considered a match for the object class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = pred_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        # Calculate softmax over class labels. (object vs background)
        pred_scores = F.softmax(pred_scores, dim=2)  # (N, 1930, 2)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == pred_locs.size(1) == pred_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            # these are fractional point coordinates
            decoded_locations = cxcy_to_xy(gcxgcy_to_cxcy(pred_locs[i], self.priors_cxcy))  # (1930, 4),

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            # Keep object class scores where object is present, we need not calculate the score
            # for background. This score is thresholded by minimum score

            object_class_scores = pred_scores[i][:, 1]  # (1930)
            score_above_min_score = object_class_scores > min_score  # torch.bool
            # number of box_prediction with good confidence score for object class
            n_above_min_score = score_above_min_score.sum().item()

            # Pull the scores of object class above minimum scores
            object_class_scores = object_class_scores[score_above_min_score]  # (n_qualified)
            # Pull decoded locations for the object class with qualified score
            class_decoded_locs = decoded_locations[score_above_min_score]  # (n_qualified, 4)

            # Sorting the predicted boxes and scores by object class scores
            # (n_qualified), (n_min_score)
            object_class_scores, sort_ind = object_class_scores.sort(dim=0, descending=True)
            class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

            # Find the overlap among predicted boxes
            overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

            # Non-Maximum Suppression (NMS)
            # This technique is used to eliminate boxes which represent same object in the image
            # Here we use suppress tensor to keep track of boxes which we need to supress
            # Value of 1 means suppress and value of 0 means don't suppress
            suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

            # Consider each box in order of decreasing scores
            for box in range(class_decoded_locs.size(0)):
                # If the box is already suppressed then continue
                if suppress[box] == 1:
                    continue

                # Otherwise suppress boxes whose overlaps with this box, are greater than maximum overlap
                # Find these boxes and update the suppress index
                boxes_ovrlp_gthn_max_ovrlp = overlap[box] > max_overlap
                boxes_ovrlp_gthn_max_ovrlp = torch.tensor(boxes_ovrlp_gthn_max_ovrlp, dtype=torch.uint8)
                suppress = torch.max(suppress, boxes_ovrlp_gthn_max_ovrlp)

                # Don't suppress this box, even though it has an overlap of 1 with itself
                suppress[box] = 0

            # Store only unsuppressed boxes for this class
            image_boxes.append(class_decoded_locs[1 - suppress])
            image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [1]).to(device))
            image_scores.append(object_class_scores[1 - suppress])

            # If the object is not present in the image, it is the 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenating all the objects in an image, into single tensor
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Save only top_k confident objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.
    It Comprises of two different losses computed together

    1) Localization loss for predicted bounding boxes
    2) Confidence loss for predicted class scores
    """

    def __init__(self, priors_cxcy, threshold=0.2, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predicted_locs, predicted_scores, boxes):
        """
        Forward propagation.
        :param predicted_locs: predicted locations/boxes w.r.t the 1930 prior boxes, a tensor of dimensions (N, 1930, 4)
        :param predicted_scores: class scores for each of the encoded boxes, a tensor of dimensions (N, 1930, 2)
        :return: a multibox loss
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        num_class = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 1930, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 1930)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            # Calculate overlap of prior boxes with true ground truth boxes,
            # For each ground truth box we calculate overlap with each prior box
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # (num_object_boxes, 1930)

            # For each prior box, find the object box that has the maximum overlap.
            # With this, each prior box will be mapped to best matching object box
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (1930)

            # Now there can be a situation where some object box is not represented by any or our positive priors
            # The reason can be that an object might not be the best overlap among the priors, so it is not
            # selected in object_for_each_prior

            # To correct this, for each object, we find the prior which has maximum overlap
            # We assign that prior box to that object box, with this each object in the image has
            # an associated prior box.

            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # We then assign the objects to their corresponding priors. Now every object is mapped
            # to atleast one prior
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # We now have to ensure that these prior boxes qualify the threshold criterion for
            # background. To ensure this we artificially five them overlap score of 1.
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Since we are working with two classes only. We need to single value (object class)
            # to identify the box classes. We assign class 1 for each prior
            label_for_each_prior = torch.ones(n_priors, dtype=torch.long)  # (1930)

            # The priors with less than self.threshold are assigned background class (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (1930)

            # Assign values to each of true_classes
            true_classes[i] = label_for_each_prior

            # Since prior boxes are created as center sized, we encode them to the form in which predicted boxes
            # are regressed
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (1930, 4)

        # Identifying positive  priors
        positive_priors = true_classes != 0  # (N, 1930)

        # CALCULATING LOCALIZATION LOSS
        # Localization loss is computed over positive priors which contains objects
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])

        # CONFIDENCE LOSS
        # Confidence loss is calculated over positive priors (objects) and most difficult negative priors (backgrounds)

        # For each image we perform Hard Negative Mining where we take the hardest (neg_pos_ratio * n_positives)
        # negative priors, which constitutes maximum loss

        # Compute number of positive priors for each image
        n_positives = positive_priors.sum(dim=1)  # (N)

        # HARD NEGATIVE MINING
        # Calculate number of hard negatives based on the ratio
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # Compute the loss for all the priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, num_class), true_classes.view(-1))  # (N * 1930)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 1930)

        # Filter out the loss for positive priors
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Now we need to find Hard Negative priors, for this we sort only negative priors in decreasing order
        # of loss value.
        # We then extract top-n hard negatives.
        conf_loss_neg = conf_loss_all.clone()  # (N, 1930)

        # To ignore positive priors here, we can set loss for them to zero
        conf_loss_neg[positive_priors] = 0.  # (N, 1930)

        # Sort negative priors by decreasing hardness
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 1930)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 1930)

        # Filter out top-N hard negatives
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 1930)
        # Compute confidence loss for hard negatives
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]

        # As mentioned in the paper , the loss is averaged over positive priors only but computed over both
        # hard negative and positive priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss
