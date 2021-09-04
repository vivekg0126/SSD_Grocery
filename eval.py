from utils import *
from datasets import GroceryDataset
import torch.utils.data as Data
from tqdm import tqdm
import argparse, json
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('-data', default='/Users/vxgupta/gr_ds/')
parser.add_argument('-object_json', default='./All_objects.json')
parser.add_argument('-checkpoint', default='./checkpoint.pth.tar', help='model checkpoint path to load checkpoint')
parser.add_argument('-batch_size', default=5, type=int, help='batch_size')
parser.add_argument('-num_workers', default=3, type=int, help='number of worker to fetch data from dataset')
parser.add_argument('-use_gpu', default=True, type=bool, help='Enable gpu use')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(args):

    # Load model checkpoint that is to be evaluated
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = checkpoint['model']
    model = model.to(device)

    # Load test data
    test_dataset = GroceryDataset(args.object_json, split='test')
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              collate_fn=test_dataset.collate_fn,
                                              num_workers=args.num_workers, pin_memory=True)
    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    all_det_boxes, all_det_labels, all_det_scores = [], [], []
    all_true_boxes, all_true_labels = [], []

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # forward pass to predict the location and scores of images
            pred_locs, pred_scores = model(images)

            # Detecting the object in the image
            det_boxes, det_labels, det_scores = model.detect_objects(pred_locs, pred_scores, min_score=0.6,
                                                                     max_overlap=0.50, top_k=200)

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            all_det_boxes.extend(det_boxes)
            all_det_labels.extend(det_labels)
            all_det_scores.extend(det_scores)
            all_true_boxes.extend(boxes)
            all_true_labels.extend(labels)

        # Calculate mAP
        APs, recall = calculate_mAP(all_det_boxes, all_det_labels, all_det_scores, all_true_boxes, all_true_labels)

    # Since we are calculating average precision of only one class ( object class)
    # mean average precision = average precision
    print('\nMean Average Precision (mAP): %.3f' % APs)
    print('\nMean Average Recall: %.3f'% recall)

    metrics = {
        "mAP" : APs,
        "Precision":APs,
        "recall": recall
    }

    with open("metrics.json", 'w') as mfile:
        json.dump(metrics, mfile, indent=2)


if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.isfile(args.object_json):
        print('Data Json file does not exist. Creating one.')
        try:
            preprocess_data(dset_path=args.data, output_folder='./')
        except:
            exit('An exception occured while creating Data File')
    evaluate(args)
