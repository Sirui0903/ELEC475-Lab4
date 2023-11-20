import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
import cv2,  argparse

from KittiDataset import KittiDataset
from KittiAnchors import Anchors



def strip_ROIs(class_ID, label_list):
    ROIs = []
    for i in range(len(label_list)):
        ROI = label_list[i]
        if ROI[1] == class_ID:
            pt1 = (int(ROI[3]),int(ROI[2]))
            pt2 = (int(ROI[5]), int(ROI[4]))
            ROIs += [(pt1,pt2)]
    return ROIs


def calc_IoU(boxA, boxB):
    # print('break 209: ', boxA, boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][1], boxB[0][1])
    yA = max(boxA[0][0], boxB[0][0])
    xB = min(boxA[1][1], boxB[1][1])
    yB = min(boxA[1][0], boxB[1][0])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1][1] - boxA[0][1] + 1) * (boxA[1][0] - boxA[0][0] + 1)
    boxBArea = (boxB[1][1] - boxB[0][1] + 1) * (boxB[1][0] - boxB[0][0] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument('-d', metavar='display', type=str, help='[y/N]')
    Parser.add_argument('--img_size', type=int, default=224)
    Parser.add_argument('--IoU_threshold', type=float, default=0.02)

    args = Parser.parse_args()

    show_images = False
    if args.d != None:
        if args.d == 'y' or args.d == 'Y':
            show_images = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Yoda model
    yoda_model = models.resnet18()
    num_features = yoda_model.fc.in_features
    yoda_model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Add dropout layer
        nn.Linear(num_features, 2)  # Assuming 2 classes (Car, NoCar)
    )
    yoda_model.load_state_dict(torch.load("./yoda_classifier_epoch39_batchsize128.pth"))
    yoda_model.to(device)
    yoda_model.eval()

    dataset = KittiDataset("./Kitti8", training=False)
    anchors = Anchors()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    all_IoUs = []
    true_labels = []
    predicted_labels = []

    i = 0
    for item in enumerate(dataset):
        idx = item[0]
        image = item[1][0]
        label = item[1][1]
        # print(i, idx, label)

        # 1. Get the car_ROIs
        idx = dataset.class_label['Car']
        car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)
        # print(car_ROIs)
        # for idx in range(len(car_ROIs)):
            # print(ROIs[idx])


        # 2. Subdivid the Kitti images using a regular grid
        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
        image1 = image.copy()
        if show_images:
            for j in range(len(anchor_centers)):
                x = anchor_centers[j][1]
                y = anchor_centers[j][0]
                cv2.circle(image1, (x, y), radius=4, color=(255, 0, 255))

        image2 = image1.copy()

        ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)
        # print('break 555: ', boxes)

        roi_tensors = [transform(Image.fromarray(roi)) for roi in ROIs]
        batch = torch.stack(roi_tensors)
        batch = batch.to(device)
        for x in range(len(batch)):
            batch[x] = torch.cat((batch[x][-1:],batch[x][1:-1],batch[x][:1]),axis = 0)
        with torch.no_grad():
            predictions = yoda_model(batch)
            _, predicted_label = torch.max(predictions, 1)



        print(predicted_label)

        ROI_IoUs = []
        yoda_IoUs = []
        for idx in range(len(ROIs)):
            # ROI classied as a ‘Car’, calculate its IoU score against the original Kitti image.
            if predicted_label[idx] == 1:
                yoda_IoUs += [anchors.calc_max_IoU(boxes[idx], car_ROIs)]
                all_IoUs += [anchors.calc_max_IoU(boxes[idx], car_ROIs)]
                box = boxes[idx]
                pt1 = (box[0][1],box[0][0])
                pt2 = (box[1][1],box[1][0])
                cv2.rectangle(image1, pt1, pt2, color=(0, 255, 255))

            ROI_IoUs += [anchors.calc_max_IoU(boxes[idx], car_ROIs)]


        # print(ROI_IoUs)


        if show_images:
            cv2.imshow('yoda_ROIs', image1)

            for k in range(len(boxes)):
                if ROI_IoUs[k] > args.IoU_threshold:
                    box = boxes[k]
                    pt1 = (box[0][1],box[0][0])
                    pt2 = (box[1][1],box[1][0])
                    cv2.rectangle(image2, pt1, pt2, color=(0, 255, 255))

            cv2.imshow('car_ROIs', image2)
            key = cv2.waitKey(0)
            if key == ord('x'):
                break

        true_labels += [1 if len(car_ROIs) > 0 else 0]
        predicted_labels += [1 if torch.sum(predicted_label == 1) > 0 else 0]

        i += 1
        # print(i)

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)*100
    mean_iou = sum(all_IoUs) / len(all_IoUs)

    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Accuracy: {accuracy:.4f}%")
    print(f"Mean IoU over all detected 'Car' ROIs: {mean_iou}")

###################################################################

main()
