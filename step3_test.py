import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, accuracy_score
import os
from PIL import Image
import numpy as np
import argparse

# Defined YODA dataset class (YodaDataset)
class YodaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []

        # read labels.txt
        with open(os.path.join(data_dir, 'labels.txt'), 'r') as file:
            lines = file.readlines()
            count = 0
            for line in lines:
                # count = count + 1
                # if count % 100 != 0:
                #     continue
                filename, name_class, name = line.strip().split(' ')
                self.data.append((filename, int(name_class), name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename, name_class, name = self.data[index]
        image_path = os.path.join(self.data_dir, filename)
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, name_class

parser = argparse.ArgumentParser(description="")
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('-b', '--batch_size', type=int, default=48)

args = parser.parse_args()

# Adjust as needed

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create YODA datasets
transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
])

test_dataset = YodaDataset(data_dir='./Kitti8_ROIs/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

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

all_labels = []
all_predictions = []

with torch.no_grad():
    for data in test_loader:
        img, label = data
        img = img.to(device)
        label = label.to(device)
        output = yoda_model(img)

        _, predicted_label = torch.max(output, 1)


        all_labels.extend(label.cpu().numpy())
        all_predictions.extend(predicted_label.cpu().numpy())
print(predicted_label)

conf_matrix = confusion_matrix(all_labels, all_predictions)
accuracy = (np.trace(conf_matrix) / np.sum(conf_matrix))*100

print(f"Accuracy: {accuracy:.4f}%")
print("Confusion Matrix:")
print(conf_matrix)