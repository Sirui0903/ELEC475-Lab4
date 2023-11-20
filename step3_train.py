import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2, os, time
from PIL import Image
from tqdm import tqdm
import argparse

batch = 256

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


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="")
parser.add_argument('-epoch', type=int, default=40)
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('--img_size', type=int, default=224)

args = parser.parse_args()


# Create YODA datasets
transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
])

train_dataset = YodaDataset(data_dir='./Kitti8_ROIs/train', transform=transform)
test_dataset = YodaDataset(data_dir='./Kitti8_ROIs/test', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Create ResNet model
resnet18_model = models.resnet18(pretrained=True)
num_features = resnet18_model.fc.in_features
resnet18_model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Add dropout layer
    nn.Linear(num_features, 2)  # Assuming 2 classes (Car, NoCar)
)

# Move model to device
resnet18_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18_model.parameters(), lr=0.001, weight_decay=1e-4)

# Training loop
train_losses = []
test_losses = []
best_test_loss = float('inf')

for epoch in range(args.epoch):
    resnet18_model.train()
    running_loss = 0.0

    # Use tqdm for progress bar
    tqdm_train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}, Training")

    for inputs, labels in tqdm_train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = resnet18_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        tqdm_train_loader.set_postfix(loss=running_loss / len(tqdm_train_loader), refresh=True)

    train_losses.append(running_loss / len(train_loader))

    # Validation
    resnet18_model.eval()
    test_loss = 0.0

    # Use tqdm for progress bar
    tqdm_test_loader = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epoch}, Testing")

    with torch.no_grad():
        for inputs, labels in tqdm_test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = resnet18_model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            tqdm_test_loader.set_postfix(loss=test_loss / len(tqdm_test_loader), refresh=True)

    test_losses.append(test_loss / len(test_loader))

    tqdm_train_loader.close()
    tqdm_test_loader.close()

    # Inside the epoch loop, after calculating the test loss
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(resnet18_model.state_dict(), f'best_model.pth')

    print(f"Epoch [{epoch+1}/{args.epoch}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

# Save the trained model
# t = time.localtime()
# timestamp = time.strftime('%b-%d-%Y_%H%M', t)
torch.save(resnet18_model.state_dict(), f'yoda_classifier_epoch{epoch}_batchsize{args.batch_size}.pth')

# Plot the loss over epochs
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'./epoch{epoch}_batchsize{args.batch_size}.png')
plt.show()


