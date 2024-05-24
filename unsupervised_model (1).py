import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split

# Model
class SimpleDepthEstimator(nn.Module):
    def __init__(self):
        super(SimpleDepthEstimator, self).__init__()
        # Use a pre-trained model as a feature extractor
        self.features = models.resnet18(pretrained=True)
        for param in self.features.parameters():
            param.requires_grad = False
        # Re-activate gradient updates for last layers
        for param in self.features.layer4.parameters():
            param.requires_grad = True
        self.features.fc = nn.Identity()  # Remove the classification head

        # Add custom layers for depth estimation
        self.depth_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output a single depth value for simplicity
        )

    def forward(self, x):
        x = self.features(x)  # Extract features
        depth = self.depth_layers(x)  # Estimate depth
        return depth



# Define transformations for training and validation
# Anton Comment :) 
# Consider augmentations that simulate different lighting conditions, add noise, or apply geometric transformations. 
# These can make your model more robust to various real-world conditions.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Load all images from the dataset folder
data_folder = '/C:\Users\nanot\Desktop\conti-ocv\OneDrive_2024-03-25\Datasett (Ferdig)'


image_filenames = os.listdir(data_folder)
images = []
for filename in image_filenames:
    img_path = os.path.join(data_folder, filename)
    image = Image.open(img_path)
    images.append(image)

# Split the data into training, validation, and testing sets
train_images, test_val_images = train_test_split(images, test_size=0.4, random_state=42)
val_images, test_images = train_test_split(test_val_images, test_size=0.5, random_state=42)


# Dataset class for tire images
class TireDataset(Dataset):
    """Basic Tire Image Dataset"""

    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
        # Anton comment :)
        # Here it seems like you are forgetting to read the target depth (actual depth) of each image ie y variable

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.ToTensor()
            image = transform(image)

        return image

# Dataset and dataloader setup
train_dataset = TireDataset(train_images, transform=train_transform)
val_dataset = TireDataset(val_images, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Training set size: {len(train_images)}")
print(f"Testing set size: {len(test_images)}")
print(f"Validation set size: {len(val_images)}")


# Instantiate the model
# Anton Comment :)
# Have other model architecture been considered or tested? 
model = SimpleDepthEstimator()

# Loss function and optimizer
# Anton Comment :)
# Have other evalution metrics been considered? 
criterion = nn.MSELoss()
optimizer = optim.Adam([{'params': model.features.layer4.parameters(), 'lr': 1e-4},
                        {'params': model.depth_layers.parameters()}], lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# Training loop
num_epochs = 25
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs in train_loader:
        # Forward pass
        outputs = model(inputs)

        # Anton comment :)
        # This line computes the mean squared error (MSE) between the model's predictions and a tensor of zeros with the same shape as the output.
        # This approach doesn't make sense for depth estimation because it essentially trains the model to predict zero depth for all inputs, which is not the desired behavior.
        # This would also explain the extremely low validation loss obsverved in our latest meeting
        loss = criterion(outputs, torch.zeros_like(outputs))  # Dummy target for depth estimation

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation loop
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, torch.zeros_like(outputs))  # Dummy target for depth estimation

            running_val_loss += loss.item() * inputs.size(0)

    epoch_val_loss = running_val_loss / len(val_loader.dataset)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

    scheduler.step(epoch_val_loss)


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # Anton Comment/Question :) 
    # How come you are doing the RGB conversion only on the validation set? 
    # Inconsistency in data preprocessing can often lead to confusion and potentially degrade the models performance. 
    transforms.Lambda(lambda x: x.convert('RGB')),  
    transforms.ToTensor(),
])




img_path = r'C:\Users\nanot\Desktop\dette\IMG_3260.jpg'  
image = Image.open(img_path)


image = val_transform(image)  


image_batch = image.unsqueeze(0)  


model.eval()  #
with torch.no_grad():
    depth_prediction = model(image_batch)


depth_prediction = depth_prediction.item()  

print(f"Predicted depth: {depth_prediction}")
