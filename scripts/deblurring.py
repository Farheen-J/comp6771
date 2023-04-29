## Imports
import os
import time
import argparse
import models
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt

# Import modules from torchvision
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

# Import modules from torchvision
from sklearn.model_selection import train_test_split

# Parse command line arguments using argparse
command_line_parser = argparse.ArgumentParser()
command_line_parser.add_argument('-e', '--epochs', type=int, default=50,
                                 help='Number of Epochs')
args = vars(command_line_parser.parse_args())

# Define helper functions
# Create a directory to save the deblurred images
image_directory = '../output/images'
os.makedirs(image_directory, exist_ok=True)


# Define a function to save an image in a format that can be opened by PIL
def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)


# Check if CUDA is available and set the device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

# Set the batch size
batch_size = 2

# Load the blurred and sharp images
gaussian_blur = os.listdir('../input/gaussian_blurred/')
gaussian_blur.sort()
sharp_images = os.listdir('../input/sharp')
sharp_images.sort()

x_blurred = []
for i in range(len(gaussian_blur)):
    x_blurred.append(gaussian_blur[i])

y_sharp = []
for i in range(len(sharp_images)):
    y_sharp.append(sharp_images[i])

print(x_blurred[10])
print(y_sharp[10])

# Split the data into training and validation sets
(x_train, x_validation, y_train, y_validation) = train_test_split(x_blurred, y_sharp, test_size=0.25)

# Print the number of images in the training and validation sets
print(len(x_train))
print(len(x_validation))

# Define transforms to be applied to the images
transform = transforms.Compose([
    transforms.ToPILImage(),  # convert the numpy array to a PIL Image
    transforms.Resize((224, 224)),  # resize the image to 224x224 pixels
    transforms.ToTensor(),  # convert the PIL Image to a PyTorch tensor
])


# Define a custom Dataset class to load the data
class DeblurDatasetLoader(Dataset):
    def __init__(self, blurred_paths, sharped_paths=None, transforms=None):
        self.X = blurred_paths
        self.y = sharped_paths
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        # Load the blurred image
        blurred_image = cv2.imread(f"../input/gaussian_blurred/{self.X[i]}")

        # Apply the transforms to the blurred image
        if self.transforms:
            blurred_image = self.transforms(blurred_image)

        # If the sharp image is available, load and transform it as well
        if self.y is not None:
            sharp_image = cv2.imread(f"../input/sharp/{self.y[i]}")
            sharp_image = self.transforms(sharp_image)
            return blurred_image, sharp_image
        else:
            return blurred_image


train_data = DeblurDatasetLoader(x_train, y_train, transform)
validation_data = DeblurDatasetLoader(x_validation, y_validation, transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

# Model
nn_model = models.CNN().to(device)
print(nn_model)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=5,
    factor=0.1,
    verbose=True
)


def fit(nn_model, dataloader, epoch):
    # Set the nn_model in training mode
    nn_model.train()

    # Initialize a variable to keep track of the running loss
    running_loss = 0.0

    # Loop over the dataloader with tqdm to show the progress of the loop.
    # total is used to specify the number of steps in the progress bar.
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):

        # Retrieve the blurred and sharp images from the current batch
        blurred_image = data[0]
        sharp_image = data[1]

        # Retrieve the blurred and sharp images from the current batch
        blurred_image = blurred_image.to(device)
        sharp_image = sharp_image.to(device)

        # Clear the gradients
        optimizer.zero_grad()

        # Pass the blurred image through the nn_model to get the output image
        output = nn_model(blurred_image)

        # Calculate the loss between the output and sharp image
        loss = criterion(output, sharp_image)

        # Perform backpropagation to compute the gradients
        loss.backward()

        # Update the parameters of the nn_model using the computed gradients
        optimizer.step()

        # Add the current loss to the running loss
        running_loss += loss.item()

    # Calculate the average train loss for the epoch
    train_loss = running_loss / len(dataloader.dataset)

    # Print the train loss
    print(f"Train Loss: {train_loss:.5f}")

    # Return the average train loss for the epoch
    return train_loss


# the training function
def validate(nn_model, dataloader, epoch):
    # Sets the model to evaluation mode
    nn_model.eval()
    running_loss = 0.0

    # Disables gradient computation for efficiency
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(validation_data) / dataloader.batch_size)):
            blurred_image = data[0]
            sharp_image = data[1]
            blurred_image = blurred_image.to(device)
            sharp_image = sharp_image.to(device)

            # Forward pass
            output = nn_model(blurred_image)

            # Calculate loss
            loss = criterion(output, sharp_image)
            running_loss += loss.item()

            # Saves the decoded image for the first batch of the first epoch
            if epoch == 0 and i == (len(validation_data) / dataloader.batch_size) - 1:
                save_decoded_image(sharp_image.cpu().data, name=f"../output/images/sharp_{epoch}.jpg")
                save_decoded_image(blurred_image.cpu().data, name=f"../output/images/blurred_{epoch}.jpg")

        # Calculate validation loss
        validation_loss = running_loss / len(dataloader.dataset)
        print(f"Val Loss: {validation_loss:.5f}")

        # Saves the decoded image for the current epoch
        save_decoded_image(output.cpu().data, name=f"../output/images/validation_output_{epoch}.jpg")

        # Return validation loss
        return validation_loss


train_loss = []
validation_loss = []

# Record the start time of training
start = time.time()

# Loop over the specified number of epochs
for epoch in range(args['epochs']):
    print(f"Epoch {epoch + 1} of {args['epochs']}")

    # Train the model and get the training loss for this epoch
    train_epoch_loss = fit(nn_model, train_loader, epoch)

    # Validate the model and get the validation loss for this epoch
    validation_epoch_loss = validate(nn_model, validation_loader, epoch)

    # Append the training and validation loss to their respective lists
    train_loss.append(train_epoch_loss)
    validation_loss.append(validation_epoch_loss)

    # Adjust the learning rate scheduler based on validation loss
    scheduler.step(validation_epoch_loss)

# Record the end time of training
end = time.time()

print(f"Took {((end - start) / 60):.3f} minutes to train")

# Plotting loss function
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='red', label='train loss', linewidth=2)
plt.plot(validation_loss, color='blue', label='validation loss', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../output/loss.png')
plt.show()

# save the model to disk
print('Saving model...')
torch.save(nn_model.state_dict(), '../output/model.pth')
