import cv2
import models
import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

device = 'cpu'

# load the trained model
model = models.CNN().to(device).eval()
model.load_state_dict(torch.load('../output/model.pth'))

# define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

name = 'flower'

image = cv2.imread(f"../testing_data/gaussian_blurred/{name}.jpg")
original_image = image.copy()
original_image = cv2.resize(original_image, (224, 224))
cv2.imwrite(f"../output/test_deblurred_images/blurred_image.jpg", original_image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = transform(image).unsqueeze(0)
print(image.shape)

with torch.no_grad():
    output = model(image)
    save_decoded_image(output.cpu().data, name=f"../output/test_deblurred_images/deblurred_image.jpg")

