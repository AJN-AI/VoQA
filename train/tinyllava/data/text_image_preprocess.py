from PIL import Image
import torch
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_text_image(image):
    image_tensor = preprocess(image).to(torch.float16)
    
    if image_tensor.size(0) == 3 and image_tensor.size(1) == 1024 and image_tensor.size(2) == 1024:
        return image_tensor
    else:
        raise ValueError("not suitable for 3x1024x1024")