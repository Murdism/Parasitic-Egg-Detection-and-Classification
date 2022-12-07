from PIL import Image
from torchvision import transforms
import torch


def preprocess_image():
    """make the images into restnet 50 model format and normalization"""
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return preprocess


if __name__ == "__main__":
    test_tensor = torch.randint(0, 255, size=(3, 200, 200), dtype=torch.uint8)

    tensor2image_transform = transforms.ToPILImage()
    img = tensor2image_transform(test_tensor)
    input_tensor = preprocess_image()(img)
    print(input_tensor.shape)
