import torch
import argparse
import cv2
import numpy as np
from segmentation_models_pytorch import UnetPlusPlus
from albumentations import Compose, Normalize
from albumentations.pytorch.transforms import ToTensorV2


def preprocess_image(image_path, img_size=(256, 256)):
    image = cv2.imread(image_path)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    transform = Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    image = cv2.resize(image, img_size)  # Resize 
    transformed = transform(image=image)
    return transformed["image"].unsqueeze(0)  


def postprocess_mask(mask, color_dict):
    mask = torch.argmax(mask, dim=1).squeeze().cpu().numpy()  
    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in color_dict.items():
        output[mask == class_id] = color
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for segmentation")
    parser.add_argument("--image_path", required=True, help="Đường dẫn tới ảnh đầu vào")
    args = parser.parse_args()

    checkpoint_path = "checkpoint/best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnetPlusPlus(
        encoder_name="efficientnet-b6",
        encoder_weights=None,
        in_channels=3,
        classes=3
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    input_image = preprocess_image(args.image_path)

    with torch.no_grad():
        input_image = input_image.to(device)
        output_mask = model(input_image)


    color_dict = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}  
    output_mask = postprocess_mask(output_mask, color_dict)
    output_path = "working/segmented_output.png"
    cv2.imwrite(output_path, cv2.cvtColor(output_mask, cv2.COLOR_RGB2BGR))
    print(f"Output Image saved at: {output_path}")
