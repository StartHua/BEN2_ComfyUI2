import sys
import os
import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
import BEN2

class BackgroundEraseNetwork:
    # Define these as class variables (outside of any method)
    RETURN_TYPES = ("IMAGE", "MASK")  # Added MASK to return types
    RETURN_NAMES = ("image", "mask")  # Added mask to return names
    FUNCTION = "process"
    CATEGORY = "BEN2"
    
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BEN2.BEN_Base().to(device).eval()
        self.model.loadcheckpoints(os.path.join(script_directory, "BEN2_Base.pth"))
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")  # Updated to include mask
    RETURN_NAMES = ("image", "mask")  # Updated to include mask
    FUNCTION = "process_image"
    CATEGORY = "BEN2"
    
    def process_image(self, input_image):
        # Handle the input tensor format from ComfyUI
        if isinstance(input_image, torch.Tensor):
            if input_image.dim() == 4:
                input_image = input_image[0]
            
            if input_image.dim() == 3:
                input_image = input_image.permute(2, 0, 1)
            
            input_image = self.to_pil(input_image)
        
        # Ensure the image is in RGBA mode
        if input_image.mode != 'RGBA':
            input_image = input_image.convert("RGBA")
        
        # Run inference to get the foreground image
        foreground = self.model.inference(input_image)
        
        # Extract alpha channel as mask
        mask = foreground.split()[-1]
        
        # Convert mask to tensor and adjust dimensions for ComfyUI
        mask_tensor = self.to_tensor(mask)
        mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
        
        # Convert the foreground to tensor
        foreground_tensor = self.to_tensor(foreground)
        
        # Convert to ComfyUI format [B, H, W, C]
        foreground_tensor = foreground_tensor.permute(1, 2, 0).unsqueeze(0)
        
        return (foreground_tensor, mask_tensor)

# Export mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BackgroundEraseNetwork": BackgroundEraseNetwork
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BackgroundEraseNetwork": "Background Erase Network Image"
}