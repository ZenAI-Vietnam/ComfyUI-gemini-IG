import base64
import os
from google import genai
from google.genai import types
import torch 
from PIL import Image
import numpy as np
from torch import Tensor
from comfy.comfy_types import IO
from io import BytesIO
from typing import Union, List

def tensor_to_image(tensor):
    # Ensure tensor is in the right format (H, W, C)
    if len(tensor.shape) == 4:
        # If batch dimension exists, take the first image
        tensor = tensor[0]
    
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [0, 1, 2]].numpy()
    return image

def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    """
    Convert PIL image(s) to tensor, matching ComfyUI's implementation.
    
    Args:
        image: Single PIL Image or list of PIL Images
        
    Returns:
        torch.Tensor: Image tensor with values normalized to [0, 1]
    """
    if isinstance(image, list):
        if len(image) == 0:
            return torch.empty(0)
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    # Convert PIL image to RGB if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Return tensor with shape [1, H, W, 3]
    return torch.from_numpy(img_array)[None,]


def generate_image(prompt: str, image, api_key: str):
    client = genai.Client(api_key=(api_key.strip() if api_key and api_key.strip() != ""
                                     else os.environ.get("GEMINI_API_KEY")))

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=[prompt, image],
        config=types.GenerateContentConfig(
            response_modalities=['Text', 'Image']
        )
    )
    
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            image = pil2tensor(image)

            return (image, )

def generate_text(prompt: str, image, api_key: str):
    client = genai.Client(api_key=(api_key.strip() if api_key and api_key.strip() != ""
                                     else os.environ.get("GEMINI_API_KEY")))

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[image, prompt],
    )
    
    print(response.text)
    return (response.text, )

class GeminiImageGenerationNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Define the required input types
        return {
            "required": {
                "image": ("IMAGE", {}),
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "gemini_key": ("STRING", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",) 
    FUNCTION = "generate"
    OUTPUT_NODE = True

    def __init__(self):
        super().__init__()
        

    def generate(self, image: Tensor, text: str, gemini_key: str):
        img_array = tensor_to_image(image)
        img_pil = Image.fromarray(img_array)

        return generate_image(text, img_pil, gemini_key)


class GeminiTextGenerationNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Define the required input types
        return {
            "required": {
                "image": ("IMAGE", {}),
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "gemini_key": ("STRING", {}),
            }
        }

    RETURN_TYPES = ("STRING",) 
    FUNCTION = "generate"
    OUTPUT_NODE = True

    def __init__(self):
        super().__init__()
        

    def generate(self, image: Tensor, text: str, gemini_key: str):
        img_array = tensor_to_image(image)
        img_pil = Image.fromarray(img_array)

        return generate_text(text, img_pil, gemini_key)

NODE_CLASS_MAPPINGS = {
    "Gemini Image Generation": GeminiImageGenerationNode,
    "Gemini Text Generation": GeminiTextGenerationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = { 
    "Gemini Image Generation": "Gemini Image Generation",
    "Gemini Text Generation": "Gemini Text Generation",
}
