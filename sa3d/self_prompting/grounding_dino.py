import os
import numpy as np
import torch
from torchvision.ops import box_convert
from PIL import Image
import cv2
from rich.console import Console
CONSOLE = Console(width=120)

from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T


class GroundingDino:
    def __init__(self) -> None:
        self.BOX_TRESHOLD = 0.35
        self.TEXT_TRESHOLD = 0.25
        model_root = 'sa3d/self_prompting/dependencies/GroundingDINO'
        self.model = load_model(os.path.join(model_root, "groundingdino/config/GroundingDINO_SwinT_OGC.py"),\
                            os.path.join(model_root, "weights/groundingdino_swint_ogc.pth"))
        CONSOLE.print('GroundingDino loaded!')
    
    @staticmethod
    def image_transform(image) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(image, None)
        return image_transformed

    @torch.no_grad()
    def __call__(self, image, text):
        image_tensor = GroundingDino.image_transform(Image.fromarray(image))
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=text,
            box_threshold=self.BOX_TRESHOLD,
            text_threshold=self.TEXT_TRESHOLD
        )
        h, w, _ = image.shape
        boxes = boxes * torch.Tensor([w, h, w, h]).to(boxes.device)
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        CONSOLE.print('Get box from GroundingDino: ', xyxy)
        return xyxy