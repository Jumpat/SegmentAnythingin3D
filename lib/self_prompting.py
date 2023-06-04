'''
    Self prompting strategy
    
    INPUT: 
        predictor: the initialized sam predictor :
        rendered_mask_score: - : H*W*1
        num_prompt: -  
        index_matrix: the matrix contains the 3D index of the rendered view : H*W*3
    OUTPUT: a list of prompts
'''
import os
import torch
import math
import numpy as np

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

@torch.no_grad()
def mask_to_prompt(predictor, rendered_mask_score, index_matrix, num_prompts = 3):
    '''main function for self prompting'''
    h, w, _ = rendered_mask_score.shape
    tmp = rendered_mask_score.view(-1)
    print("tmp min:", tmp.min(), "tmp max:", tmp.max())
    rand = torch.ones_like(tmp)
    topk_v, topk_p = torch.topk(tmp*rand, k = 1)[0].cpu(), torch.topk(tmp*rand, k = 1)[1].cpu()

    if topk_v <= 0:
        print("No prompt is available")
        return np.zeros((0,2)), np.ones((0))

    prompt_points = []
    prompt_points.append([topk_p[0] % w, topk_p[0] // w])

    print((topk_p[0] % w).item(), (topk_p[0] // w).item(), h, w)

    tmp_mask = rendered_mask_score.clone().detach()

    area = to8b(tmp_mask.cpu().numpy()).sum() / 255
    r = np.sqrt(area / math.pi)
    masked_r = max(int(r) // 2, 2)
    # masked_r = max(int(r) // 3, 2)

    pre_tmp_mask_score = None
    for _ in range(num_prompts - 1):
        # mask out a region around the last prompt point
        input_label = np.ones(len(prompt_points))
        previous_masks, previous_scores, previous_logits = predictor.predict(
            point_coords=np.array(prompt_points),
            point_labels=input_label,
            multimask_output=False,
        )

        l = 0 if prompt_points[-1][0]-masked_r <= 0 else prompt_points[-1][0]-masked_r
        r = w-1 if prompt_points[-1][0]+masked_r >= w-1 else prompt_points[-1][0]+masked_r

        t = 0 if prompt_points[-1][1]-masked_r <= 0 else prompt_points[-1][1]-masked_r
        b = h-1 if prompt_points[-1][1]+masked_r >= h-1 else prompt_points[-1][1]+masked_r
        tmp_mask[t:b+1, l:r+1, :] = -1e5

        # bool: H W
        previous_mask_tensor = torch.tensor(previous_masks[0])
        previous_mask_tensor = previous_mask_tensor.unsqueeze(0).unsqueeze(0).float()
        previous_mask_tensor = torch.nn.functional.max_pool2d(previous_mask_tensor, 25, stride = 1, padding = 12)
        previous_mask_tensor = previous_mask_tensor.squeeze(0).permute([1,2,0])
#         tmp_mask[previous_mask_tensor > 0] = -1e5
        previous_max_score = torch.max(rendered_mask_score[previous_mask_tensor > 0])

        previous_point_index = torch.zeros_like(index_matrix)
        previous_point_index[:,:,0] = prompt_points[-1][1] / h
        previous_point_index[:,:,1] = prompt_points[-1][0] / w
        previous_point_index[:,:,2] = index_matrix[int(prompt_points[-1][1]), int(prompt_points[-1][0]), 2]
        distance_matrix = torch.sqrt(((index_matrix - previous_point_index)**2).sum(-1))
        distance_matrix = (distance_matrix.unsqueeze(-1) - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())

        cur_tmp_mask = tmp_mask - distance_matrix * max(previous_max_score, 0)

        if pre_tmp_mask_score is None:
            pre_tmp_mask_score = cur_tmp_mask
        else:
            pre_tmp_mask_score[pre_tmp_mask_score < cur_tmp_mask] = cur_tmp_mask[pre_tmp_mask_score < cur_tmp_mask]
            pre_tmp_mask_score[tmp_mask == -1e5] = -1e5

        tmp_val_point = pre_tmp_mask_score.view(-1).max(dim = 0)

        if tmp_val_point[0] <= 0:
            print("There are", len(prompt_points), "prompts")
            break
        prompt_points.append([int(tmp_val_point[1].cpu() % w), int(tmp_val_point[1].cpu() // w)])

    prompt_points = np.array(prompt_points)
    input_label = np.ones(len(prompt_points))

    return prompt_points, input_label


from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import groundingdino.datasets.transforms as T
from torchvision.ops import box_convert
from PIL import Image

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


def grounding_dino_prompt(image, text):
    
    image_tensor = image_transform(Image.fromarray(image))
    model_root = './dependencies/GroundingDINO'
    
    model = load_model(os.path.join(model_root, "groundingdino/config/GroundingDINO_SwinT_OGC.py"), os.path.join(model_root, "weights/groundingdino_swint_ogc.pth"))
    
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    
    h, w, _ = image.shape
    print("boxes device", boxes.device)
    boxes = boxes * torch.Tensor([w, h, w, h]).to(boxes.device)
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    print(xyxy)
    return xyxy



'''
# new prompt strategy: bbox based
# cannot be applied to 360
try:
    prompt = seg_m_for_prompt[:,:,no]
    prompt = prompt > 0
    box_prompt = masks_to_boxes(prompt.unsqueeze(0))
    width = box_prompt[0,2] - box_prompt[0,0]
    height = box_prompt[0,3] - box_prompt[0,1]
    box_prompt[0,0] -= 0.05*width
    box_prompt[0,2] += 0.05*width
    box_prompt[0,1] -= 0.05*height
    box_prompt[0,3] += 0.05*height
#                             print(box_prompt)
    transformed_boxes = predictor.transform.apply_boxes_torch(box_prompt, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    masks = masks.float()
except:
    continue
'''


'''
# mask based 

H,W,_ = prompt.shape
target_size = RLS.get_preprocess_shape(H,W, 256)
prompt = torch.nn.functional.interpolate(torch.tensor(prompt).float().unsqueeze(0).permute([0,3,1,2]), target_size, mode = 'bilinear')
h,w = prompt.shape[-2:]
padh = 256 - h
padw = 256 - w
prompt = F.pad(prompt, (0, padw, 0, padh))
prompt = (prompt / 255) * 40 - 20
print(prompt.shape)
prompt = prompt.squeeze(1).cpu().numpy()

masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        mask_input=prompt,
        multimask_output=False,
    )
'''