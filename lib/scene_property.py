##NOTE, define the scene property with const values
##TODO: use a interactive backend.
import numpy as np
import torch

INPUT_POINT = {
    'horns': np.array([[450, 300], [480, 500]]),
    'horns_center': np.array([[450, 300], [480, 500]]),
    'horns_left': np.array([[150, 400], [50, 520]]),
    'bicycle': np.array([[700, 0], [600, 100]]),
    'room': np.array([[450,625],[425,600]]),
    'garden': np.array([[380, 200], [380, 135]]),
    'shoerack': np.array([[600,400],[400,280]]),
    'fortress': np.array([[400,500],[500,300]]),
    'orchids': np.array([[400, 300], [400, 180], [800, 580], [750, 300]]),
    'chesstable': np.array([[600,200],[600,500]]),
    'stove': np.array([[0,2300],[80,250]]),
    'trex': np.array([[300, 300], [800, 480], [680, 600], [500, 400],[430,400]]),
    '360v2_bonsai': np.array([[450, 250], [450, 150]]),
    '360v2_kitchen_part': np.array([[280,260],[300,230]]),
    '360v2_kitchen_part2': np.array([[560,300],[580,300]]),
    '360v2_kitchen': np.array([[400,300],[500,200]]),
    '360v2_counter': np.array([[175,60],[175, 100]]),
    'replica_office0': np.array([[400, 200],[350, 250]]),
    'auto': np.array([[]]),
}

INPUT_BOX = {
    'horns': torch.tensor([
                                [200, 0, 700, 650],
                                [0, 300, 200, 600],
                                [780, 400, 920, 500],
                            ]),
    '360v2_bonsai': torch.tensor([
                                [500, 380, 720, 590],
                                [200, 550, 700, 750],
                                [200, 270, 460, 460],
                            ]),
    'orchids': torch.tensor([
                                [500, 380, 720, 590],
                                [200, 550, 700, 750],
                                [200, 270, 460, 460],
                            ]),
    'santarex': torch.tensor([
                                [300, 100, 430, 250],
                                [200, 180, 1000, 750],
                            ]),
    'butcher': torch.tensor([
                                [200, 350, 400, 500],
                                [430, 400, 720, 580],
                                [300, 550, 630, 750],
                                [450,150,700,300],
                                [50, 230, 250, 430],
                            ]),
    'pond' : torch.tensor([
                                [520, 580, 610, 650],
                                [700, 30, 830, 140],
                                [260, 50, 420, 190],
                            ]),
    'bonsai': torch.tensor([
                                [400, 100, 600, 280],
                                [380, 230, 600, 360],
                                [340, 300, 600, 420],
                            ]),
}