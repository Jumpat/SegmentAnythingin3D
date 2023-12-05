import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

''' Misc
'''
mse2psnr = lambda x : -10. * torch.log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

@torch.jit.script
def cal_IoU(a: Tensor, b: Tensor) -> Tensor:
    """Calculates the Intersection over Union (IoU) between two tensors.

    Args:
        a: A tensor of shape (N, H, W).
        b: A tensor of shape (N, H, W).

    Returns:
        A tensor of shape (N,) containing the IoU score between each pair of
        elements in a and b.
    """
    intersection = torch.count_nonzero(torch.logical_and(a == b, a != 0))
    union = torch.count_nonzero(a + b)
    return intersection / union

def to_tensor(array, device=torch.device('cuda')):
    '''cvt numpy array to cuda tensor, if already tensor, do nothing
    '''
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device).float()
    elif isinstance(array, torch.Tensor) and not array.is_cuda:
        array = array.to(device).float()
    else:
        pass
    return array

def print_grad(net, console):
    # print grad check
    v_n = []
    v_v = []
    v_g = []
    for name, parameter in net.named_parameters():
        v_n.append(name)
        console.print('Param name: ', name)
        v_v.append(parameter.detach().cpu().numpy() if parameter is not None else np.array([0, 0]))
        v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else np.array([0, 0]))
    for i in range(len(v_n)):
        console.print('Param name now: ', v_n[i])
        if len(v_v[i]) == 0:
            continue
        if np.max(v_v[i]).item() - np.min(v_v[i]).item() < 1e-6:
            color = '\033[31m' + '\033[1m' + '*'
        else:
            color = '\033[92m' + '\033[1m' + ' '
        console.print('%svalue %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))
        console.print('%sgrad  %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))
