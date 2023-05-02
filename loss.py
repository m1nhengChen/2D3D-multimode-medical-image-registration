import torch
import numpy as np
from torch import nn


def cal_ncc(I, J, eps=1e-10):
    # compute local sums via convolution
    cross = (I - torch.mean(I)) * (J - torch.mean(J))
    I_var = (I - torch.mean(I)) * (I - torch.mean(I))
    J_var = (J - torch.mean(J)) * (J - torch.mean(J))

    cc = torch.sum(cross) / torch.sum(torch.sqrt(I_var * J_var + eps))

    # test = torch.mean(cc)
    return torch.mean(cc)


# NCC loss
def ncc(I, J, device='cuda', win=None, eps=1e-10):
    return 1 - cal_ncc(I, J, eps)


# cosine similarity
def cos_sim(a, b, device='cuda', win=None, eps=1e-10):
    return torch.sum(torch.multiply(a, b)) / (np.sqrt(torch.sum((a) ** 2)) * np.sqrt(torch.sum((b) ** 2)) + eps)


# NCCL loss

def NCCL(I, J, device='cuda', kernel_size=5, win=None, eps=1e-10):
    '''
    Normalized cross-correlation (NCCL) based on the LOG
    operator is obtained. The Laplacian image is obtained by convolution of the reference image
    and DRR image with the LOG operator. The zero-crossing point in the Laplacian image
    is no longer needed to obtain the image’s detailed edge. However, two Laplacian images’
    consistency is directly measured to use image edge and detail information effectively. This
    paper uses cosine similarity to measure the similarity between Laplacian images.
    '''
    # compute filters

    with torch.no_grad():
        if kernel_size == 5:
            kernel_LoG = torch.Tensor([[[[-2, -4, -4, -4, -2], [-4, 0, 8, 0, -4], [-4, 8, 24, 8, -4], [-4, 0, 8, 0, -4],
                                         [-2, -4, -4, -4, -2]]]])
            kernel_LoG = torch.nn.Parameter(kernel_LoG, requires_grad=False)
            LoG = nn.Conv2d(1, 1, 5, 1, 1, bias=False)
        elif kernel_size == 9:
            kernel_LoG = torch.Tensor([[[[0, 1, 1, 2, 2, 2, 1, 1, 0],
                                         [1, 2, 4, 5, 5, 5, 4, 2, 1],
                                         [1, 4, 5, 3, 0, 3, 5, 4, 1],
                                         [2, 5, 3, -12, -24, -12, 3, 5, 2],
                                         [2, 5, 0, -24, -40, -24, 0, 5, 2],
                                         [2, 5, 3, -12, -24, -12, 3, 5, 2],
                                         [1, 4, 5, 3, 0, 3, 4, 4, 1],
                                         [1, 2, 4, 5, 5, 5, 4, 2, 1],
                                         [0, 1, 1, 2, 2, 2, 1, 1, 0]]]])
            kernel_LoG = torch.nn.Parameter(kernel_LoG, requires_grad=False)
            LoG = nn.Conv2d(1, 1, 9, 1, 1, bias=False)
        LoG.weight = kernel_LoG
        LoG = LoG.to(device)
    LoG_I = LoG(I)
    LoG_J = LoG(J)
    # cosine_similarity
    return 1.5 - cal_ncc(I, J) - 0.5 * cos_sim(LoG_I, LoG_J)


def NCCS(I, J, device='cuda', kernel_size=5, win=None, eps=1e-10):
    '''Normalized Cross-Correlation Based on Sobel Operator'''
    # compute filters
    with torch.no_grad():
        kernel_X = torch.Tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        kernel_X = torch.nn.Parameter(kernel_X, requires_grad=False)
        kernel_Y = torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        kernel_Y = torch.nn.Parameter(kernel_Y, requires_grad=False)
        SobelX = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelX.weight = kernel_X
        SobelY = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelY.weight = kernel_Y

        SobelX = SobelX.to(device)
        SobelY = SobelY.to(device)

    Ix = SobelX(I)
    Iy = SobelY(I)
    Jx = SobelX(J)
    Jy = SobelY(J)
    delta_I = torch.sqrt(Ix ** 2 + Iy ** 2)
    delta_J = torch.sqrt(Jx ** 2 + Jy ** 2)
    return 1 - cal_ncc(I, J) * cos_sim(delta_I, delta_J)
