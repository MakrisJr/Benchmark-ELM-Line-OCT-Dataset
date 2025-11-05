import torch
from torch.autograd import Function

'''
def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return  s / (i + 1)
'''
def dice_coeff(pred, gt):

    p = pred.contiguous().view(-1)
    g = gt.contiguous().view(-1)
    intersection = (p * g).sum()

    p_sum = torch.sum(p * p)
    g_sum = torch.sum(g * g)
    
    return (2. * intersection + 1) / (p_sum + g_sum + 1) 



def Dice_Loss(pred, gt):

    p = pred.contiguous().view(-1)
    g = gt.contiguous().view(-1)
    intersection = (p * g).sum()

    p_sum = torch.sum(p * p)
    g_sum = torch.sum(g * g)
    
    return 1 - ((2. * intersection + 1) / (p_sum + g_sum + 1) )
