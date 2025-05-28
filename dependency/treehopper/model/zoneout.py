import torch
from torch.autograd import Function
import sys

sys.path.append("..")

class ZoneoutFunction(Function):
    gen = torch.Generator().manual_seed(42)
    @staticmethod
    def forward(ctx, current_input, previous_input, p, training, mask=None):
        if not training or p == 0:
            ctx.save_for_backward(torch.ones_like(current_input), torch.ones_like(previous_input))
            return current_input

        if mask is None:
            mask = torch.bernoulli(torch.full_like(current_input, 1 - p), generator=ZoneoutFunction.gen)
        
        previous_mask = 1 - mask
        ctx.save_for_backward(mask, previous_mask)

        output = mask * current_input + previous_mask * previous_input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, previous_mask = ctx.saved_tensors
        grad_current = grad_output * mask
        grad_previous = grad_output * previous_mask
        return grad_current, grad_previous, None, None, None  # `None` for non-trainable parameters

def zoneout(current_input, previous_input, p=0.15, training=False, mask=None):
    return ZoneoutFunction.apply(current_input, previous_input, p, training, mask)
