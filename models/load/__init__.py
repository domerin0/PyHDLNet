
def load_torch(model_path, model_class=None):
    import torch
    if model_class is None:
        raise ValueError("model_class must be specified")

    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def quantize_tensor(tensor, bits=8, signed=False, symmetric=True):
    '''
        Quantization that supports signed and unsigned integers 
        as well as symmetric or asymetric ranges.
    '''
    import torch
    max_val = 2**(bits) - 1 if not signed else 2**(bits - 1) - 1
    min_val = 0 if not signed else -2**(bits - 1)
    max_neg_val = torch.abs(torch.min(tensor))
    max_pos_val = torch.abs(torch.max(tensor))
    tensor = tensor.clone()
    max_tensor_val = torch.max(torch.abs(tensor))
    if symmetric:
        if signed:
            tensor = (tensor * (max_val / max_tensor_val))
        else:
            tensor = (tensor * ((max_val // 2) / max_tensor_val)) + \
                (max_val // 2)
    else:
        if signed:
            pos_tensor = torch.where(
                tensor > 0, tensor * (max_val / max_pos_val), 0)
            neg_tensor = torch.where(
                tensor < 0, tensor * torch.abs(min_val / max_neg_val), 0)
            tensor = pos_tensor + neg_tensor
        else:
            pos_tensor = torch.where(
                tensor > 0,
                tensor * ((max_val//2) / max_pos_val) + (max_val // 2),
                0
            )
            neg_tensor = torch.where(
                tensor < 0,
                tensor * torch.abs((max_val//2) /
                                   max_neg_val) + (max_val // 2),
                0
            )
            tensor = pos_tensor + neg_tensor

    if bits == 8:
        dtype = torch.int8 if signed else torch.uint8
    elif bits == 16:
        dtype = torch.int16 if signed else torch.uint16
    else:
        dtype = torch.int32

    tensor = torch.clamp(
        torch.round(tensor),
        min=min_val,
        max=max_val
    ).type(dtype).data.tolist()

    return tensor
