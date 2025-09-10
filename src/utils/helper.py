def count_num_parms(model):
    '''
    counts the number of trainable parameters in a model
    '''
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total number of trainable parameters: {num_trainable_params}"