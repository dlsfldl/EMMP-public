import torch
import torch.nn as nn
class BaseModel(nn.Module):
    def __init__(self) -> None:
        self.device = torch.device('cpu')
        super().__init__()
    
    def to(self, *args, **kwargs):
        model = super().to(*args, **kwargs)
        if type(args[0]) == torch.device:
            model.device = args[0]
        elif hasattr(args[0], 'device'):
            model.device = args[0].device
        elif type(args[0]) == str:
            model.device = args[0]
        return model
    
    def cuda(self, *args, **kwargs):
        model = super().cuda(*args, **kwargs)
        model.device = torch.device('cuda')
        return model
    
    def cpu(self, *args, **kwargs):
        model = super().cpu(*args, **kwargs)
        model.device = torch.device('cpu')
        return model