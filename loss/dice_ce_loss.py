import torch
import torch.nn as nn
from dice_loss import MultiDiceLossW

class DiceCELoss(nn.Module):
    def __init__(self, 
                 include_background, 
                 squared_pred,
                 activation=True, 
                 lambda_dice: float = 1.0,
                 lambda_ce: float = 1.0,
                 ):
        super(DiceCELoss, self).__init__()
        self.lambda_dice, self.lambda_ce = lambda_dice, lambda_ce
        self.dice = MultiDiceLossW(
            weight=None, 
            activation=activation,
            include_background=include_background,
            squared_pred=squared_pred)
        # ce must input inactive data
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input must be inactive data
        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss
        return total_loss
