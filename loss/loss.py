class MultiDiceLossW(nn.Module):
    ''' multi-class dice loss with weight. '''
    def __init__(self, 
                 weight=None, 
                 activation=False, 
                 include_background=False,
                 squared_pred=False,
                 ):
        super(MultiDiceLossW, self).__init__()
        self.weight = weight
        self.squared_pred = squared_pred
        self.include_background = include_background
        self.activation = activation

    def forward(self, logit, labels):
        ''' logit and labels must be one-hot. '''
        cls = labels.shape[1]
        
        outs = logit
        if self.activation == True:
            outs = torch.sigmoid(outs) if cls == 1 else F.softmax(outs, dim=1)

        if self.include_background == False:
            if cls == 1:
                print("single channel prediction, `include_background=False` ignored.")
            else:
                cls -= 1 
                outs = outs[:, 1:]
                labels = labels[:, 1:]
        labels = labels.float()

        self.weight = [1.0] * cls if self.weight is None else self.weight

        total_loss = 0
        for i in range(cls):
            _i_dice_loss = _dice_loss(outs[:, i], labels[:, i], self.squared_pred, 1)
            _i_w_dice_loss = _i_dice_loss * self.weight[i]
            total_loss += _i_w_dice_loss
        return total_loss / cls


class DiceCELoss(nn.Module):
    def __init__(self, 
                 include_background, 
                 squared_pred,
                 activation=False, 
                 lambda_dice: float = 1.0,
                 lambda_ce: float = 1.0,
                 ):
        super(DiceCELoss, self).__init__()
        assert  0.0 <= lambda_dice <= 1.0, \
            "lambda_dice should not be less than 0.0 and should not be greater than 1.0"
        assert  0.0 <= lambda_ce <= 1.0, \
            "lambda_dice should not be less than 0.0 and should not be greater than 1.0"
        self.lambda_dice, self.lambda_ce = lambda_dice, lambda_ce

        self.dice = MultiDiceLossW(
            weight=None, 
            activation=False,
            include_background=include_background,
            squared_pred=squared_pred)
        
        # one-hot data must be processed with softmax
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss
        return total_loss
