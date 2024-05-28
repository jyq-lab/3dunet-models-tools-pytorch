def _dice_loss(predict, target, pow=False, alpha=1):
    smooth = 1e-5

    intersection = torch.sum(predict * target)
    if pow:
        predict = torch.pow(predict, 2)
        target = torch.pow(target, 2)

    union = torch.sum(predict) + torch.sum(target) * alpha 
    _dice = ( ((1 + alpha) * intersection) + smooth ) / ( union + smooth )
    return 1 - _dice


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
            outs = torch.sigmoid(outs) if cls == 1 else torch.softmax(outs, dim=1)

        if self.include_background == False:
            if cls == 1:
                print("single channel prediction, `include_background=False` ignored.")
            else:
                cls -= 1 
                outs = outs[:, 1:]
                labels = labels[:, 1:]
        labels = labels.float()

        self.weight = [1.0] * cls if self.weight is None else self.weight
        assert len(self.weight) == cls, " weight must correspond to category(include background) "

        total_loss = 0
        for i in range(cls):
            _i_dice_loss = _dice_loss(outs[:, i], labels[:, i], self.squared_pred, 10)
            print("{:.4f}".format(_i_dice_loss.item()), end=' ')

            _i_w_dice_loss = _i_dice_loss * self.weight[i]
            total_loss += _i_w_dice_loss
        return total_loss / cls
