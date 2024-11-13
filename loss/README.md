### dice loss
|  |  | 
| ---- | ----|
|weight of each category.|is activation required.|
|`weight`|weight of each category|

`weight` weight of each category.  
  
`include_background` include the background. (single channel prediction, 'include_background=False' is ignored.)  
`squared_pred` use squared versions of targets and predictions in the denominator or not.
```python
from dice_loss import MultiDiceLossW

MultiDiceLossW(weight=[1,1], activation=True, include_background=False, squared_pred=True)
```
### dice+ce loss
`activation` is activation required.  
`include_background` include the background.  
`squared_pred` use squared versions of targets and predictions in the denominator or not.  
`lambda_dice` the weight of dice.  
`lambda_ce` the weight of ce.
