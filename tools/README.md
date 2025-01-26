$$ DSC = \frac{2|X \cap Y|}{|X| + |Y|} $$

$$ Precision = \frac{TP}{TP + FP} $$

$$ Recall = \frac{TP}{TP + FN} $$ 


### Split_Comb.py
* 3D image to patchs.
* [reference blog](https://blog.csdn.net/qq_39233558/article/details/137139232?fromshare=blogdetail&sharetype=blogdetail&sharerId=137139232&sharerefer=PC&sharesource=qq_39233558&sharefrom=from_link)

### base.py
### image.py
### test.py  
<table>
  <tr>
    <th>Function</th>
    <th>Description</th>
    <th>Formula</th>
  </tr>
  <tr>
    <td>dice_score</td>
    <td>Calculate DSC score.</td>
    <td align="center">$DSC = \frac{2|X \cap Y|}{|X| + |Y|}$</td>
  </tr>
  <tr>
    <td>euclidean_distance</td>
    <td>Calculate euclidean distance.</td>
    <td align="center">$d(A, B)=\sqrt{(x_2 - x_1)^2+(y_2 - y_1)^2+(z_2 - z_1)^2}$</td>
  </tr>
</table>
