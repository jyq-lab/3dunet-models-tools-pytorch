### Split_Comb.py
* 3D image to patchs. [reference blog](https://blog.csdn.net/qq_39233558/article/details/137139232?fromshare=blogdetail&sharetype=blogdetail&sharerId=137139232&sharerefer=PC&sharesource=qq_39233558&sharefrom=from_link)

### base.py
### image.py
| Function | Description |
|----------|-------------|
|dicom_to_nii_gz||
|draw_msra_gaussian_3d||
|crop_img_from_keypoint||
|get_max_pred||
|compute_bounding_box||
|bb_iou_3d||

### test.py  
<table>
  <tr>
    <th>Function</th>
    <th>Description</th>
    <th>Formula</th>
  </tr>
  <tr>
    <td>dice_score</td>
    <td>Calculate the DSC score by regions.</td>
    <td align="center">$DSC = \frac{2|X \cap Y|}{|X| + |Y|}$</td>
  </tr>
  <tr>
    <td>euclidean_distance</td>
    <td>Calculate the Euclidean distance between two 3D points.</td>
    <td align="center">$d(A, B)=\sqrt{(x_2 - x_1)^2+(y_2 - y_1)^2+(z_2 - z_1)^2}$</td>
  </tr>
  <tr>
    <td>calculate</td>
    <td>Calculate accuracy, precision, specificity, recall, iou and dice based on voxels.</td>
    <td align="center">
      $Accuracy=\frac{TP + TN}{TP + TN + FP + FN}$</br>
      $Precision = \frac{TP}{TP + FP}$</br>
      $Specificity=\frac{TN}{TN + FP}$</br>
      $Recall=\frac{TP}{TP + FN}$</br>
      $IoU=\frac{TP}{TP + FP + FN}$</br>
      $DSC=\frac{2TP}{2TP + FP + FN}$
    </td>
  </tr>
  <tr>
    <td>hausdorff_distance</td>
    <td>Calculate hausdorff distance.</td>
    <td align="center">$\text{HD}(A, B) = \max\left\{ \max_{a \in A} \min_{b \in B} d(a, b), \max_{b \in B} \min_{a \in A} d(a, b) \right\}$</td>
  </tr>
</table>
