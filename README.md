The code includes training and inference procedures for # CDNet: Centripetal Direction Network for Nuclear Instance Segmentation(https://openaccess.thecvf.com/content/ICCV2021/papers/He_CDNet_Centripetal_Direction_Network_for_Nuclear_Instance_Segmentation_ICCV_2021_paper.pdf).


There is a result collation error(U-Net) in Table 4 in the original paper. 
The correct result is：


#### ADE20K Panoptic Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method Name</th>
<th valign="bottom">Dice</th>
<th valign="bottom">AJI</th>
<!-- TABLE BODY -->
<!-- U-Net -->
 <tr><td align="left">U-Net</a></td>
<td align="center">0.8184</td>
<td align="center">0.5910</td>
</tr>
<!-- Mask-RCNN -->
 <tr><td align="left">Mask-RCNN</a></td>
<td align="center">0.7600</td>
<td align="center">0.5460</td>
</tr>
<!-- DCAN -->
 <tr><td align="left">DCAN</a></td>
<td align="center">0.7920</td>
<td align="center">0.5250</td>
</tr>
<!-- Micro-Net -->
 <tr><td align="left">Micro-Net</a></td>
<td align="center">0.7970</td>
<td align="center">0.5600</td>
</tr>
<!-- DIST -->
 <tr><td align="left">DIST</a></td>
<td align="center">0.7890</td>
<td align="center">0.5590</td>
</tr>
<!-- CIA-Net -->
 <tr><td align="left">CIA-Net</a></td>
<td align="center">0.8180</td>
<td align="center">0.6200</td>
</tr>
<!-- FullNet -->
 <tr><td align="left">U-Net</a></td>
<td align="center">0.8027</td>
<td align="center">0.6039</td>
</tr>
<!-- Hover-Net -->
 <tr><td align="left">Hover-Net</a></td>
<td align="center">0.8260</td>
<td align="center">0.6180</td>
</tr>
</tbody></table>
<!-- BRP-Net -->
 <tr><td align="left">BRP-Net</a></td>
<td align="center"> - </td>
<td align="center">0.6422</td>
</tr>
</tbody></table>
<!-- PFF-Net -->
 <tr><td align="left">PFF-Net</a></td>
<td align="center">0.8091</td>
<td align="center">0.6107</td>
</tr>
</tbody></table>
<!-- Our CDNet -->
 <tr><td align="left">Our CDNet</a></td>
<td align="center">0.8316</td>
<td align="center">0.6331</td>
</tr>
</tbody></table>


Detailed operation procedures will be released in the near future.


