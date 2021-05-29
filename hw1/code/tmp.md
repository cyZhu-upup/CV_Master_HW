最后定义图片中后半部分，即FCOS Head

```python
self.head = layers.PointHead(cfg, feature_shapes)  # 用于分类和回归，外加一个centerness打分，在layers/point_head.py中定义
```

**point_head.py**中，两个通道各4个卷积block

```python
self.cls_subnet = M.Sequential(*cls_subnet)
self.bbox_subnet = M.Sequential(*bbox_subnet)
```

再定义3个输出，分别是分类Score，bounding box回归 ，作者新定义的center-ness，作用是纠正point回到gt box的中心

```python
self.cls_score = M.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
self.bbox_pred = M.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
self.ctrness = M.Conv2d(in_channels, num_anchors * 1, kernel_size=3, stride=1, padding=1)
```

定义这里有点问题：ctrness分支应该是和分类分支平行的，看代码里变成和回归分支一起了，不知道是不是这样操作更好，反正和paper中的定义不同

```python
def forward(self, features: List[Tensor]):
		logits, offsets, ctrness = [], [], []
    for feature, scale, stride in zip(features, self.scale_list, self.stride_list):
        logits.append(self.cls_score(self.cls_subnet(feature)))  # 分类score
        bbox_subnet = self.bbox_subnet(feature)
        offsets.append(F.relu(self.bbox_pred(bbox_subnet) * scale) * stride)  
        ctrness.append(self.ctrness(bbox_subnet))  # 看论文这应该是在cls_subnet后面
    return logits, offsets, ctrness
```

ctrness的定义：

$centerness=\sqrt{\frac{min(l^*,r^*)}{max(l^*,r^*)}\times \frac{min(t^*,b^*)}{max(t^*,b^*)}}$

```python
ctrness = F.sqrt(
  F.maximum(F.min(left_right, axis=1) / F.max(left_right, axis=1), 0)
  * F.maximum(F.min(top_bottom, axis=1) / F.max(top_bottom, axis=1), 0)
)
```



AP=0.236,AP50=0.394,AP75=0.245



ATSS:

AP=0.240, AP50=0.390,AP75=0.251	