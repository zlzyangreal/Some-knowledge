# 1. 增加代码在loss.py
## 1.1 顶部加大类
```python
def bbox_iou_wiseiou(box1, box2, xywh=True, eps=1e-7):
    """
    Calculate IoU and additional metrics needed for Wise-IoU.
    
    Args:
        box1: Predicted boxes (N, 4) in xyxy or xywh format
        box2: Ground truth boxes (N, 4) in xyxy or xywh format
        xywh: If True, boxes are in xywh format, else xyxy
        eps: Small value to avoid division by zero
        
    Returns:
        iou: IoU values (N,)
        distance: Center point distance (N,)
        diagonal: Diagonal length of minimum enclosing box (N,)
    """
    if xywh:
        # Convert from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2 / 2, x2 + w2 / 2, y2 - h2 / 2, y2 + h2 / 2
    else:
        # Already in xyxy format
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    # Center point distance
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    
    # Center points
    c1_x = (b1_x1 + b1_x2) / 2
    c1_y = (b1_y1 + b1_y2) / 2
    c2_x = (b2_x1 + b2_x2) / 2
    c2_y = (b2_y1 + b2_y2) / 2
    
    # Distance between centers
    distance = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2
    
    # Diagonal of minimum enclosing box
    diagonal = cw ** 2 + ch ** 2 + eps
    
    return iou.squeeze(-1), distance.squeeze(-1), diagonal.squeeze(-1)


class WiseIoULoss(nn.Module):
    """
    Wise-IoU Loss with dynamic non-monotonic focusing mechanism.
    
    The loss consists of three versions:
    - WIoUv1: Basic Wise-IoU with distance attention
    - WIoUv2: Adds IoU-based focusing mechanism  
    - WIoUv3: Dynamic non-monotonic focusing (recommended)
    
    Args:
        version (str): Version of Wise-IoU ('v1', 'v2', or 'v3')
        alpha (float): Power factor for distance attention (default: 2.0)
        beta (float): Scaling factor for outlier degree (default: 2.0)
        monotonic (bool): Use monotonic focusing if True (WIoUv2), else non-monotonic (WIoUv3)
        eps (float): Small value to avoid division by zero
    """
    
    def __init__(self, version='v3', alpha=2.0, beta=2.0, monotonic=False, eps=1e-7):
        super().__init__()
        self.version = version.lower()
        self.alpha = alpha
        self.beta = beta
        self.monotonic = monotonic
        self.eps = eps
        
        # For dynamic beta computation (v3)
        self.register_buffer('_iou_mean', torch.tensor(0.0))
        self.register_buffer('_update_count', torch.tensor(0))
        
    def forward(self, pred_boxes, target_boxes, xywh=False):
        """
        Calculate Wise-IoU loss.
        
        Args:
            pred_boxes: Predicted boxes (N, 4)
            target_boxes: Ground truth boxes (N, 4)
            xywh: If True, boxes are in xywh format, else xyxy
            
        Returns:
            loss: Wise-IoU loss value
        """
        # Calculate IoU and geometric metrics
        iou, distance, diagonal = bbox_iou_wiseiou(pred_boxes, target_boxes, xywh=xywh, eps=self.eps)
        
        # Distance ratio (normalized distance)
        r = distance / diagonal
        
        if self.version == 'v1':
            # WIoUv1: Distance attention only
            w = torch.exp(-r.detach() ** self.alpha)
            loss = (1 - iou) * w
            
        elif self.version == 'v2' or (self.version == 'v3' and self.monotonic):
            # WIoUv2: Monotonic focusing
            w = self._compute_distance_attention(r)
            loss = (1 - iou) * w
            
        else:  # v3 with non-monotonic
            # WIoUv3: Dynamic non-monotonic focusing (recommended)
            w = self._compute_distance_attention(r)
            
            # Dynamic focusing mechanism
            if self.training:
                # Update running mean of IoU
                with torch.no_grad():
                    batch_iou_mean = iou.detach().mean()
                    self._update_count += 1
                    momentum = 0.9 if self._update_count > 1 else 0.0
                    self._iou_mean = momentum * self._iou_mean + (1 - momentum) * batch_iou_mean
            
            # Compute outlier degree
            beta_dynamic = self._compute_outlier_degree(iou)
            
            # Apply non-monotonic focusing
            loss = (1 - iou) ** beta_dynamic * w
        
        return loss.mean()
    
    def _compute_distance_attention(self, r):
        """
        Compute distance attention weight.
        
        Args:
            r: Distance ratio (N,)
            
        Returns:
            w: Attention weight (N,)
        """
        return torch.exp(-r.detach() ** self.alpha)
    
    def _compute_outlier_degree(self, iou):
        """
        Compute dynamic outlier degree for focusing.
        
        Args:
            iou: IoU values (N,)
            
        Returns:
            beta: Dynamic beta values (N,)
        """
        # Use running mean IoU as anchor
        iou_mean = self._iou_mean if self.training else iou.detach().mean()
        
        # Compute outlier degree
        # High values for samples far from mean (outliers)
        outlier = torch.abs(iou.detach() - iou_mean)
        
        # Scale to appropriate range using beta hyperparameter
        beta = self.beta * outlier / (iou_mean + self.eps)
        
        # Ensure beta is in reasonable range
        beta = beta.clamp(min=0.5, max=3.0)
        
        return beta

```
## 1.2 在`BboxLoss`下加小类
```python
class WiseBboxLoss(nn.Module):
    """
    Bounding box loss using Wise-IoU for YOLO26.
    Replaces standard IoU loss with Wise-IoU.
    """
    
    def __init__(self, reg_max=16, wiseiou_version='v3', alpha=2.0, beta=2.0):
        super().__init__()
        # ✅ 直接使用 DFLoss，因为现在它已经在上面定义过了
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.wiseiou_loss = WiseIoULoss(version=wiseiou_version, alpha=alpha, beta=beta)
    
    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Wise-IoU and DFL losses for bounding boxes.
        
        Args:
            pred_dist: Predicted distributions (B, N, 4*reg_max)
            pred_bboxes: Predicted boxes in xyxy format (B, N, 4)
            anchor_points: Anchor points (N, 2)
            target_bboxes: Target boxes in xyxy format (B, N, 4)
            target_scores: Target scores (B, N, C)
            target_scores_sum: Sum of target scores
            fg_mask: Foreground mask (B, N)
            imgsz: Image size tensor (2,) [H, W]
            stride: Stride tensor (N, 1)
            
        Returns:
            loss_iou: Wise-IoU loss
            loss_dfl: DFL loss
        """
        from .tal import bbox2dist
        
        # Apply foreground mask and compute weights
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # Compute Wise-IoU loss on foreground samples
        pred_bboxes_fg = pred_bboxes[fg_mask]
        target_bboxes_fg = target_bboxes[fg_mask]
        
        if len(pred_bboxes_fg) > 0:
            # Calculate Wise-IoU for each foreground sample
            wiseiou_loss = self.wiseiou_loss(pred_bboxes_fg, target_bboxes_fg, xywh=False)
            # Apply weights
            loss_iou = (wiseiou_loss.view(-1, 1) * weight).sum() / target_scores_sum
        else:
            loss_iou = torch.tensor(0.0, device=pred_bboxes.device)
        
        # DFL loss (Distribution Focal Loss)
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # Normalize ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        
        return loss_iou, loss_dfl


# Example usage for integrating into YOLO26
# Replace the BboxLoss initialization in v8DetectionLoss.__init__ with:
#
# self.bbox_loss = WiseBboxLoss(
#     m.reg_max, 
#     wiseiou_version='v3',  # or 'v1', 'v2'
#     alpha=2.0,
#     beta=2.0
# ).to(device)


# Configuration example
WISEIOU_CONFIGS = {
    'v1': {'version': 'v1', 'alpha': 2.0, 'beta': 2.0},  # Distance attention only
    'v2': {'version': 'v2', 'alpha': 2.0, 'beta': 2.0},  # Monotonic focusing
    'v3': {'version': 'v3', 'alpha': 2.0, 'beta': 2.0},  # Non-monotonic (recommended)
    'v3_aggressive': {'version': 'v3', 'alpha': 3.0, 'beta': 3.0},  # Stronger focusing
}

```
## 1.3 修改原文件的代码
```python
        # self.assigner = TaskAlignedAssigner(
        #     topk=tal_topk,
        #     num_classes=self.nc,
        #     alpha=0.5,
        #     beta=6.0,
        #     stride=self.stride.tolist(),
        #     topk2=tal_topk2,
        # )
        # self.bbox_loss = BboxLoss(m.reg_max).to(device)
        # self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )

        # --- 新增逻辑：读取配置参数来决定使用哪个 Loss ---
        # getattr(h, '参数名', 默认值)
        # 如果我们在训练命令中没传 iou_type，它就默认为 'default' (使用原版 BboxLoss)
        iou_type = getattr(h, 'iou_type', 'default') 

        if iou_type == 'wise':
            print("🚀 Using Wise-IoU Loss!")
            self.bbox_loss = WiseBboxLoss(
                reg_max=m.reg_max,
                wiseiou_version=getattr(h, 'wise_version', 'v3'), # 支持从参数读取 v1/v2/v3
                alpha=1.2,
                beta=6.0
            ).to(device)
        else:
            # 默认情况，使用原版 Loss
            self.bbox_loss = BboxLoss(m.reg_max).to(device)
        # -----------------------------------------------

        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
```
# 2. 添加声明
/home/alphafox/Desktop/Code/Yolo26/ultralytics/ultralytics/cfg/default.yaml
最后一排加入
```yaml
# Custom Arguments
iou_type: default  # 默认为 default, 可选: wise
wise_version: v3   # 默认为 v3, 可选: v1, v2
```
# 3. 启动函数
```python
def train_wise_iou_yolo():
    print("=" * 60)
    print("Training YOLO26n with Wise-IoU Loss (v3)")
    print("=" * 60)
    
    model = YOLO("/home/alphafox/Desktop/Code/Yolo26/ultralytics/ultralytics/cfg/models/26/yolo26.yaml")
    
    results = model.train(
        task="detect",
        data="/home/alphafox/Desktop/Code/Yolo26/ultralytics/Data/xiaoliao.yaml",
        epochs=200,
        imgsz=640,
        batch=4,
        name="yolo26_wiseiou_xiaoliao", 
        
        exist_ok=True,
        device=0,
        workers=8,
        patience=0,
        
        iou_type="wise",      # 这会触发 if iou_type == 'wise':
        wise_version="v3",    # (可选) 如果你在 loss.py 里也写了读取这个参数
    )
    
    return results
```