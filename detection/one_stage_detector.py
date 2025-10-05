# Credit to Justin Johnsons' EECS-598 course at the University of Michigan,
# from which this assignment is heavily drawn.
import math
from typing import Dict, List, Optional

import torch
from detection_utils import *
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision.ops import sigmoid_focal_loss
from torchvision import models
from torchvision.models import feature_extraction
from torchvision.models.regnet import RegNet_X_400MF_Weights

class DetectorBackboneWithFPN(nn.Module):
    """
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(weights=RegNet_X_400MF_Weights.IMAGENET1K_V1)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        # print("For dummy input images with shape: (2, 3, 224, 224)")
        # for level_name, feature_shape in dummy_out_shapes:
        #     print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.lateral_convs = nn.ModuleDict()
        self.output_convs = nn.ModuleDict()
        
        for level_name, feature_shape in dummy_out_shapes:
            in_channels = feature_shape[1]
            self.lateral_convs[level_name] = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
            self.output_convs[level_name] = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################
        lateral_p5 = self.lateral_convs["c5"](backbone_feats["c5"])
        lateral_p4 = self.lateral_convs["c4"](backbone_feats["c4"]) 
        lateral_p3 = self.lateral_convs["c3"](backbone_feats["c3"])
        # start at deepest level
        fpn_feats["p5"] = lateral_p5
        fpn_feats["p4"] = lateral_p4 + F.interpolate(fpn_feats["p5"], size=lateral_p4.shape[-2:], mode="nearest")
        fpn_feats["p3"] = lateral_p3 + F.interpolate(fpn_feats["p4"], size=lateral_p3.shape[-2:], mode="nearest")
        
        fpn_feats["p3"] = self.output_convs["c3"](fpn_feats["p3"])
        fpn_feats["p4"] = self.output_convs["c4"](fpn_feats["p4"])
        fpn_feats["p5"] = self.output_convs["c5"](fpn_feats["p5"])
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats

class FCOSPredictionNetwork(nn.Module):
    """
    FCOS prediction network that accepts FPN feature maps from different levels
    and makes three predictions at every location: bounding boxes, class ID and
    centerness. This module contains a "stem" of convolution layers, along with
    one final layer per prediction. For a visual depiction, see Figure 2 (right
    side) in FCOS paper: https://arxiv.org/abs/1904.01355

    We will use feature maps from FPN levels (P3, P4, P5) and exclude (P6, P7).
    """

    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        """
        Args:
            num_classes: Number of object classes for classification.
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN, since the head directly
                operates on them.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
        """
        super().__init__()

        ######################################################################
        # TODO: Create a stem of alternating 3x3 convolution layers and RELU
        # activation modules. Note there are two separate stems for class and
        # box stem. The prediction layers for box regression and centerness
        # operate on the output of `stem_box`.
        # See FCOS figure again; both stems are identical.
        #
        # Use `in_channels` and `stem_channels` for creating these layers, the
        # docstring above tells you what they mean. Initialize weights of each
        # conv layer from a normal distribution with mean = 0 and std dev = 0.01
        # and all biases with zero. Use conv stride = 1 and zero padding such
        # that size of input features remains same: remember we need predictions
        # at every location in feature map, we shouldn't "lose" any locations.
        ######################################################################
        # Fill these.
        stem_cls = [] # for classification
        stem_box = [] # for regression and centerness
        # Replace "pass" statement with your code
        curr_channels = in_channels
        for out_channels in stem_channels:
            stem_cls.append(nn.Conv2d(curr_channels, out_channels, kernel_size=3, stride=1, padding=1))
            stem_cls.append(nn.ReLU())
            
            stem_box.append(nn.Conv2d(curr_channels, out_channels, kernel_size=3, stride=1, padding=1))
            stem_box.append(nn.ReLU())
            curr_channels = out_channels


        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        # Initialize all layers.
        for stems in (self.stem_cls, self.stem_box):
            for layer in stems:
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        ######################################################################
        # TODO: Create THREE 3x3 conv layers for individually predicting three
        # things at every location of feature map:
        #     1. object class logits (`num_classes` outputs)
        #     2. box regression deltas (4 outputs: LTRB deltas from locations)
        #     3. centerness logits (1 output)
        ######################################################################
        final_channels = stem_channels[-1]  
        # Replace these lines with your code, keep variable names unchanged.
        self.pred_cls = nn.Conv2d(final_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.pred_box = nn.Conv2d(final_channels, 4, kernel_size=3, stride=1, padding=1)  # Box regression conv
        self.pred_ctr = nn.Conv2d(final_channels, 1, kernel_size=3, stride=1, padding=1)  # Centerness conv

        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # OVERRIDE: Use a negative bias in `pred_cls` to improve training
        # stability. Without this, the training will most likely diverge.
        # STUDENTS: You do not need to get into details of why this is needed.
        if self.pred_cls is not None:
            torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict the desired outputs at every location
        (as described above). Format them such that channels are placed at the
        last dimension, and (H, W) are flattened (having channels at last is
        convenient for computing loss as well as perforning inference).

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}. Each
                tensor will have shape `(batch_size, fpn_channels, H, W)`. For an
                input (224, 224) image, H = W are (28, 14, 7) for (p3, p4, p5).

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Classification logits: `(batch_size, H * W, num_classes)`.
            2. Box regression deltas: `(batch_size, H * W, 4)`
            3. Centerness logits:     `(batch_size, H * W, 1)`
        """

        ######################################################################
        # TODO: Iterate over every FPN feature map and obtain predictions using
        # the layers defined above. Remember that prediction layers of box
        # regression and centerness will operate on output of `stem_box`,
        # and classification layer operates separately on `stem_cls`.
        #
        # CAUTION: The original FCOS model uses shared stem for centerness and
        # classification. Recent follow-up papers commonly place centerness and
        # box regression predictors with a shared stem, which we follow here.
        #
        # DO NOT apply sigmoid to classification and centerness logits.
        ######################################################################
        # Fill these with keys: {"p3", "p4", "p5"}, same as input dictionary.
        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}
        for level_name, features in feats_per_fpn_level.items():
            cls_features = self.stem_cls(features)
            box_features = self.stem_box(features)
            
            cls_pred = self.pred_cls(cls_features)    # (B, num_classes, H, W)
            box_pred = self.pred_box(box_features)    # (B, 4, H, W)
            ctr_pred = self.pred_ctr(box_features)    # (B, 1, H, W)
            
            B, _, H, W = cls_pred.shape
            
            class_logits[level_name] = cls_pred.permute(0, 2, 3, 1).reshape(B, H*W, -1)
            boxreg_deltas[level_name] = box_pred.permute(0, 2, 3, 1).reshape(B, H*W, 4)
            centerness_logits[level_name] = ctr_pred.permute(0, 2, 3, 1).reshape(B, H*W, 1)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return [class_logits, boxreg_deltas, centerness_logits]

class FCOS(nn.Module):
    """
    FCOS: Fully-Convolutional One-Stage Detector

    This class puts together everything you implemented so far. It contains a
    backbone with FPN, and prediction layers (head). It computes loss during
    training and predicts boxes during inference.
    """

    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]
    ):
        super().__init__()
        self.num_classes = num_classes

        ######################################################################
        # TODO: Initialize backbone and prediction network using arguments.  #
        ######################################################################
        # Feel free to delete these two lines: (but keep variable names same)
        self.backbone = DetectorBackboneWithFPN(out_channels=fpn_channels)
        self.pred_net = FCOSPredictionNetwork(num_classes, fpn_channels, stem_channels)
        # Replace "pass" statement with your code
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Averaging factor for training loss; EMA of foreground locations.
        # STUDENTS: See its use in `forward` when you implement losses.
        self._normalizer = 150  # per image

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        Args:
            images: Batch of images, tensors of shape `(B, C, H, W)`.
            gt_boxes: Batch of training boxes, tensors of shape `(B, N, 5)`.
                `gt_boxes[i, j] = (x1, y1, x2, y2, C)` gives information about
                the `j`th object in `images[i]`. The position of the top-left
                corner of the box is `(x1, y1)` and the position of bottom-right
                corner of the box is `(x2, x2)`. These coordinates are
                real-valued in `[H, W]`. `C` is an integer giving the category
                label for this bounding box. Not provided during inference.
            test_score_thresh: During inference, discard predictions with a
                confidence score less than this value. Ignored during training.
            test_nms_thresh: IoU threshold for NMS during inference. Ignored
                during training.

        Returns:
            Losses during training and predictions during inference.
        """

        ######################################################################
        # TODO: Process the image through backbone, FPN, and prediction head #
        # to obtain model predictions at every FPN location.                 #
        # Get dictionaries of keys {"p3", "p4", "p5"} giving predicted class #
        # logits, deltas, and centerness.                                    #
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        backbone_feats = self.backbone(images) #backbone with FPN
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net(backbone_feats) #prediction network

        ######################################################################
        # TODO: Get absolute co-ordinates `(xc, yc)` for every location in
        # FPN levels.
        #
        # HINT: You have already implemented everything, just have to
        # call the functions properly.
        ######################################################################

        shape_per_fpn_level = {level_name: features.shape for level_name, features in backbone_feats.items()}
        
        locations_per_fpn_level = get_fpn_location_coords(
            shape_per_fpn_level, 
            self.backbone.fpn_strides,
            dtype=torch.float32,
            device=images.device
        )

        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        if not self.training:
            # During inference, just go to this method and skip rest of the
            # forward pass.
            # fmt: off
            return self.inference(
                images, locations_per_fpn_level,
                pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
            # fmt: on

        ######################################################################
        # TODO: Assign ground-truth boxes to feature locations. We have this
        # implemented in a `fcos_match_locations_to_gt`. This operation is NOT
        # batched so call it separately per GT boxes in batch.
        ######################################################################
        # List of dictionaries with keys {"p3", "p4", "p5"} giving matched
        # boxes for locations per FPN level, per image. Fill this list:
        matched_gt_boxes = []
        # call for each image in batch
        for i in range(images.shape[0]):
            matched_gt_boxes.append(
                fcos_match_locations_to_gt(
                    locations_per_fpn_level=locations_per_fpn_level,
                    strides_per_fpn_level=self.backbone.fpn_strides,
                    gt_boxes=gt_boxes[i]
                )
            )

        # Calculate GT deltas for these matched boxes. Similar structure
        # as `matched_gt_boxes` above. Fill this list:
        matched_gt_deltas = []
        
        # Process each image in the batch separately
        for idx in range(images.shape[0]):
            # Get matched boxes for this image
            matched_gt_boxes_per_image = matched_gt_boxes[idx]
            
            # Calculate deltas for matched boxes
            matched_gt_deltas_per_image = {}
            for level_name in locations_per_fpn_level.keys():
                stride = self.backbone.fpn_strides[level_name]
                matched_gt_deltas_per_image[level_name] = fcos_get_deltas_from_locations(
                    locations_per_fpn_level[level_name],
                    matched_gt_boxes_per_image[level_name],
                    stride
                )
            matched_gt_deltas.append(matched_gt_deltas_per_image)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Collate lists of dictionaries, to dictionaries of batched tensors.
        # These are dictionaries with keys {"p3", "p4", "p5"} and values as
        # tensors of shape (batch_size, locations_per_fpn_level, 5 or 4)
        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)

        # Combine predictions and GT from across all FPN levels.
        # shape: (batch_size, num_locations_across_fpn_levels, ...)
        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

        # Perform EMA update of normalizer by number of positive locations.
        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image

        ######################################################################
        # TODO: Calculate losses per location for classification, box reg and
        # centerness. Remember to set box/centerness losses for "background"
        # positions to zero.
        ######################################################################
        # Identify foreground locations (class != -1)
        foreground_mask = matched_gt_boxes[:, :, 4] != -1  # Shape: (batch, locations)
        
        # 1. Classification Loss: Use Sigmoid Focal Loss for each class
        gt_classes = matched_gt_boxes[:, :, 4].long()  # Shape: (batch, locations)
        
        # Create binary targets for each class (multi-label classification)
        batch_size, num_locations = gt_classes.shape
        gt_classes_binary = torch.zeros(batch_size, num_locations, self.num_classes, 
                                       device=gt_classes.device, dtype=torch.float32)
        
        # Set positive class to 1 for foreground locations only
        foreground_locations = gt_classes >= 0
        gt_classes_binary[foreground_locations, gt_classes[foreground_locations]] = 1.0
        
        # Use sigmoid focal loss with alpha=0.25, gamma=2.0 (standard FCOS parameters)
        # Reshape for focal loss: (batch * locations, num_classes)
        pred_cls_flat = pred_cls_logits.view(-1, self.num_classes)
        gt_classes_flat = gt_classes_binary.view(-1, self.num_classes)
        
        loss_cls_flat = sigmoid_focal_loss(
            pred_cls_flat,
            gt_classes_flat,
            alpha=0.25,
            gamma=2.0,
            reduction='none'
        )  # Shape: (batch * locations, num_classes)
        
        # Reshape back and sum over classes
        loss_cls = loss_cls_flat.view(batch_size, num_locations, self.num_classes).sum(dim=-1)
        
        # Only apply loss to foreground locations for classification
        loss_cls = loss_cls * foreground_mask.float()
        
        # 2. Box Regression Loss: Use IoU loss for better localization
        # We need to convert deltas back to boxes for IoU computation
        # Only compute IoU loss for foreground locations to avoid unnecessary computation
        
        if foreground_mask.sum() > 0:
            # Extract foreground predictions and targets
            fg_pred_deltas = pred_boxreg_deltas[foreground_mask]  # Shape: (num_fg, 4)
            fg_gt_deltas = matched_gt_deltas[foreground_mask]     # Shape: (num_fg, 4)
            
            # Convert deltas to boxes (l, t, r, b format)
            # Deltas represent distances to box edges from center point
            pred_boxes_ltrb = fg_pred_deltas  # Already in l,t,r,b format
            gt_boxes_ltrb = fg_gt_deltas      # Already in l,t,r,b format
            
            # Convert to x1,y1,x2,y2 format for IoU calculation
            # For a center point at (cx, cy): x1 = cx - l, y1 = cy - t, x2 = cx + r, y2 = cy + b
            # Since we don't have center coordinates here, we'll use the deltas directly
            # assuming they represent the box extents
            pred_x1 = -pred_boxes_ltrb[:, 0]  # -left
            pred_y1 = -pred_boxes_ltrb[:, 1]  # -top  
            pred_x2 = pred_boxes_ltrb[:, 2]   # right
            pred_y2 = pred_boxes_ltrb[:, 3]   # bottom
            
            gt_x1 = -gt_boxes_ltrb[:, 0]
            gt_y1 = -gt_boxes_ltrb[:, 1] 
            gt_x2 = gt_boxes_ltrb[:, 2]
            gt_y2 = gt_boxes_ltrb[:, 3]
            
            # Compute IoU
            pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
            gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
            
            inter_x1 = torch.max(pred_x1, gt_x1)
            inter_y1 = torch.max(pred_y1, gt_y1)
            inter_x2 = torch.min(pred_x2, gt_x2)
            inter_y2 = torch.min(pred_y2, gt_y2)
            
            inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
            
            union_area = pred_area + gt_area - inter_area
            iou = inter_area / (union_area + 1e-6)  # Add small epsilon for numerical stability
            
            # IoU loss: 1 - IoU
            fg_box_loss = 1.0 - iou
            
            # Create full loss tensor with zeros for background
            loss_box = torch.zeros_like(foreground_mask, dtype=torch.float32)
            loss_box[foreground_mask] = fg_box_loss
        else:
            # No foreground locations
            loss_box = torch.zeros_like(foreground_mask, dtype=torch.float32)
        
        # 3. Centerness Loss: Only for foreground locations
        gt_centerness = fcos_make_centerness_targets(matched_gt_deltas.view(-1, 4))
        gt_centerness = gt_centerness.view(matched_gt_deltas.shape[0], matched_gt_deltas.shape[1])
        
        # For more stable training, only compute loss on valid centerness targets
        # Background locations have gt_centerness = -1, which we should exclude
        valid_centerness_mask = (gt_centerness >= 0) & foreground_mask
        
        if valid_centerness_mask.sum() > 0:
            # Only compute loss where we have valid centerness targets
            valid_pred_ctr = pred_ctr_logits.squeeze(-1)[valid_centerness_mask]
            valid_gt_ctr = gt_centerness[valid_centerness_mask]
            
            # Clamp centerness targets to valid range [0, 1] for numerical stability
            valid_gt_ctr = torch.clamp(valid_gt_ctr, 0.0, 1.0)
            
            # Use BCE with logits but only on valid locations
            valid_loss_ctr = F.binary_cross_entropy_with_logits(
                valid_pred_ctr, valid_gt_ctr, reduction='none'
            )
            
            # Create full loss tensor with zeros for invalid locations
            loss_ctr = torch.zeros_like(foreground_mask, dtype=torch.float32)
            loss_ctr[valid_centerness_mask] = valid_loss_ctr
        else:
            # No valid centerness targets
            loss_ctr = torch.zeros_like(foreground_mask, dtype=torch.float32)

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################
        # Sum all locations and average by the EMA of foreground locations.
        # In training code, we simply add these three and call `.backward()`
        return {
            "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
            "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
            "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.

                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).

                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions: these values are `sqrt(class_prob * ctrness)`
                  where class_prob and ctrness are obtained by applying sigmoid
                  to corresponding logits.
        """

        # Gather scores and boxes from all FPN levels in this list. Once
        # gathered, we will perform NMS to filter highly overlapping predictions.
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():

            # Get locations and predictions from a single level.
            # We index predictions by `[0]` to remove batch dimension.
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]

            ##################################################################
            # TODO: FCOS uses the geometric mean of class probability and
            # centerness as the final confidence score. This helps in getting
            # rid of excessive amount of boxes far away from object centers.
            # Compute this value here (recall sigmoid(logits) = probabilities)
            #
            # Then perform the following steps in order:
            #   1. Get the most confidently predicted class and its score for
            #      every box. Use level_pred_scores: (N, num_classes) => (N, )
            #   2. Only retain prediction that have a confidence score higher
            #      than provided threshold in arguments.
            #   3. Obtain predicted boxes using predicted deltas and locations
            #   4. Clip XYXY box-cordinates that go beyond the height and
            #      and width of input image.
            ##################################################################
            # Feel free to delete this line: (but keep variable names same)
            level_pred_boxes, level_pred_classes, level_pred_scores = (
                None,
                None,
                None,  # Need tensors of shape: (N, 4) (N, ) (N, )
            )

            # Compute geometric mean of class logits and centerness:
            level_pred_scores = torch.sqrt(
                level_cls_logits.sigmoid() * level_ctr_logits.sigmoid()
            )
            
            # Step 1: Get the most confidently predicted class and its score
            level_pred_scores, level_pred_classes = level_pred_scores.max(dim=1)
            
            # Step 2: 
            keep_mask = level_pred_scores > test_score_thresh
            level_pred_scores = level_pred_scores[keep_mask]
            level_pred_classes = level_pred_classes[keep_mask]

            # Step 3: 
            level_deltas = level_deltas[keep_mask]
            level_locations = level_locations[keep_mask]
            
            stride = self.backbone.fpn_strides[level_name]
            
            level_pred_boxes = fcos_apply_deltas_to_locations(
                level_deltas, level_locations, stride
            )

            # Step 4: 
            H, W = images.shape[2:4]  # Get image height and width
            level_pred_boxes[:, [0, 2]] = torch.clamp(level_pred_boxes[:, [0, 2]], min=0, max=W)
            level_pred_boxes[:, [1, 3]] = torch.clamp(level_pred_boxes[:, [1, 3]], min=0, max=H)

            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################

            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)

        ######################################################################
        # Combine predictions from all levels and perform NMS.
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        keep = class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]
        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )
