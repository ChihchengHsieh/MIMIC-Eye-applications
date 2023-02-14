# the task performers here, take a 3-d tensor, then return the output of its task.and
#

from collections import OrderedDict
from typing import List
import torch
from torch import nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNNHeads
from torchvision.models.detection.faster_rcnn import (
    MultiScaleRoIAlign,
    RPNHead,
    FastRCNNPredictor,
    AnchorGenerator,
)

from torchvision.models.detection.transform import resize_boxes, resize_keypoints
from torchvision.models.detection.roi_heads import paste_masks_in_image

from models.unet import UNetDecoder
from .rcnn import (
    EyeRCNNTransform,
    XAMIRegionProposalNetwork,
    XAMIRoIHeads,
    XAMITwoMLPHead,
)


class GeneralTaskPerformer(nn.Module):
    def __init__(self, name: str, loses: List[str]) -> None:
        self.name = name
        self.loses = loses
        super().__init__()

    def forward(self, x, z, targets):
        pass


class ObjectDetectionParameters(object):
    def __init__(self, image_size=512) -> None:
        self.image_size = image_size
        # rpn params
        self.rpn_pre_nms_top_n_train = 2000
        self.rpn_pre_nms_top_n_test = 1000
        self.rpn_post_nms_top_n_train = 2000
        self.rpn_post_nms_top_n_test = 1000
        self.rpn_nms_thresh = 0.7
        self.rpn_fg_iou_thresh = 0.7
        self.rpn_bg_iou_thresh = 0.3
        self.rpn_batch_size_per_image = 256
        self.rpn_positive_fraction = 0.5
        self.rpn_score_thresh = 0.0

        # box predictor params
        self.box_roi_pool = None
        self.box_head = None
        self.box_predictor = None
        self.box_score_thresh = 0.05
        self.box_nms_thresh = 0.5
        self.box_detections_per_img = 100
        self.box_fg_iou_thresh = 0.5
        self.box_bg_iou_thresh = 0.5
        self.box_batch_size_per_image = 512
        self.box_positive_fraction = 0.25
        self.bbox_reg_weights = None

        # self.anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        # self.aspect_ratios = ((0.5, 1.0, 2.0),) * len(self.anchor_sizes)

        self.anchor_sizes = ((32, 64, 128, 256, 512),)
        self.aspect_ratios = ((0.5, 1.0, 2.0),)
        # older version that doesn't consider the len of anchor sizes
        # anchor_generator = AnchorGenerator(
        #     sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)
        # )

        # mlp head params
        self.XAMITwoMLPHead_representation_size = 1024
        self.XAMITwoMLPHead_dropout_rate = 0

        # transform params
        self.transform_min_size = (800,)
        self.transform_max_size = 1333



class ObjectDetectionPerformer(GeneralTaskPerformer):
    def __init__(
        self, params: ObjectDetectionParameters, out_channels, num_classes
    ) -> None:
        super().__init__(
            name="performer-object_detection",
            loses=[
                "loss_box_reg",
                "loss_classifier",
                "loss_mask",
                "loss_objectness",
                "loss_rpn_box_reg",
            ],
        )

        self.out_channels = out_channels
        self.num_classes = num_classes
        self.params = params

        self.__init_rpn(self.params)
        self.__init_roi_heads(self.params)
        self.__init_transform(self.params)

    def forward(self, x, z, targets):

        if isinstance(z, torch.Tensor):
            z = OrderedDict([("0", z)])

        proposals, proposal_losses = self.rpn(x["image_list"], z, targets)

        detections, detector_losses = self.roi_heads(
            z, proposals, x["image_list"].image_sizes, targets,
        )

        detections = self.postprocess(
            detections, x["image_list"].image_sizes, x["original_image_sizes"]
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return {
            "losses": losses,
            "outputs": detections,
        }

    def postprocess(
        self, result, image_shapes, original_image_sizes,
    ):
        # if self.training:
        #     return result
        for i, (pred, im_s, o_im_s) in enumerate(
            zip(result, image_shapes, original_image_sizes)
        ):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
        return result

    def __init_rpn(self, params):

        rpn_anchor_generator = AnchorGenerator(
            params.anchor_sizes, params.aspect_ratios
        )

        rpn_head = RPNHead(
            self.out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )

        rpn_pre_nms_top_n = dict(
            training=params.rpn_pre_nms_top_n_train,
            testing=params.rpn_pre_nms_top_n_test,
        )
        rpn_post_nms_top_n = dict(
            training=params.rpn_post_nms_top_n_train,
            testing=params.rpn_post_nms_top_n_test,
        )

        self.rpn = XAMIRegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            params.rpn_fg_iou_thresh,
            params.rpn_bg_iou_thresh,
            params.rpn_batch_size_per_image,
            params.rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            params.rpn_nms_thresh,
            score_thresh=params.rpn_score_thresh,
        )

    def __init_roi_heads(self, params: ObjectDetectionParameters):

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2
        )

        # if box_roi_pool is None:
        #     box_roi_pool = MultiScaleRoIAlign(
        #         featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        #     )

        resolution = box_roi_pool.output_size[0]
        box_head = XAMITwoMLPHead(
            self.out_channels * resolution ** 2,
            params.XAMITwoMLPHead_representation_size,
            dropout_rate=params.XAMITwoMLPHead_dropout_rate,
        )

        box_predictor = FastRCNNPredictor(
            params.XAMITwoMLPHead_representation_size, self.num_classes
        )

        self.roi_heads = XAMIRoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            params.box_fg_iou_thresh,
            params.box_bg_iou_thresh,
            params.box_batch_size_per_image,
            params.box_positive_fraction,
            params.bbox_reg_weights,
            params.box_score_thresh,
            params.box_nms_thresh,
            params.box_detections_per_img,
        )

    def __init_transform(self, params: ObjectDetectionParameters):

        # should we update this one?
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        self.transform = EyeRCNNTransform(
            params.transform_min_size,
            params.transform_max_size,
            image_mean,
            image_std,
            fixed_size=[params.image_size, params.image_size],
        )


    def valid_bbox(self, targets):
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        " Found invalid box {} for target at index {}.".format(
                            degen_bb, target_idx
                        )
                    )


class ObjectDetectionWithMaskParameters(object):
    def __init__(self, image_size=512) -> None:
        self.image_size = image_size
        # rpn params
        self.rpn_pre_nms_top_n_train = 2000
        self.rpn_pre_nms_top_n_test = 1000
        self.rpn_post_nms_top_n_train = 2000
        self.rpn_post_nms_top_n_test = 1000
        self.rpn_nms_thresh = 0.7
        self.rpn_fg_iou_thresh = 0.7
        self.rpn_bg_iou_thresh = 0.3
        self.rpn_batch_size_per_image = 256
        self.rpn_positive_fraction = 0.5
        self.rpn_score_thresh = 0.0

        # box predictor params
        self.box_roi_pool = None
        self.box_head = None
        self.box_predictor = None
        self.box_score_thresh = 0.05
        self.box_nms_thresh = 0.5
        self.box_detections_per_img = 100
        self.box_fg_iou_thresh = 0.5
        self.box_bg_iou_thresh = 0.5
        self.box_batch_size_per_image = 512
        self.box_positive_fraction = 0.25
        self.bbox_reg_weights = None

        # self.anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        # self.aspect_ratios = ((0.5, 1.0, 2.0),) * len(self.anchor_sizes)

        self.anchor_sizes = ((32, 64, 128, 256, 512),)
        self.aspect_ratios = ((0.5, 1.0, 2.0),)
        # older version that doesn't consider the len of anchor sizes
        # anchor_generator = AnchorGenerator(
        #     sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)
        # )

        # mlp head params
        self.XAMITwoMLPHead_representation_size = 1024
        self.XAMITwoMLPHead_dropout_rate = 0

        # transform params
        self.transform_min_size = (800,)
        self.transform_max_size = 1333

        # mask head params

        self.mask_layers = (256, 256, 256, 256)
        self.mask_dilation = 1
        self.mask_predictor_in_channels = 256  # == mask_layers[-1]
        self.mask_dim_reduced = 256


class ObjectDetectionWithMaskPerformer(GeneralTaskPerformer):
    def __init__(
        self, params: ObjectDetectionWithMaskParameters, out_channels, num_classes
    ) -> None:
        super().__init__(
            name="performer-object_detection",
            loses=[
                "loss_box_reg",
                "loss_classifier",
                "loss_mask",
                "loss_objectness",
                "loss_rpn_box_reg",
            ],
        )

        self.out_channels = out_channels
        self.num_classes = num_classes
        self.params = params

        self.__init_rpn(self.params)
        self.__init_roi_heads(self.params)
        self.__init_transform(self.params)
        self.__init_mask_heads(self.params)

    def forward(self, x, z, targets):

        if isinstance(z, torch.Tensor):
            z = OrderedDict([("0", z)])

        proposals, proposal_losses = self.rpn(x["image_list"], z, targets)

        detections, detector_losses = self.roi_heads(
            z, proposals, x["image_list"].image_sizes, targets,
        )

        detections = self.postprocess(
            detections, x["image_list"].image_sizes, x["original_image_sizes"]
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return {
            "losses": losses,
            "outputs": detections,
        }

    def postprocess(
        self, result, image_shapes, original_image_sizes,
    ):
        # if self.training:
        #     return result
        for i, (pred, im_s, o_im_s) in enumerate(
            zip(result, image_shapes, original_image_sizes)
        ):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result

    def __init_rpn(self, params):

        rpn_anchor_generator = AnchorGenerator(
            params.anchor_sizes, params.aspect_ratios
        )

        rpn_head = RPNHead(
            self.out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )

        rpn_pre_nms_top_n = dict(
            training=params.rpn_pre_nms_top_n_train,
            testing=params.rpn_pre_nms_top_n_test,
        )
        rpn_post_nms_top_n = dict(
            training=params.rpn_post_nms_top_n_train,
            testing=params.rpn_post_nms_top_n_test,
        )

        self.rpn = XAMIRegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            params.rpn_fg_iou_thresh,
            params.rpn_bg_iou_thresh,
            params.rpn_batch_size_per_image,
            params.rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            params.rpn_nms_thresh,
            score_thresh=params.rpn_score_thresh,
        )

    def __init_roi_heads(self, params: ObjectDetectionWithMaskParameters):

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2
        )

        # if box_roi_pool is None:
        #     box_roi_pool = MultiScaleRoIAlign(
        #         featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        #     )

        resolution = box_roi_pool.output_size[0]
        box_head = XAMITwoMLPHead(
            self.out_channels * resolution ** 2,
            params.XAMITwoMLPHead_representation_size,
            dropout_rate=params.XAMITwoMLPHead_dropout_rate,
        )

        box_predictor = FastRCNNPredictor(
            params.XAMITwoMLPHead_representation_size, self.num_classes
        )

        self.roi_heads = XAMIRoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            params.box_fg_iou_thresh,
            params.box_bg_iou_thresh,
            params.box_batch_size_per_image,
            params.box_positive_fraction,
            params.bbox_reg_weights,
            params.box_score_thresh,
            params.box_nms_thresh,
            params.box_detections_per_img,
        )

    def __init_transform(self, params: ObjectDetectionWithMaskParameters):

        # should we update this one?
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        self.transform = EyeRCNNTransform(
            params.transform_min_size,
            params.transform_max_size,
            image_mean,
            image_std,
            fixed_size=[params.image_size, params.image_size],
        )

    def __init_mask_heads(self, params: ObjectDetectionWithMaskParameters):

        # extra for the mask-rcnn
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
        )

        mask_head = MaskRCNNHeads(
            self.out_channels, params.mask_layers, params.mask_dilation
        )

        mask_predictor = MaskRCNNPredictor(
            params.mask_predictor_in_channels, params.mask_dim_reduced, self.num_classes
        )

        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor

    def valid_bbox(self, targets):
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        " Found invalid box {} for target at index {}.".format(
                            degen_bb, target_idx
                        )
                    )


class HeatmapGeneratorParameters(object):
    def __init__(self, input_channel, decoder_channels) -> None:
        super().__init__()

        self.input_channel = input_channel
        self.decoder_channels = decoder_channels


class HeatmapGenerator(GeneralTaskPerformer):

    """
    Expecting targets -> {
        heatmap: tensor        
    }
    """

    def __init__(self, params: HeatmapGeneratorParameters) -> None:
        super().__init__(name="performer-heatmap_generator", loses=["heatmap_loss"])
        self.model = UNetDecoder(params.input_channel, params.decoder_channels)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x, z, targets):
        output = self.model(z)

        loss = self.loss_fn(
            output, torch.stack([t["fixations"] for t in targets], dim=0).float()
        )

        return {
            "losses": {"heatmap_loss": loss},
            "outputs": output,
        }
