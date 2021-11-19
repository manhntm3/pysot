# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
import torch.onnx as tonnx
import onnx

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.core.xcorr import xcorr_depthwise

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):
        zf = self.backbone(z)
        zzf = zf
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        import pdb; pdb.set_trace()
        self.zf = zf


    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }

    def convert_rpn_head(self, net, input0, input1):
        cls, loc = net(input0, input1)
        # for out in output:
        #     print(out.shape)
        for i in range(3):

            rpnIdx = str(i + 2)

            # Classificationn score: 
            net_i_cls = getattr(getattr(net, 'rpn' + rpnIdx), "cls")
            
            # Build three models:
            net_i_cls_conv_kernel = getattr(net_i_cls, "conv_kernel")
            net_i_cls_conv_search = getattr(net_i_cls, "conv_search")
            # net_i_cls_xcorr_depthwise = getattr(net_i_cls, "")
            net_i_cls_head = getattr(net_i_cls, "head")

            kernel, search = input0[i], input1[i]
            ONNX_FILE_PATH = "siameserpn_mobilev2_rpn_head_"+ str(i) + "_cls_conv_kernel.onnx"
            tonnx.export(net_i_cls_conv_kernel, args=(kernel), f=ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True)
            onnx_model = onnx.load(ONNX_FILE_PATH)
            onnx.checker.check_model(onnx_model)

            ONNX_FILE_PATH = "siameserpn_mobilev2_rpn_head_"+ str(i) + "_cls_conv_search.onnx"
            tonnx.export(net_i_cls_conv_search, args=(search), f=ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True)
            onnx_model = onnx.load(ONNX_FILE_PATH)
            onnx.checker.check_model(onnx_model)

            kernel = net_i_cls_conv_kernel(kernel)
            search = net_i_cls_conv_search(search)
            feature = xcorr_depthwise(search, kernel)

            ONNX_FILE_PATH = "siameserpn_mobilev2_rpn_head_"+ str(i) + "_cls_head.onnx"
            tonnx.export(net_i_cls_head, args=(feature), f=ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True)
            onnx_model = onnx.load(ONNX_FILE_PATH)
            onnx.checker.check_model(onnx_model)

            # out = net_i_cls_head(feature)

            # Location score: 
            net_i_loc = getattr(getattr(net, 'rpn' + rpnIdx), "loc")
            
            net_i_loc_conv_kernel = getattr(net_i_loc, "conv_kernel")
            net_i_loc_conv_search = getattr(net_i_loc, "conv_search")
            # net_i_loc_xcorr_depthwise = getattr(net_i_loc, "")
            net_i_loc_head = getattr(net_i_loc, "head")

            kernel, search = input0[i], input1[i]
            ONNX_FILE_PATH = "siameserpn_mobilev2_rpn_head_"+ str(i) + "_loc_conv_kernel.onnx"
            tonnx.export(net_i_loc_conv_kernel, args=(kernel), f=ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True)
            onnx_model = onnx.load(ONNX_FILE_PATH)
            onnx.checker.check_model(onnx_model)

            ONNX_FILE_PATH = "siameserpn_mobilev2_rpn_head_"+ str(i) + "_loc_conv_search.onnx"
            tonnx.export(net_i_loc_conv_search, args=(search), f=ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True)
            onnx_model = onnx.load(ONNX_FILE_PATH)
            onnx.checker.check_model(onnx_model)

            kernel = net_i_loc_conv_kernel(kernel)
            search = net_i_loc_conv_search(search)
            feature = xcorr_depthwise(search, kernel)

            ONNX_FILE_PATH = "siameserpn_mobilev2_rpn_head_"+ str(i) + "_loc_head.onnx"
            tonnx.export(net_i_loc_head, args=(feature), f=ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True)
            onnx_model = onnx.load(ONNX_FILE_PATH)
            onnx.checker.check_model(onnx_model)

            
        return cls, loc

    def convert_backbone(self, net, input, stage):
        output = net(input)
        # print("Output for the " + name + "layers: ")
        # for out in output:
        #     print(out.shape)
        ONNX_FILE_PATH = "siameserpn_mobilev2_backbone_" + stage + ".onnx"
        # tonnx.export(net, args=(input), f=ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True, dynamic_axes={
        #               # dict value: manually named axes
        #               "input": {0:"in_zero"},
        #               # list value: automatic names
        #               "output": {0:"out_zero", 1:"out_one", 2:"out_two"},
        #           })
        tonnx.export(net, args=(input), f=ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True)

        onnx_model = onnx.load(ONNX_FILE_PATH)
        # check that the model converted fine
        onnx.checker.check_model(onnx_model)
        return output

    def convert_neck(self, net, input, stage):
        output = net(input)
        # print("Output for the " + name + "layers: ")
        # for out in output:
        #     print(out.shape)
        ONNX_FILE_PATH = "siameserpn_mobilev2_neck_" + stage +".onnx"
        tonnx.export(net, args=(input), f=ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True)

        onnx_model = onnx.load(ONNX_FILE_PATH)
        # check that the model converted fine
        onnx.checker.check_model(onnx_model)
        return output

    def convert_tensorrt(self, x, stage):
        xf = self.convert_backbone(self.backbone, x.cuda(), stage)
        if cfg.ADJUST.ADJUST:
            xf = self.convert_neck(self.neck, xf, stage)
        cls, loc = self.convert_rpn_head(self.rpn_head, self.zf, xf)    

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    # def forward(self, data):
    #     """ only used in training
    #     """
    #     template = data['template'].cuda()
    #     search = data['search'].cuda()
    #     label_cls = data['label_cls'].cuda()
    #     label_loc = data['label_loc'].cuda()
    #     label_loc_weight = data['label_loc_weight'].cuda()

    #     # get feature
    #     zf = self.backbone(template)
    #     xf = self.backbone(search)
    #     if cfg.MASK.MASK:
    #         zf = zf[-1]
    #         self.xf_refine = xf[:-1]
    #         xf = xf[-1]
    #     if cfg.ADJUST.ADJUST:
    #         zf = self.neck(zf)
    #         xf = self.neck(xf)
    #     cls, loc = self.rpn_head(zf, xf)

    #     # get loss
    #     cls = self.log_softmax(cls)
    #     cls_loss = select_cross_entropy_loss(cls, label_cls)
    #     loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

    #     outputs = {}
    #     outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
    #         cfg.TRAIN.LOC_WEIGHT * loc_loss
    #     outputs['cls_loss'] = cls_loss
    #     outputs['loc_loss'] = loc_loss

    #     if cfg.MASK.MASK:
    #         # TODO
    #         mask, self.mask_corr_feature = self.mask_head(zf, xf)
    #         mask_loss = None
    #         outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
    #         outputs['mask_loss'] = mask_loss
    #     return outputs


