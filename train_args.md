all_model_setups = [
    ## Retrained with AP

    # baseline
    # ModelSetup(
    #     name="baseline_mobilenet",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_baseline_args,
    #     # **clinical_deconv_upsample_args,
    #     # **clinical_using_backbone_args,
    #     # **element_wise_sum_fusor_args,
    #     # **no_fusion_1D_args,
    # ),

    # # 1D+3D (sum)
    # ModelSetup(
    #     name="clinical_mobilenet_sum",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **clinical_using_backbone_args,
    #     **element_wise_sum_fusor_args,
    #     **fusion_1D_args,
    # ),

    # # 1D+3D (hadamard)
    # ModelSetup(
    #     name="clinical_mobilenet_hadamard",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **clinical_using_backbone_args,
    #     **hadamard_fusor_args,
    #     **fusion_1D_args,
    # ),

    # # Concat(Conv2D)
    # ModelSetup(
    #     name="clinical_mobilenet_concat_conv",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **clinical_using_backbone_args,
    #     **concat_fusor_args,
    #     **fusion_1D_args,
    # ),

    # # Concat(Linear)
    # ModelSetup(
    #     name="clinical_mobilenet_concat_linear",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **clinical_using_backbone_args,
    #     **concat_token_mixer_fusor_args,
    #     **fusion_1D_args,
    # ),

    # Concat(Conv2D_block)  # with block not working better.
    # ModelSetup(
    #     name="clinical_mobilenet_concat_convb",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **clinical_using_backbone_args,
    #     **concat_block_fusor_args,
    #     **fusion_1D_args,
    # ),

    # not using backbone, just spatialisation.
    # ModelSetup(
    #     name="clinical_mobilenet_concat_conv",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **clinical_not_using_backbone_args,

    #     ## depend on the performance to select which one to go.
    #     # **concat_block_fusor_args,
    #     **concat_fusor_args,
    #     **fusion_1D_args,
    # ),

    # what to do for spatialisation.

    # only spatialisation vs spatialisation + backbone => which one is better

    # spatiallisation methods: # then test out all these spatialisation method.

    # ModelSetup(
    #     name="clinical_mobilenet_concat_convb_nb",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **clinical_not_using_backbone_args,

    #     ## depend on the performance to select which one to go.
    #     # **concat_block_fusor_args,
    #     # **concat_fusor_args,
    #     **fusion_1D_args,
    # ),

    # ModelSetup(
    #     name="clinical_mobilenet_hadamard_convb_nb",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **clinical_not_using_backbone_args,

    #     ## depend on the performance to select which one to go.
    #     # **concat_block_fusor_args,
    #     # **concat_fusor_args,
    #     **fusion_1D_args,
    # ),

    # deconv
    # interpolate
    # repeat
    
    # deconv is already done.
    
    # interpolate.
    # ModelSetup(
    #     name="clinical_mobilenet_concat_conv_inter",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_interpolate_upsample_args,

    #     # depend on the result, decide to use backbone or not
    #     # **clinical_not_using_backbone_args,
    #     **clinical_using_backbone_args,

    #     ## depend on the performance to select which one to go.
    #     # **concat_block_fusor_args,
    #     **concat_fusor_args,
    #     **fusion_1D_args,
    # ),

    # # repeat.
    # ModelSetup(
    #     name="clinical_mobilenet_concat_conv_repeat",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_repeat_upsample_args,

    #     # depend on the result, decide to use backbone or not
    #     # **clinical_not_using_backbone_args,
    #     **clinical_using_backbone_args,

    #     ## depend on the performance to select which one to go.
    #     # **concat_block_fusor_args,
    #     **concat_fusor_args,
    #     **fusion_1D_args,
    # ),

    # TRY OUT DEFORMABLE

    # ModelSetup(
    #     name="clinical_mobilenet_concat_deform",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,

    #     # depend on the result, decide to use backbone or not
    #     # **clinical_not_using_backbone_args,
    #     **clinical_using_backbone_args,

    #     ## depend on the performance to select which one to go.
    #     # **concat_block_fusor_args,
    #     # **concat_fusor_args
    #     **concat_deform_fusor_args,
    #     **fusion_1D_args,
    # ),

    ######
    #  Try other models.


    # baseline
    ModelSetup(
        name="baseline_resnet50",
        **lesion_detection_best_args,
        **resnet50_args,
        **small_model_args,
        **common_args,
        **bb_to_mask_args,
        ## multi-modal
        **lesion_detection_baseline_args,
        # **clinical_deconv_upsample_args,
        # **clinical_using_backbone_args,
        # **element_wise_sum_fusor_args,
        # **no_fusion_1D_args,
    ),

    ModelSetup(
        name="clinical_resnet50_concat",
        **lesion_detection_best_args,
        **resnet50_args,
        **small_model_args,
        **common_args,
        **bb_to_mask_args,
        ## multi-modal
        **lesion_detection_with_clinical_args,
        **clinical_deconv_upsample_args,

        # depend on the result, decide to use backbone or not
        # **clinical_not_using_backbone_args,
        **clinical_using_backbone_args,

        ## depend on the performance to select which one to go.
        # **concat_block_fusor_args,
        # **concat_fusor_args
        **concat_fusor_args,
        **fusion_1D_args,
    ),

    ModelSetup(
        name="clinical_resnet50_hadamard",
        **lesion_detection_best_args,
        **resnet50_args,
        **small_model_args,
        **common_args,
        **bb_to_mask_args,
        ## multi-modal
        **lesion_detection_with_clinical_args,
        **clinical_deconv_upsample_args,

        # depend on the result, decide to use backbone or not
        # **clinical_not_using_backbone_args,
        **clinical_using_backbone_args,

        ## depend on the performance to select which one to go.
        # **concat_block_fusor_args,
        # **concat_fusor_args
        **hadamard_fusor_args,
        **fusion_1D_args,
    ),



    # ModelSetup(
    #     name="clinical_convnext_base_concat",
    #     **lesion_detection_ap_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **concat_fusor_args,
    #     **no_fusion_1D_args,
    # ),


    # ModelSetup(
    #     name="baseline_convnext_base_sum",
    #     **lesion_detection_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **clinical_not_using_backbone_args,
    #     **element_wise_sum_fusor_args,
    #     **fusion_1D_args,
    # ),

    # ModelSetup(
    #     name="clinical_convnext_base_sum",
    #     **lesion_detection_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **clinical_using_backbone_args,
    #     **element_wise_sum_fusor_args,
    #     **fusion_1D_args,
    # ),

    # ModelSetup(
    #     name="clinical_convnext_base_product",
    #     **lesion_detection_ap_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **hadamard_fusor_args,
    #     **no_fusion_1D_args,
    # ),

    # ModelSetup(
    #     name="clinical_convnext_base_concat",
    #     **lesion_detection_ap_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **concat_fusor_args,
    #     **no_fusion_1D_args,
    # ),

    # ModelSetup(
    #     name="clinical_convnext_base_concat_b",
    #     **lesion_detection_ap_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **concat_block_fusor_args,
    #     **no_fusion_1D_args,
    # ),

    # ModelSetup(
    #     name="clinical_convnext_base_tm",
    #     **lesion_detection_ap_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **concat_token_mixer_fusor_args,
    #     **no_fusion_1D_args,
    # ),

    # ModelSetup(
    #     name="clinical_convnext_base_tm_b",
    #     **lesion_detection_ap_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_deconv_upsample_args,
    #     **concat_block_token_mixer_fusor_args,
    #     **no_fusion_1D_args,
    # ),

    ## TO TRAIN
    # ModelSetup(
    #     name="clinical_convnext_base",
    #     **lesion_detection_ap_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_interpolate_upsample_args,
    #     **element_wise_sum_fusor_args,
    #     **no_fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="clinical_convnext_base",
    #     **lesion_detection_ap_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **lesion_detection_with_clinical_args,
    #     **clinical_interpolate_upsample_args,
    #     **element_wise_sum_fusor_args,
    #     **no_fusion_1D_args,
    # ),

    # RECORDS
    # ModelSetup(
    #     name="baseline_convnext_base",
    #     **lesion_detection_ap_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     **lesion_detection_baseline_args,
    # ),
    # ModelSetup(
    #     name="clinical_densenet_3D",
    #     **lesion_detection_ap_best_args,
    #     **densenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **no_fusion_1D_args,
    #     **element_wise_sum_fusor_args,
    #     **lesion_detection_with_clinical_args,
    # ),
    # ModelSetup(
    #     name="clinical_densenet_1D_3D",
    #     **lesion_detection_ap_best_args,
    #     **densenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     ## multi-modal
    #     **fusion_1D_args,
    #     **element_wise_sum_fusor_args,
    #     **lesion_detection_with_clinical_args,
    # ),
    # ModelSetup(
    #     name="baseline_densenet",
    #     **lesion_detection_ap_best_args,
    #     **densenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **bb_to_mask_args,
    #     **lesion_detection_baseline_args,
    # ),
    ##### Records
    # ModelSetup(
    #     name="lesion_dsetection_baseline_mobilenet",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_baseline_args,
    #     **bb_to_mask_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_mobilenet",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_args,
    #     **bb_to_mask_args,
    #     **fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_mobilenet",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_args,
    #     **bb_to_mask_args,
    #     **no_fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_baseline_resnet18",
    #     **lesion_detection_best_args,
    #     **resnet18_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_baseline_args,
    #     **bb_to_mask_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_resnet18_1D_3D",
    #     **lesion_detection_best_args,
    #     **resnet18_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_args,
    #     **bb_to_mask_args,
    #     **fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_resnet18_3D",
    #     **lesion_detection_best_args,
    #     **resnet18_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_args,
    #     **bb_to_mask_args,
    #     **no_fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_convnext_base_3D",
    #     **lesion_detection_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_args,
    #     **bb_to_mask_args,
    #     **no_fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_convnext_base_1D_3D",
    #     **lesion_detection_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_args,
    #     **bb_to_mask_args,
    #     **fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_baseline_convnext_base",
    #     **lesion_detection_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_baseline_args,
    #     **bb_to_mask_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_baseline_efficientnet_b0",
    #     **lesion_detection_best_args,
    #     **efficientnet_b0_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_baseline_args,
    #     **bb_to_mask_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_efficientnet_b0_1D_3D",
    #     **lesion_detection_best_args,
    #     **efficientnet_b0_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_args,
    #     **bb_to_mask_args,
    #     **fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_efficientnet_b0_3D",
    #     **lesion_detection_best_args,
    #     **efficientnet_b0_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_args,
    #     **bb_to_mask_args,
    #     **no_fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_baseline_densenet",
    #     **lesion_detection_best_args,
    #     **densenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_baseline_args,
    #     **bb_to_mask_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_densenet_1D_3D",
    #     **lesion_detection_best_args,
    #     **densenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_args,
    #     **bb_to_mask_args,
    #     **fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_densenet_3D",
    #     **lesion_detection_best_args,
    #     **densenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_args,
    #     **bb_to_mask_args,
    #     **no_fusion_1D_args,
    # ),
    #  ModelSetup(
    #     name="lesion_dsetection_baseline_resnet50",
    #     **lesion_detection_best_args,
    #     **resnet50_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_baseline_args,
    #     **bb_to_mask_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_resnet50_1D_3D",
    #     **lesion_detection_best_args,
    #     **resnet50_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_args,
    #     **bb_to_mask_args,
    #     **fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_resnet50_3D",
    #     **lesion_detection_best_args,
    #     **resnet50_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_args,
    #     **bb_to_mask_args,
    #     **no_fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_baseline_efficientnet_b5",
    #     **lesion_detection_best_args,
    #     **efficientnet_b5_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_baseline_args,
    #     **bb_to_mask_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_efficientnet_b5_1D_3D",
    #     **lesion_detection_best_args,
    #     **efficientnet_b5_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_args,
    #     **bb_to_mask_args,
    #     **fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_efficientnet_b5_3D",
    #     **lesion_detection_best_args,
    #     **efficientnet_b5_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_args,
    #     **bb_to_mask_args,
    #     **no_fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_convnext_base_3D_cocat_token_mixer",
    #     **lesion_detection_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_concat_with_token_mixer_args,
    #     **bb_to_mask_args,
    #     **no_fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_convnext_base_3D_hadamard",
    #     **lesion_detection_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_hadmard_args,
    #     **bb_to_mask_args,
    #     **no_fusion_1D_args,
    # ),
    ### also test out the tabular data expander.
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_expander_convnext_base_3D_cocat",
    #     **lesion_detection_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_concat_args,
    #     **bb_to_mask_args,
    #     **no_fusion_1D_args,
    #     **clinical_expander_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_convnext_base_3D_cocat",
    #     **lesion_detection_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_concat_args,
    #     **bb_to_mask_args,
    #     **no_fusion_1D_args,
    # ),
    ### prepare to train
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_convnext_base_3D_cocat_with_norm_act_token_mixer",
    #     **lesion_detection_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_concat_with_norm_act_token_mixer_args,
    #     **bb_to_mask_args,
    #     **no_fusion_1D_args,
    # ),
    # test out pos_weight for loss term of objectness_loss
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_convnext_base_3D_cocat_pos_weight_10",
    #     **lesion_detection_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_concat_args,
    #     **bb_to_mask_args,
    #     **no_fusion_1D_args,
    #     **pos_weight_10_args,
    # ),
    # ModelSetup(
    #     name="lesion_dsetection_with_clinical_convnext_base_3D_cocat_pos_weight_100",
    #     **lesion_detection_best_args,
    #     **convnext_base_args,
    #     **small_model_args,
    #     **common_args,
    #     **lesion_detection_with_clinical_concat_args,
    #     **bb_to_mask_args,
    #     **no_fusion_1D_args,
    #     **pos_weight_100_args,
    # ),
    ### to train.
    ## if it doesn't work, we use AP and do it again.
    # ModelSetup(
    #     name="lesion_dsetection_with_fix_mobilenet",
    #     **lesion_detection_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     **common_args,
    #     **with_fix_args,
    #     **reporting_report_args,  # see if disabling this will get more time.
    # ),
]

# pick the best one.
