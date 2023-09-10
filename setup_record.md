    # ModelSetup(
    #     name="512_image_clinical_resnet18",
    #     **common_args,  #
    #     **clinical_cl_args,
    #     **contrastive_learning_best_args,  # x
    #     **resnet18_args,  # and use mobilenet
    #     **backbone_out_channels_64,  # # don't use this next time.
    #     **batch_size_16_args,  # 
    #     ## multi-modal
    #     **contrastive_learning_args,  # x
    #     **image_512_args,  # ,
    #     **physionet_df_args,  #
    # ),
    # t1 work
    # ModelSetup(
    #     name="128_image_chexpert",
    #     **common_args,
    #     **chexpert_best_args,
    #     **resnet50_args,
    #     **small_model_args,
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **image_128_args,
    # ),
    # swap dataset. (trainable)
    # ModelSetup(
    #     name="128_image_chexpert_physio",
    #     **common_args,
    #     **chexpert_best_args,
    #     **resnet50_args,
    #     **small_model_args,
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **image_128_args,
    #     **physionet_df_args,
    # ),
    # mobilenet (worse than resnet)
    # ModelSetup(
    #     name="128_image_chexpert_mobile",
    #     **common_args,
    #     **chexpert_best_args,
    #     **mobilenet_args,
    #     **small_model_args,
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **image_128_args,
    #     **physionet_df_args,
    # ),
    # larger image. (working well)
    # ModelSetup(
    #     name="128_image_chexpert_mobile",
    #     **common_args,
    #     **chexpert_best_args,
    #     **resnet50_args,
    #     **small_model_args,
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **image_512_args,
    #     **physionet_df_args,
    # ),
    ## larger batch size
    # ModelSetup(
    #     name="512_image_chexpert_res",
    #     **common_args,
    #     **chexpert_best_args,
    #     **resnet50_args,
    #     **small_model_args,
    #     **batch_size_64_args,
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **image_128_args,
    #     **physionet_df_args,
    # ),
    # backbone_16
    # ModelSetup(
    #     name="128_image_chexpert_mobile",
    #     **common_args,
    #     **chexpert_best_args,
    #     **resnet50_args,
    #     **backbone_out_channels_16,
    #     **batch_size_64_args,
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **image_128_args,
    #     **physionet_df_args,
    # ),
    # contrastive_learning_best_args = {
    #     "performance_standards": [
    #         {
    #             "task": TaskStrs.XRAY_CLINICAL_CL,
    #             "metric": "accuracy",
    #         },
    #     ]
    # }
    # contrastive_learning_args = {
    #     # "backbone_out_channels": None,
    #     "sources": [SourceStrs.XRAYS, SourceStrs.CLINICAL_1D],
    #     "tasks": [
    #         TaskStrs.XRAY_CLINICAL_CL,
    #     ],
    #     "fusor": FusionStrs.NO_ACTION,
    # }
    # move into contrastive learning. (working)
    # ModelSetup(
    #     name="512_image_chexpert_resnet50",
    #     **common_args, #
    #     **contrastive_learning_best_args, #x
    #     **resnet50_args, #,
    #     **backbone_out_channels_16, #
    #     **batch_size_16_args,  #
    #     ## multi-modal
    #     **contrastive_learning_args, # x
    #     **image_512_args, #,
    #     **physionet_df_args, #
    # ),
    # swap to clinical instead of chexpert. (working)
    # ModelSetup(
    #     name="512_image_clinical_resnet50",
    #     **common_args, #
    #     **clinical_cl_args,
    #     **contrastive_learning_best_args, #x
    #     **resnet50_args, #,
    #     **backbone_out_channels_16, #
    #     **batch_size_16_args,  #
    #     ## multi-modal
    #     **contrastive_learning_args, # x
    #     **image_512_args, #,
    #     **physionet_df_args, #
    # ),
    # try a not working one. then we will know the reason. (Not working)
    # ModelSetup(
    #     name="512_image_clinical_resnet50",
    #     **common_args, #
    #     **clinical_cl_args,
    #     **contrastive_learning_best_args, #x
    #     **mobilenet_args, # and use mobilenet
    #     **backbone_out_channels_16, #
    #     **batch_size_64_args,  # => larger
    #     ## multi-modal
    #     **contrastive_learning_args, # x
    #     **image_512_args, #,
    #     **physionet_df_args, #
    # ),
    # change batch size to 16, but use mobilenet (not working)
    # ModelSetup(
    #     name="512_image_clinical_mobilenet",
    #     **common_args, #
    #     **clinical_cl_args,
    #     **contrastive_learning_best_args, #x
    #     **mobilenet_args, # and use mobilenet
    #     **backbone_out_channels_16, #
    #     **batch_size_16_args,  # => larger
    #     ## multi-modal
    #     **contrastive_learning_args, # x
    #     **image_512_args, #,
    #     **physionet_df_args, #
    # ),
    # resnet but samller image
    # ModelSetup(
    #     name="128_image_clinical_resnet50",
    #     **common_args,  #
    #     **clinical_cl_args,
    #     **contrastive_learning_best_args,  # x
    #     **resnet50_args,  # and use mobilenet
    #     **backbone_out_channels_16,  #
    #     **batch_size_16_args,  # => larger
    #     ## multi-modal
    #     **contrastive_learning_args,  # x
    #     **image_128_args,  # ,
    #     **physionet_df_args,  #
    # ),
    # resnet but larger batch size
    # ModelSetup(
    #     name="128_image_clinical_resnet50",
    #     **common_args,  #
    #     **clinical_cl_args,
    #     **contrastive_learning_best_args,  # x
    #     **resnet50_args,  # and use mobilenet
    #     **backbone_out_channels_16,  #
    #     **batch_size_256_args,  # => larger
    #     ## multi-modal
    #     **contrastive_learning_args,  # x
    #     **image_128_args,  # ,
    #     **physionet_df_args,  #
    # ),
    ### Real training one ###
    # ModelSetup(
    #     name="512_image_clinical_resnet18",
    #     **common_args,  #
    #     **clinical_cl_args,
    #     **contrastive_learning_best_args,  # x
    #     **resnet18_args,  # and use mobilenet
    #     **backbone_out_channels_64,  #
    #     **batch_size_16_args,  # 
    #     ## multi-modal
    #     **contrastive_learning_args,  # x
    #     **image_512_args,  # ,
    #     **physionet_df_args,  #
    # ),

    ### real training without aug
    # ModelSetup(
    #     name="512_image_clinical_resnet18_woaug",
    #     **common_args,  #
    #     **clinical_cl_args,
    #     **contrastive_learning_best_args,  # x
    #     **resnet18_args,  # and use mobilenet
    #     **backbone_out_channels_64,  #
    #     **batch_size_16_args,  # 
    #     ## multi-modal
    #     **contrastive_learning_args,  # x
    #     **image_512_args,  # ,
    #     **physionet_df_args,  #
    # ),

    ## real training without 64 backbone.
    ModelSetup(
        name="512_image_clinical_resnet18_no64",
        **common_args,  #
        **clinical_cl_args,
        **contrastive_learning_best_args,  # x
        **resnet18_args,  # and use mobilenet
        **backbone_out_channels_64,  #
        # **batch_size_16_args,  # 
        ## multi-modal
        **contrastive_learning_args,  # x
        **image_512_args,  # ,
        **physionet_df_args,  #
    ),

    # ModelSetup(
    #     name="chexpert_test",
    #     **common_args,
    #     **chexpert_best_args,
    #     **mobilenet_args,
    #     **image_512_args, # larger image
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **physionet_df_args,
    #     **batch_size_64_args, # smaller batch size
    #     **backbone_out_channels_16 # smaller channels.
    # ),
    ## using this one to do contrastive learning. (and not working.)
    # ModelSetup(
    #     name="cl_physio_discrete_mobilenet_li",
    #     **common_args,
    #     **contrastive_learning_best_args,
    #     **mobilenet_args,
    #     **image_512_args, # larger image
    #     ## multi-modal
    #     **contrastive_learning_args,
    #     **physionet_df_args,
    #     **batch_size_64_args, # smaller batch size
    #     **backbone_out_channels_16 # smaller channels.
    # ),
    ######### Till this, see which argument make that untrain on val and test. # this is the one not working
    # ModelSetup(
    #     name="chexpert_test",
    #     **common_args,
    #     **chexpert_best_args,
    #     **mobilenet_args,
    #     **image_512_args, # larger image
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **physionet_df_args,
    #     **batch_size_64_args, # smaller batch size
    #     **backbone_out_channels_16 # smaller channels.
    # ),
    # ModelSetup(
    #     name="512_image_chexpert",
    #     **common_args,
    #     **chexpert_best_args,
    #     **resnet50_args,
    #     **small_model_args,
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **image_512_args,
    # ),
    # image size 128
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
    #     # **clinical_using_backbone_args,p
    #     # **element_wise_sum_fusor_args,
    #     # **no_fusion_1D_args,
    # ),
    # ModelSetup(
    #     name="load_test",
    #     **common_args,
    #     **chexpert_best_args,
    #     **resnet50_args,
    #     **image_128_args,
    #     # **small_model_args,
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **pretrained_cl_load_fix_weight_args,
    #     **batch_size_4_args,
    # ),
    #  ModelSetup(
    #     name="load_test",
    #     **common_args,
    #     **chexpert_best_args,
    #     **resnet50_args,
    #     **image_128_args,
    #     # **small_model_args,
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **pretrained_cl_load_no_fix_weight_args,
    #     **batch_size_4_args,
    # ),
    # contrastive learning test.
    # ModelSetup(
    #     name="cl_physionet",
    #     **common_args,
    #     **contrastive_learning_best_args,
    #     **resnet50_args,
    #     **image_128_args,
    #     ## multi-modal
    #     **contrastive_learning_args,
    #     **physionet_df_args,
    #     **batch_size_256_args,
    # ),
    # ModelSetup(
    #     name="cl_physio_discrete_mobilenet_li",
    #     **common_args,
    #     **contrastive_learning_best_args,
    #     **mobilenet_args,
    #     **image_512_args, # larger image
    #     ## multi-modal
    #     **contrastive_learning_args,
    #     **physionet_df_args,
    #     **batch_size_64_args, # smaller batch size
    #     **backbone_out_channels_16 # smaller channels.
    # ),
    # train a mobilenet with smaller image size.
    # ModelSetup(
    #     name="chexpert_test",
    #     **common_args,
    #     **chexpert_best_args,
    #     **mobilenet_args,
    #     **image_512_args, # larger image
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **physionet_df_args,
    #     **batch_size_64_args, # smaller batch size
    #     **backbone_out_channels_16 # smaller channels.
    # ),

    ### run three for testing the performance of pretrained backbones.
    ## baseline
    ## fix weights

    # ModelSetup(
    #     name="load_test",
    #     **common_args,
    #     **chexpert_best_args,
    #     **resnet50_args,
    #     **image_128_args,
    #     # **small_model_args,
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **pretrained_cl_load_fix_weight_args,
    #     **batch_size_4_args,
    # ),

    ### real running for downstream tasks
    # ModelSetup(
    #     name="chexpert_cl_fix",
    #     **common_args,
    #     **chexpert_best_args,
    #     **resnet18_args,
    #     **image_512_args,
    #     # **small_model_args,
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **pretrained_cl_load_fix_weight_args,
    #     **batch_size_4_args,
    # ),
    # ModelSetup(
    #     name="chexpert_baseline",
    #     **common_args,
    #     **chexpert_best_args,
    #     **resnet18_args,
    #     **image_512_args,
    #     # **small_model_args,
    #     **backbone_out_channels_none,
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **batch_size_4_args,
    # ),
    # no fix weights
    # ModelSetup(
    #     name="chexpert_cl_no_fix",
    #     **common_args,
    #     **chexpert_best_args,
    #     **resnet18_args,
    #     **image_512_args,
    #     # **small_model_args,
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **pretrained_cl_load_no_fix_weight_args,
    #     **batch_size_4_args,
    # ),

    ### lesion detection.
    # ModelSetup(
    #     name="lesion_detection_baseline",
    #     **common_args,
    #     **lesion_detection_best_args,
    #     **lesion_detection_baseline_args,
    #     **resnet18_args,
    #     **image_512_args,
    #     # **small_model_args,
    #     **backbone_out_channels_none,
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **batch_size_4_args,
    #     **no_bb_to_mask_args,
    # ),

    ### without any pretrained (ImageNet):
    # ModelSetup(
    #     name="chexpert_baseline_no_pretrain",
    #     **common_args,
    #     **chexpert_best_args,
    #     **resnet18_args,
    #     **image_512_args,
    #     # **small_model_args,
    #     **backbone_out_channels_64,
    #     ## multi-modal
    #     **baseline_chexpert_args,
    #     **batch_size_4_args,
    #     **dont_use_pytorch_pretrained,
    # ),

    # ## for lesion detection using fixation + xray
    # ModelSetup(
    #     name="fixations_xray_lesion_detection",
    #     **common_args,
    #     ## batch size
    #     **batch_size_4_args,
    #     ## model 
    #     **resnet18_args,
    #     ## image size
    #     **image_512_args,
    #     ## define early-stop strategy.
    #     **lesion_detection_best_args, 
    #     ## define input and output
    #     **with_fix_args, 
    #     ## for model size.
    #     **small_model_args,
    #     ## fusion
    #     **element_wise_sum_fusor_args,
    # ),

    # ## only use xray for lesion detection.
    # ModelSetup(
    #     name="xray_lesion_detection",
    #     **common_args,
    #     ## batch size
    #     **batch_size_4_args,
    #     ## model 
    #     **resnet18_args,
    #     ## image size
    #     **image_512_args,
    #     ## define early-stop strategy.
    #     **lesion_detection_best_args, 
    #     ## define input and output
    #     **lesion_detection_baseline_args, 
    #     ## for model size.
    #     **small_model_args,
    # ),

    # ## for chexpert using fixation + xray
    # ModelSetup(
    #     name="fixations_xray_chexpert",
    #     **common_args,
    #     ## batch size
    #     **batch_size_4_args,
    #     ## model 
    #     **resnet18_args,
    #     ## image saize
    #     **image_512_args,
    #     ## define early-stop strategy.
    #     **chexpert_best_args, 
    #     ## define input and output
    #     **with_fix_chexpert_args, 
    #     ## for model size.
    #     **small_model_args,
    #     ## fusion
    #     **element_wise_sum_fusor_args,
    # ),

    # ## only use xray for chexpert.
    # ModelSetup(
    #     name="xray_chexpert",
    #     **common_args,
    #     ## batch size
    #     **batch_size_4_args,
    #     ## model 
    #     **resnet18_args,
    #     ## image size
    #     **image_512_args,
    #     ## define early-stop strategy.
    #     **chexpert_best_args, 
    #     ## define input and output
    #     **baseline_chexpert_args, 
    #     ## for model size.
    #     **small_model_args,
    # ),


    ###################  Contrastive learning setup ###################

    # continuous
    # ModelSetup(
    #     name="continuous_clinical__no64",
    #     **common_args,  #
    #     **continuous_clinical_cl_args,
    #     **using_aug_for_cl,
    #     **contrastive_learning_best_args,  # x
    #     **resnet18_args,  # and use mobilenet
    #     # **backbone_out_channels_64,  #
    #     **batch_size_16_args,  # 
    #     ## multi-modal
    #     **contrastive_learning_args,  # x
    #     **image_512_args,  # ,
    #     **physionet_df_args,  #
    # ),
    # discrete
    # ModelSetup(
    #     name="discrete_clinical__no64",
    #     **common_args,  #
    #     **discrete_clinical_cl_args,
    #     **using_aug_for_cl,
    #     **contrastive_learning_best_args,  # x
    #     **resnet18_args,  # and use mobilenet
    #     # **backbone_out_channels_64,  #
    #     **batch_size_16_args,  # 
    #     ## multi-modal
    #     **contrastive_learning_args,  # x
    #     **image_512_args,  # ,
    #     **physionet_df_args, #
    # ),

    #### Training in order - Baseline ####