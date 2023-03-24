# MIMIC-Eye-applications

ModelSetup(name='lesion_dsetection_with_fixation_convnext_base_silent_report', sources=['xrays', 'fixations'], tasks=['lesion-detection'], fusor='element-wise sum', decoder_channels=[128, 64, 32, 16, 8], lesion_label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'], save_early_stop_model=True, record_training_performance=False, backbone='convnext_base', optimiser='sgd', lr=0.01, weight_decay=1e-05, sgb_momentum=0.9, image_backbone_pretrained=True, heatmap_backbone_pretrained=False, image_size=512, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], backbone_out_channels=64, batch_size=8, warmup_epochs=0, model_warmup_epochs=0, loss_warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.5, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=[20, 40, 60, 80, 100], multiStepLR_gamma=0.5, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=False, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False, gt_in_train_till=0, measure_test=True, eval_freq=10, use_iobb=True, iou_thrs=array([0.5]), fiaxtions_mode_input='reporting', fiaxtions_mode_label='reporting', clinical_num=['age', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'acuity'], clinical_cat=['gender'], categorical_col_maps={'gender': 2}, clinical_cat_emb_dim=32, clinical_conv_channels=32, clinical_upsample='deconv', chexpert_label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'], negbio_label_cols=['Atelectasis_negbio', 'Cardiomegaly_negbio', 'Consolidation_negbio', 'Edema_negbio', 'Enlarged Cardiomediastinum_negbio', 'Fracture_negbio', 'Lung Lesion_negbio', 'Lung Opacity_negbio', 'No Finding_negbio', 'Pleural Effusion_negbio', 'Pleural Other_negbio', 'Pneumonia_negbio', 'Pneumothorax_negbio', 'Support Devices_negbio'], performance_standard_task='lesion-detection', performance_standard_metric='ap', random_flip=True, use_dynamic_weight=True)
