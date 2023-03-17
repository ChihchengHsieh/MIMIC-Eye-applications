
### object-detection + heatmap-generation

```python
========================================For Training [basline_test]========================================
ModelSetup(name='basline_test', sources=['image'], tasks=['object-detection', 'heatmap-generation'], decoder_channels=[128, 64, 32, 16, 1], label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'], save_early_stop_model=True, record_training_performance=True, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=1e-09, image_backbone_pretrained=True, heatmap_backbone_pretrained=False, image_size=512, backbone_out_channels=64, batch_size=16, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=False, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False, gt_in_train_till=0, measure_test=True, eval_freq=10, use_iobb=True, iou_thrs=array([0.5]))
===========================================================================================================

Best AP validation model has been saved to: [val_ar_0_5102_ap_0_1966_test_ar_0_5504_ap_0_2362_epoch18_02-14-2023 07-03-16_basline_test]
Best AR validation model has been saved to: [val_ar_0_5684_ap_0_1598_test_ar_0_5862_ap_0_1839_epoch12_02-14-2023 05-08-38_basline_test]
The final model has been saved to: [val_ar_0_4393_ap_0_1682_test_ar_0_4864_ap_0_2030_epoch30_02-14-2023 10-36-44_basline_test]

===========================================================================================================
Using pretrained backbone. mobilenet_v3
[model]: 5,961,224
Max AP on test: [0.2332]
```

### object-detection alone

```python
========================================For Training [basline_test]========================================
ModelSetup(name='basline_test', sources=['image'], tasks=['object-detection'], decoder_channels=[128, 64, 32, 16, 1], label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'], save_early_stop_model=True, record_training_performance=True, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=1e-09, image_backbone_pretrained=True, heatmap_backbone_pretrained=False, image_size=512, backbone_out_channels=64, batch_size=16, warmup_epochzs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=False, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False, gt_in_train_till=0, measure_test=True, eval_freq=10, use_iobb=True, iou_thrs=array([0.5]))
===========================================================================================================

Best AP validation model has been saved to: [val_ar_0_5550_ap_0_1937_test_ar_0_5481_ap_0_1928_epoch19_02-14-2023 16-45-12_basline_test]
Best AR validation model has been saved to: [val_ar_0_6200_ap_0_1788_test_ar_0_6222_ap_0_1942_epoch12_02-14-2023 15-28-54_basline_test]
The final model has been saved to: [val_ar_0_5150_ap_0_1809_test_ar_0_5314_ap_0_1928_epoch30_02-14-2023 18-41-59_basline_test]

===========================================================================================================
Using pretrained backbone. mobilenet_v3
[model]: 5,593,289
Max AP on test: [0.2048]
```


## TODO:

- [ ] test out the evaluation. 
- [ ] implement the global disease diagnosis.
- [ ] implement clinical feature extractor (tabular)
- [ ] implement another sequential feature extractor. (using RNN or transformer)




## With fix [0.6788]
``` python

====================| Epoch [30] Done | It has took [283.78] min, Avg time: [567.57] sec/epoch | Estimate time for [30] epochs: [283.78] min | Epoch took [523] sec | Patience [5] |====================
====================| Training Done, start testing! | [30] Epochs Training time: [17028] seconds, Avg time / Epoch: [567.6] seconds====================
====================Best Performance model has been saved to: [val_chexpert-classification_auc_0_6562_test_chexpert-classification_auc_0_6610_epoch29_02-24-2023 09-40-30_chexpert_without_fix]====================
Evaluation:  [ 0/29]  eta: 0:01:19  loss: 0.3232 (0.3232)  chexpert-classification_performer-image_classfication_classification_loss: 0.3232 (0.3232)  model_time: 1677196160.0000 (1677196157.3763)  evaluator_time: 0.0000 (0.0000)  time: 2.7360  data: 1.6313  max mem: 3242
Evaluation:  [28/29]  eta: 0:00:02  loss: 0.2754 (0.2816)  chexpert-classification_performer-image_classfication_classification_loss: 0.2754 (0.2816)  model_time: 1677196160.0000 (1677196197.6145)  evaluator_time: 0.0000 (0.0000)  time: 2.7844  data: 1.7020  max mem: 3242
Evaluation: Total time: 0:01:21 (2.8039 s / it)
Averaged stats: loss: 0.2754 (0.2816)  chexpert-classification_performer-image_classfication_classification_loss: 0.2754 (0.2816)  model_time: 1677196160.0000 (1677196197.6145)  evaluator_time: 0.0000 (0.0000)
====================The final model has been saved to: [val_chexpert-classification_auc_0_6156_test_chexpert-classification_auc_0_6034_epoch30_02-24-2023 09-50-36_chexpert_without_fix]====================



========================================For Training [chexpert_with_fix]========================================
ModelSetup(name='chexpert_with_fix', sources=['xrays'], tasks=['fixation-generation', 'chexpert-classification'], fusor='element-wise sum', decoder_channels=[128, 64, 32, 16, 8], lesion_label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'], save_early_stop_model=True, record_training_performance=True, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=1e-09, image_backbone_pretrained=True, heatmap_backbone_pretrained=False, image_size=512, backbone_out_channels=64, batch_size=16, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=False, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False, gt_in_train_till=0, measure_test=True, eval_freq=10, use_iobb=True, iou_thrs=array([0.5]), fiaxtions_mode='reporting', clinical_num=['age', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'acuity'], clinical_cat=['gender'], categorical_col_maps={'gender': 2}, clinical_cat_emb_dim=32, clinical_conv_channels=32, clinical_upsample='deconv', chexpert_label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'], negbio_label_cols=['Atelectasis_negbio', 'Cardiomegaly_negbio', 'Consolidation_negbio', 'Edema_negbio', 'Enlarged Cardiomediastinum_negbio', 'Fracture_negbio', 'Lung Lesion_negbio', 'Lung Opacity_negbio', 'No Finding_negbio', 'Pleural Effusion_negbio', 'Pleural Other_negbio', 'Pneumonia_negbio', 'Pneumothorax_negbio', 'Support Devices_negbio'], performance_standard_task='chexpert-classification', performance_standard_metric='auc')
================================================================================================================

Best performance model has been saved to: [val_chexpert-classification_auc_0_6643_test_chexpert-classification_auc_0_6788_epoch29_02-24-2023 04-37-15_chexpert_with_fix]
The final model has been saved to: [val_chexpert-classification_auc_0_6541_test_chexpert-classification_auc_0_6558_epoch30_02-24-2023 05-05-25_chexpert_with_fix]

================================================================================================================
```


# Without fix [0.6562]

```python

========================================For Training [chexpert_without_fix]========================================
ModelSetup(name='chexpert_without_fix', sources=['xrays'], tasks=['chexpert-classification'], fusor='element-wise sum', decoder_channels=[128, 64, 32, 16, 8], lesion_label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'], save_early_stop_model=True, record_training_performance=True, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=1e-09, image_backbone_pretrained=True, heatmap_backbone_pretrained=False, image_size=512, backbone_out_channels=64, batch_size=16, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=False, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False, gt_in_train_till=0, measure_test=True, eval_freq=10, use_iobb=True, iou_thrs=array([0.5]), fiaxtions_mode='reporting', clinical_num=['age', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'acuity'], clinical_cat=['gender'], categorical_col_maps={'gender': 2}, clinical_cat_emb_dim=32, clinical_conv_channels=32, clinical_upsample='deconv', chexpert_label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'], negbio_label_cols=['Atelectasis_negbio', 'Cardiomegaly_negbio', 'Consolidation_negbio', 'Edema_negbio', 'Enlarged Cardiomediastinum_negbio', 'Fracture_negbio', 'Lung Lesion_negbio', 'Lung Opacity_negbio', 'No Finding_negbio', 'Pleural Effusion_negbio', 'Pleural Other_negbio', 'Pneumonia_negbio', 'Pneumothorax_negbio', 'Support Devices_negbio'], performance_standard_task='chexpert-classification', performance_standard_metric='auc')
===================================================================================================================

Best performance model has been saved to: [val_chexpert-classification_auc_0_6562_test_chexpert-classification_auc_0_6610_epoch29_02-24-2023 09-40-30_chexpert_without_fix]
The final model has been saved to: [val_chexpert-classification_auc_0_6156_test_chexpert-classification_auc_0_6034_epoch30_02-24-2023 09-50-36_chexpert_without_fix]

===================================================================================================================
```

###

```python
ModelSetup(name='lesion_dsetection_baseline_densenet161', sources=['xrays'], tasks=['lesion-detection'], fusor='element-wise sum', decoder_channels=[128, 64, 32, 16, 8], lesion_label_cols=['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality'], save_early_stop_model=True, record_training_performance=False, backbone='densenet161', optimiser='sgd', lr=0.01, weight_decay=1e-05, sgb_momentum=0.9, image_backbone_pretrained=True, heatmap_backbone_pretrained=False, image_size=512, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], backbone_out_channels=64, batch_size=8, warmup_epochs=0, model_warmup_epochs=0, loss_warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.5, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=False, multiStepLR_milestones=[20, 40, 60, 80, 100], multiStepLR_gamma=0.5, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=False, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False, gt_in_train_till=0, measure_test=True, eval_freq=10, use_iobb=True, iou_thrs=array([0.5]), fiaxtions_mode_input='reporting', fiaxtions_mode_label='reporting', clinical_num=['age', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'acuity'], clinical_cat=['gender'], categorical_col_maps={'gender': 2}, clinical_cat_emb_dim=32, clinical_conv_channels=32, clinical_upsample='deconv', chexpert_label_cols=['Atelectasis_chexpert', 'Cardiomegaly_chexpert', 'Consolidation_chexpert', 'Edema_chexpert', 'Enlarged Cardiomediastinum_chexpert', 'Fracture_chexpert', 'Lung Lesion_chexpert', 'Lung Opacity_chexpert', 'No Finding_chexpert', 'Pleural Effusion_chexpert', 'Pleural Other_chexpert', 'Pneumonia_chexpert', 'Pneumothorax_chexpert', 'Support Devices_chexpert'], negbio_label_cols=['Atelectasis_negbio', 'Cardiomegaly_negbio', 'Consolidation_negbio', 'Edema_negbio', 'Enlarged Cardiomediastinum_negbio', 'Fracture_negbio', 'Lung Lesion_negbio', 'Lung Opacity_negbio', 'No Finding_negbio', 'Pleural Effusion_negbio', 'Pleural Other_negbio', 'Pneumonia_negbio', 'Pneumothorax_negbio', 'Support Devices_negbio'], performance_standard_task='lesion-detection', performance_standard_metric='ap', random_flip=True, use_dynamic_weight=True)

```