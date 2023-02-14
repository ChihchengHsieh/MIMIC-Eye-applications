
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

