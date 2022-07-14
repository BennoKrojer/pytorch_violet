space = {
    'batchsize': [6,16,32,64,128],
    'lr_head': [1e-3, 1e-4, 5e-4, 5e-5],
    'lr_base': [1e-3, 1e-4, 5e-4, 5e-5],
    'decay_head': [1e-3, 1e-2, 5e-3],
    'decay_base': [1e-3, 1e-2, 5e-3],
    'activation': ['gelu', 'relu'],
    'freeze_lang': [True, False],
    'freeze_vision': [True, False],
    'freeze_cross': [False],
    'video_only': [True, False],
    'img_size': [256, 320],
    'dropout': [0.05,0.1,0.15,0.2],
    'aggregation': ['mean_pooling_patches', 'vision_CLS'],
    'max_length': [25, 50, 75, 100],
    'use_clip_txt_cls': [True, False],
    'fuse_cls': [True]
}