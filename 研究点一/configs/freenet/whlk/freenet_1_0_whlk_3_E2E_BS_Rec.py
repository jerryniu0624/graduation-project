config = dict(
    model=dict(
        type='FreeNetE2E_BA_Rec',
        params=dict(
            in_channels=270,
            out_channels=3,
            hidden_channels=20,
            rec_bs = 1,
            minBand = 400,
            maxBand = 1000, # 这里不对
            nBandDataset = 270,
            dataset = 'LK',
            num_classes=6,
            block_channels=(96, 128, 192, 256),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    data=dict(
        train=dict(
            type='NewLongKouLoader',
            params=dict(
                training=True,
                num_workers=0,
                image_mat_path='./WHU-Hi-LongKou/WHU_Hi_LongKou.mat',
                gt_mat_path='./WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat',
                num_train_samples_per_class=10,
                sub_minibatch=20,
                slic = False,
                n_segments = 1000,
                compactness = 1
            )
        ),
        test=dict(
            type='NewLongKouLoader',
            params=dict(
                training=False,
                num_workers=0,
                image_mat_path='./WHU-Hi-LongKou/WHU_Hi_LongKou.mat',
                gt_mat_path='./WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat',
                num_train_samples_per_class=0,
                sub_minibatch=20,
                slic = False,
                n_segments = 1000,
                compactness = 1
            )
        )
    ),
    optimizer=dict(
        type='sgd',
        params=dict(
            momentum=0.9,
            weight_decay=0.001
        )
    ),
    learning_rate=dict(
        type='poly',
        params=dict(
            base_lr=0.01,
            power=0.9,
            max_iters=2000),
    ),
    train=dict(
        forward_times=1,
        num_iters=2000,
        eval_per_epoch=True,
        summary_grads=False,
        summary_weights=False,
        eval_after_train=True,
        resume_from_last=False,
    ),
    test=dict(
        draw=dict(
            image_size=(512, 217),
            palette=[
                0, 0, 0,
                192, 192, 192,
                0, 255, 1,
                0, 255, 255,
                0, 128, 1,
                255, 0, 254,
                165, 82, 40,
                129, 0, 127,
                255, 0, 0,
                255, 255, 0, ]
        )
    ),
)
