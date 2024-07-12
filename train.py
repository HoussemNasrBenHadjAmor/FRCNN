"""
USAGE

# Training with Faster RCNN ResNet50 FPN model without mosaic or any other augmentation:
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --config data_configs/voc.yaml --no-mosaic --batch-size 4

# Training on ResNet50 FPN with custom project folder name with mosaic augmentation (ON by default):
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --config data_configs/voc.yaml --project-name resnet50fpn_voc --batch-size 4

# Training on ResNet50 FPN with custom project folder name with mosaic augmentation (ON by default) and added training augmentations:
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --use-train-aug --config data_configs/voc.yaml --project-name resnet50fpn_voc --batch-size 4
"""

from torch_utils.engine import (train_one_epoch, evaluate)

from utils.metrics import (ap_per_class)

from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
from models.create_fasterrcnn_model import create_model
from utils.general import (
    set_training_dir, Averager, 
    save_model, save_loss_plot, save_box_loss, save_cls_loss, save_dfl_loss, save_precision, save_recal, save_map50, save_map50_95, save_precision_recall_curve, save_precision_confidence_curve, save_recall_confidence_curve, save_f1_confidence_curve,
    show_tranformed_image,
    save_mAP, save_model_state, SaveBestModel
)
from utils.logging import (
    set_log, 
    coco_log
)

import torch
import argparse
import yaml
import numpy as np
import sys

torch.multiprocessing.set_sharing_strategy('file_system')

# For same annotation colors each time.
np.random.seed(42)

def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', default='fasterrcnn_resnet50_fpn',
        help='name of the model'
    )
    parser.add_argument(
        '-c', '--config', default=None,
        help='path to the data config file'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-e', '--epochs', default=5, type=int,
        help='number of epochs to train for'
    )
    parser.add_argument(
        '-j', '--workers', default=4, type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '-b', '--batch-size', dest='batch_size', default=4, type=int, 
        help='batch size to load the data'
    )
    parser.add_argument(
        '-ims', '--img-size', dest='img_size', default=640, type=int, 
        help='image size to feed to the network'
    )
    parser.add_argument(
        '-pn', '--project-name', default=None, type=str, dest='project_name',
        help='training result dir name in outputs/training/, (default res_#)'
    )
    parser.add_argument(
        '-vt', '--vis-transformed', dest='vis_transformed', action='store_true',
        help='visualize transformed images fed to the network'
    )
    parser.add_argument(
        '-nm', '--no-mosaic', dest='no_mosaic', action='store_false',
        help='pass this to not to use mosaic augmentation'
    )
    parser.add_argument(
        '-uta', '--use-train-aug', dest='use_train_aug', action='store_true',
        help='whether to use train augmentation, uses some advanced augmentation \
              that may make training difficult when used with mosaic'
    )
    parser.add_argument(
        '-ca', '--cosine-annealing', dest='cosine_annealing', action='store_true',
        help='use cosine annealing warm restarts'
    )
    parser.add_argument(
        '-w', '--weights', default=None, type=str,
        help='path to model weights if using pretrained weights'
    )
    parser.add_argument(
        '-r', '--resume-training', dest='resume_training', action='store_true',
        help='whether to resume training, if true, \
             loads previous training plots and epochs \
             and also loads the otpimizer state dictionary'
    )
    args = vars(parser.parse_args())
    return args

def main(args):
    # Initialize W&B with project name.
    # wandb_init(name=args['project_name'])
    # Load the data configurations
    with open(args['config']) as file:
        data_configs = yaml.safe_load(file)
    
    # Settings/parameters/constants.
    TRAIN_DIR_IMAGES = data_configs['TRAIN_DIR_IMAGES']
    TRAIN_DIR_LABELS = data_configs['TRAIN_DIR_LABELS']
    VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
    VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = data_configs['NC']
    NUM_WORKERS = args['workers']
    DEVICE = args['device']
    NUM_EPOCHS = args['epochs']
    SAVE_VALID_PREDICTIONS = data_configs['SAVE_VALID_PREDICTION_IMAGES']
    BATCH_SIZE = args['batch_size']
    VISUALIZE_TRANSFORMED_IMAGES = args['vis_transformed']
    OUT_DIR = set_training_dir(args['project_name'])
    COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))
    # Set logging file.
    set_log(OUT_DIR)
    # writer = set_summary_writer(OUT_DIR)

    # Model configurations
    IMAGE_WIDTH = args['img_size']
    IMAGE_HEIGHT = args['img_size']
    
    train_dataset = create_train_dataset(
        TRAIN_DIR_IMAGES, TRAIN_DIR_LABELS,
        IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES,
        use_train_aug=args['use_train_aug'],
        mosaic=args['no_mosaic']
    )
    valid_dataset = create_valid_dataset(
        VALID_DIR_IMAGES, VALID_DIR_LABELS, 
        IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES
    )
    train_loader = create_train_loader(train_dataset, BATCH_SIZE, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    if VISUALIZE_TRANSFORMED_IMAGES:
        show_tranformed_image(train_loader, DEVICE, CLASSES, COLORS)

    # Initialize the Averager class.
    train_loss_hist = Averager()
    # Train and validation loss lists to store loss values of all
    # iterations till ena and plot graphs for all iterations.
    train_loss_list = []
    loss_cls_list = []
    loss_box_reg_list = []
    loss_objectness_list = []
    loss_rpn_list = []
    train_loss_list_epoch = []
    val_map_05 = []
    val_map = []
    all_train_box_loss_per_epoch = []
    all_train_cls_loss_per_epochs = []
    all_train_dfl_loss_per_epoch = []
    val_precision = []
    val_recall = []
    val_mAP50 = []
    val_mAP50_95 = []
    start_epochs = 0

    if args['weights'] is None:
        print('Building model from scratch...')
        build_model = create_model[args['model']]
        model = build_model(num_classes=NUM_CLASSES, pretrained=True)

    # Load pretrained weights if path is provided.
    if args['weights'] is not None:
        print('Loading pretrained weights...')
        
        # Load the pretrained checkpoint.
        checkpoint = torch.load(args['weights'], map_location=DEVICE) 
        keys = list(checkpoint['model_state_dict'].keys())
        ckpt_state_dict = checkpoint['model_state_dict']
        # Get the number of classes from the loaded checkpoint.
        old_classes = ckpt_state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]

        # Build the new model with number of classes same as checkpoint.
        build_model = create_model[args['model']]
        model = build_model(num_classes=old_classes)
        # Load weights.
        model.load_state_dict(ckpt_state_dict)

        # Change output features for class predictor and box predictor
        # according to current dataset classes.
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
            in_features=in_features, out_features=NUM_CLASSES, bias=True
        )
        model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
            in_features=in_features, out_features=NUM_CLASSES*4, bias=True
        )

        if args['resume_training']:
            print('RESUMING TRAINING...')
            # Update the starting epochs, the batch-wise loss list, 
            # and the epoch-wise loss list.
            if checkpoint['epoch']:
                start_epochs = checkpoint['epoch']
                print(f"Resuming from epoch {start_epochs}...")
            if checkpoint['train_loss_list']:
                print('Loading previous batch wise loss list...')
                train_loss_list = checkpoint['train_loss_list']
            if checkpoint['train_loss_list_epoch']:
                print('Loading previous epoch wise loss list...')
                train_loss_list_epoch = checkpoint['train_loss_list_epoch']
            if checkpoint['val_map']:
                print('Loading previous mAP list')
                val_map = checkpoint['val_map']
            if checkpoint['val_map_05']:
                val_map_05 = checkpoint['val_map_05']
        
    print(model)
    model = model.to(DEVICE)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Get the model parameters.
    params = [p for p in model.parameters() if p.requires_grad]
    # Define the optimizer.
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005)
    if args['resume_training']: 
        # LOAD THE OPTIMIZER STATE DICTIONARY FROM THE CHECKPOINT.
        print('Loading optimizer state dictionary...')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if args['cosine_annealing']:
        # LR will be zero as we approach `steps` number of epochs each time.
        # If `steps = 5`, LR will slowly reduce to zero every 5 epochs.
        steps = NUM_EPOCHS + 10
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=steps,
            T_mult=1,
            verbose=False
        )
    else:
        scheduler = None

    save_best_model = SaveBestModel()

    for epoch in range(start_epochs, NUM_EPOCHS):
        train_loss_hist.reset()

        _,  train_box_loss_per_epoch, \
             train_cls_loss_per_epochs, \
             train_dfl_loss_per_epoch, \
             batch_loss_list, \
             batch_loss_cls_list, \
             batch_loss_box_reg_list, \
             batch_loss_objectness_list, \
             batch_loss_rpn_list = train_one_epoch(
            model, 
            optimizer, 
            train_loader, 
            DEVICE, 
            epoch, 
            train_loss_hist,
            print_freq=100,
            scheduler=scheduler
        )

        coco_evaluator, stats, val_pred_image, category_ids, category_names, tp, conf, pred_cls, target_cls = evaluate(
            model, 
            valid_loader, 
            device=DEVICE,
            save_valid_preds=SAVE_VALID_PREDICTIONS,
            out_dir=OUT_DIR,
            classes=CLASSES,
            colors=COLORS
        )

        # Append the current epoch's batch-wise losses to the `train_loss_list`.
        train_loss_list.extend(batch_loss_list)
        loss_cls_list.extend(batch_loss_cls_list)
        loss_box_reg_list.extend(batch_loss_box_reg_list)
        loss_objectness_list.extend(batch_loss_objectness_list)
        loss_rpn_list.extend(batch_loss_rpn_list)
        # Append curent epoch's average loss to `train_loss_list_epoch`.
        val_map_05.append(stats[1])
        val_map.append(stats[0])
        val_precision.append(stats[0])
        val_recall.append(stats[8])
        val_mAP50.append(stats[1])
        val_mAP50_95.append(stats[0])
        all_train_box_loss_per_epoch.extend(train_box_loss_per_epoch)
        all_train_cls_loss_per_epochs.extend(train_cls_loss_per_epochs)
        all_train_dfl_loss_per_epoch.extend(train_dfl_loss_per_epoch)
        
        np_tp = np.array(tp, dtype=bool)
        np_tp = np_tp[:, np.newaxis]
        ap_per_class(category_names,np_tp, np.array(conf), np.array(pred_cls), np.array(target_cls), plot=True , save_dir = OUT_DIR)
        

        # Save box loss for each epoch
        save_box_loss(all_train_box_loss_per_epoch, OUT_DIR, title='Box_Regression_Loss_Per_Epoch')
        # Save class loss for each epoch 
        save_cls_loss(all_train_cls_loss_per_epochs, OUT_DIR, title='Cls_Loss_Per_Epoch')
        # Save dfl (Focal Loss) loss for each epoch
        save_dfl_loss(all_train_dfl_loss_per_epoch, OUT_DIR, title='Dfl_Loss_Per_Epoch')
        # Save precision for each epoch
        save_precision(val_precision, OUT_DIR, title='Metrics-precision(B)')
        # Save recal for each epoch
        save_recal(val_recall, OUT_DIR, title='Metrics-recal(B)')
        # Save mAP50  for each epoch
        save_map50(val_mAP50, OUT_DIR, title='Metrics-mAP50(B)')
        # Save mAP50_95  for each epoch
        save_map50_95(val_mAP50_95, OUT_DIR, title='Metrics-mAP50-95(B)')
        # Save precision recall curve for each epoch
        #save_precision_recall_curve(coco_evaluator, category_ids, category_names, OUT_DIR, 'Precision-Recall Curve')
        # Save precision confidence curve for each epoch
        #save_precision_confidence_curve(coco_evaluator, category_ids, category_names, OUT_DIR, 'Precision-Confidence Curve')
        # Save recall confidence curve for each epoch
        #save_recall_confidence_curve(coco_evaluator, category_ids, category_names, OUT_DIR, 'Recall-Confidence Curve')
        # Save f1 confidence curve for each epoch
        #save_f1_confidence_curve(coco_evaluator, category_ids, category_names, OUT_DIR, 'F1-Confidence Curve')
        # Save loss plot for epoch-wise list.
        save_loss_plot(
            OUT_DIR, 
            train_loss_list_epoch,
            'epochs',
            'train loss',
            save_name='train_loss_epoch' 
        )

        save_loss_plot(
            OUT_DIR, 
            loss_cls_list, 
            'iterations', 
            'loss cls',
            save_name='loss_cls'
        )

        save_loss_plot(
            OUT_DIR, 
            loss_box_reg_list, 
            'iterations', 
            'loss bbox reg',
            save_name='loss_bbox_reg'
        )

        save_loss_plot(
            OUT_DIR,
            loss_objectness_list,
            'iterations',
            'loss obj',
            save_name='loss_obj'
        )

        save_loss_plot(
            OUT_DIR,
            loss_rpn_list,
            'iterations',
            'loss rpn bbox',
            save_name='loss_rpn_bbox'
        )

        # Save mAP plots.
        save_mAP(OUT_DIR, val_map_05, val_map)

        coco_log(OUT_DIR, stats)

        # Save the current epoch model state. This can be used 
        # to resume training. It saves model state dict, number of
        # epochs trained for, optimizer state dict, and loss function.
        save_model(
            epoch, 
            model, 
            optimizer, 
            train_loss_list, 
            train_loss_list_epoch,
            val_map,
            val_map_05,
            OUT_DIR,
            data_configs,
            args['model']
        )
        # Save the model dictionary only for the current epoch.
        save_model_state(model, OUT_DIR, data_configs, args['model'])
        # Save best model if the current mAP @0.5:0.95 IoU is
        # greater than the last hightest.
        save_best_model(
            model, 
            val_map[-1], 
            epoch, 
            OUT_DIR,
            data_configs,
            args['model']
        )

if __name__ == '__main__':
    args = parse_opt()
    main(args)