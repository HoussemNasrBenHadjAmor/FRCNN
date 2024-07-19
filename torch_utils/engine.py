import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.models.detection.mask_rcnn
from torch_utils import utils
from torch_utils.coco_eval import CocoEvaluator
from torch_utils.coco_utils import get_coco_api_from_dataset
from utils.general import save_validation_results
from utils.metrics import(iou)


def train_one_epoch(
    model, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    train_loss_hist,
    print_freq, 
    scaler=None,
    scheduler=None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # List to store batch losses.
    batch_loss_list = []
    batch_loss_cls_list = []
    batch_loss_box_reg_list = []
    batch_loss_objectness_list = []
    batch_loss_rpn_list = []
    train_box_loss_per_epoch = []
    train_class_loss_per_epoch = []
    train_dfl_loss_per_epoch = [] 

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    step_counter = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        step_counter += 1
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        batch_loss_list.append(loss_value)
        batch_loss_cls_list.append(loss_dict_reduced['loss_classifier'].detach().cpu())
        batch_loss_box_reg_list.append(loss_dict_reduced['loss_box_reg'].detach().cpu())
        batch_loss_objectness_list.append(loss_dict_reduced['loss_objectness'].detach().cpu())
        batch_loss_rpn_list.append(loss_dict_reduced['loss_rpn_box_reg'].detach().cpu())
        train_loss_hist.send(loss_value)
        
        if scheduler is not None:
            scheduler.step(epoch + (step_counter/len(data_loader)))

    train_box_loss_per_epoch.append(sum(batch_loss_box_reg_list)/len(batch_loss_box_reg_list))
    train_class_loss_per_epoch.append(sum(batch_loss_cls_list)/len(batch_loss_cls_list))
    train_dfl_loss_per_epoch.append(sum(batch_loss_rpn_list)/len(batch_loss_rpn_list))

    return metric_logger, train_box_loss_per_epoch, train_class_loss_per_epoch, train_dfl_loss_per_epoch, batch_loss_list, batch_loss_cls_list, batch_loss_box_reg_list, batch_loss_objectness_list, batch_loss_rpn_list


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def calculate_precision(tp, pred_cls):
    tp = np.array(tp)
    pred_cls = np.array(pred_cls)
    precision = sum(tp) / len(pred_cls)
    return precision   

def calculate_recall(tp, fn_count):
    tp_count = sum(tp)  # Total number of true positives
    recall = tp_count / (tp_count + fn_count)
    return recall

@torch.inference_mode()
def evaluate(
    model, 
    data_loader, 
    device, 
    save_valid_preds=False,
    out_dir=None,
    classes=None,
    colors=None
):
    batch_loss_objectness_list=[]
    batch_loss_box_reg_list=[]
    batch_loss_cls_list=[]
    val_box_loss_per_epoch = []
    val_class_loss_per_epoch = []
    val_dfl_loss_per_epoch = []
    tp = []
    conf = []
    pred_cls = []
    target_cls = []
    fn_count = 0

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    counter = 0
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        counter += 1
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Extracting boxes, scores, and labels from outputs
        for output, target in zip(outputs, targets):
            pred_boxes = output['boxes'].cpu().detach().numpy()
            pred_scores = output['scores'].cpu().detach().numpy()
            pred_labels = output['labels'].cpu().detach().numpy()

            gt_boxes = target['boxes'].cpu().detach().numpy()
            gt_labels = target['labels'].cpu().detach().numpy()

            detected = []
            for i, pred_box in enumerate(pred_boxes):
                if len(gt_boxes) == 0:
                    tp.append(0)
                    conf.append(pred_scores[i])
                    pred_cls.append(pred_labels[i])
                    continue

                ious = [iou(pred_box, gt_box) for gt_box in gt_boxes]
                max_iou = max(ious)
                max_iou_idx = ious.index(max_iou)

                if max_iou > 0.5 and gt_labels[max_iou_idx] == pred_labels[i] and max_iou_idx not in detected:
                    tp.append(1)
                    detected.append(max_iou_idx)
                else:
                    tp.append(0)

                conf.append(pred_scores[i])
                pred_cls.append(pred_labels[i])

            target_cls.extend(gt_labels)

            fn_count += len(gt_labels) - len(detected)
    
          # Calculate validation losses
        
        #loss_dict = model(images, targets)
        #losses = sum(loss for loss in loss_dict.values())
        
        # reduce losses over all GPUs for logging purposes
        #loss_dict_reduced = utils.reduce_dict(loss_dict)
        #losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        #loss_value = losses_reduced.item()

        #loss_dict_reduced = utils.reduce_dict(loss_dict)
        #print(f'loss_dict_reduced : {loss_dict_reduced}')
        #batch_loss_cls_list.append(loss_dict_reduced['loss_classifier'].detach().cpu())
        #batch_loss_box_reg_list.append(loss_dict_reduced['loss_box_reg'].detach().cpu())
        #batch_loss_objectness_list.append(loss_dict_reduced['loss_objectness'].detach().cpu())
      
        
        model_time = time.time() - model_time
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        if save_valid_preds and counter == 1:
            # The validation prediction image which is saved to disk
            # is returned here which is again returned at the end of the
            # function for WandB logging.
            val_saved_image = save_validation_results(
                images, outputs, counter, out_dir, classes, colors
            )

    #val_box_loss_per_epoch.append(sum(batch_loss_box_reg_list)/len(batch_loss_box_reg_list))
    #val_class_loss_per_epoch.append(sum(batch_loss_cls_list)/len(batch_loss_cls_list))
    #val_dfl_loss_per_epoch.append(sum(batch_loss_rpn_list)/len(batch_loss_rpn_list))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    category_ids = data_loader.dataset.get_category_ids()
    category_names = data_loader.dataset.get_category_names()

    return coco_evaluator, stats, val_saved_image, category_ids, category_names, tp, conf, pred_cls, target_cls, fn_count 