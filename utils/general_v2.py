import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')

# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation mAP @0.5:0.95 IoU higher than the previous highest, then save the
    model state.
    """
    def __init__(
        self, best_valid_map=float(0)
    ):
        self.best_valid_map = best_valid_map
        
    def __call__(
        self, 
        model, 
        current_valid_map, 
        epoch, 
        OUT_DIR,
        config,
        model_name
    ):
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            print(f"\nBEST VALIDATION mAP: {self.best_valid_map}")
            print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'config': config,
                'model_name': model_name
                }, f"{OUT_DIR}/best_model.pth")

def show_tranformed_image(train_loader, device, classes, colors):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    
    """
    if len(train_loader) > 0:
        for i in range(2):
            images, targets = next(iter(train_loader))
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            # Get all the predicited class names.
            pred_classes = [classes[i] for i in targets[i]['labels'].cpu().numpy()]
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
            for box_num, box in enumerate(boxes):
                class_name = pred_classes[box_num]
                color = colors[classes.index(class_name)]
                cv2.rectangle(sample,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            color, 2,
                            cv2.LINE_AA)
                cv2.putText(sample, classes[labels[box_num]], 
                            (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, color, 2, cv2.LINE_AA)
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def save_box_loss(loss, OUT_DIR, title):
    figure = plt.figure(figsize=(10, 5))
    train_ax = figure.add_subplot(1, 1, 1)
    train_ax.plot(loss, label='Train Loss')
    train_ax.set_title(title)
    train_ax.set_xlabel('Epochs')
    train_ax.set_ylabel('Loss')
    train_ax.legend()
    figure.savefig(f"{OUT_DIR}/{title}.png")
    plt.close(figure)

def save_cls_loss(loss, OUT_DIR, title):
    figure = plt.figure(figsize=(10, 5))
    train_ax = figure.add_subplot(1, 1, 1)
    train_ax.plot(loss, label='Train Loss')
    train_ax.set_title(title)
    train_ax.set_xlabel('Epochs')
    train_ax.set_ylabel('Loss')
    train_ax.legend()
    figure.savefig(f"{OUT_DIR}/{title}.png")
    plt.close(figure)

def save_dfl_loss(loss, OUT_DIR, title):
    figure = plt.figure(figsize=(10, 5))
    train_ax = figure.add_subplot(1, 1, 1)
    train_ax.plot(loss, label='Train Loss')
    train_ax.set_title(title)
    train_ax.set_xlabel('Epochs')
    train_ax.set_ylabel('Loss')
    train_ax.legend()
    figure.savefig(f"{OUT_DIR}/{title}.png")
    plt.close(figure)

def save_precision(loss, OUT_DIR, title):
    figure = plt.figure(figsize=(10, 5))
    train_ax = figure.add_subplot(1, 1, 1)
    train_ax.plot(loss, label='Train Loss')
    train_ax.set_title(title)
    train_ax.set_xlabel('Epochs')
    train_ax.set_ylabel('Loss')
    train_ax.legend()
    figure.savefig(f"{OUT_DIR}/{title}.png")
    plt.close(figure)

def save_recal(loss, OUT_DIR, title):
    figure = plt.figure(figsize=(10, 5))
    train_ax = figure.add_subplot(1, 1, 1)
    train_ax.plot(loss, label='Train Loss')
    train_ax.set_title(title)
    train_ax.set_xlabel('Epochs')
    train_ax.set_ylabel('Loss')
    train_ax.legend()
    figure.savefig(f"{OUT_DIR}/{title}.png")
    plt.close(figure)

def save_map50_95(loss, OUT_DIR, title):
    figure = plt.figure(figsize=(10, 5))
    train_ax = figure.add_subplot(1, 1, 1)
    train_ax.plot(loss, label='Train Loss')
    train_ax.set_title(title)
    train_ax.set_xlabel('Epochs')
    train_ax.set_ylabel('Loss')
    train_ax.legend()
    figure.savefig(f"{OUT_DIR}/{title}.png")
    plt.close(figure)

def save_map50(loss, OUT_DIR, title):
    figure = plt.figure(figsize=(10, 5))
    train_ax = figure.add_subplot(1, 1, 1)
    train_ax.plot(loss, label='Train Loss')
    train_ax.set_title(title)
    train_ax.set_xlabel('Epochs')
    train_ax.set_ylabel('Loss')
    train_ax.legend()
    figure.savefig(f"{OUT_DIR}/{title}.png")
    plt.close(figure)        

def extract_precision_recall(coco_evaluator, category_ids, category_names):
    precision = coco_evaluator.coco_eval['bbox'].eval['precision']
    recalls = coco_evaluator.coco_eval['bbox'].eval['recall']

    pr_data = {}
    for idx, cat_id in enumerate(category_ids):
        if cat_id == 0:  # Skip background class
            continue

        # Ensure index is within bounds
        if idx >= precision.shape[2]:
            print(f"Skipping category index {idx} as it is out of bounds.")
            continue

        # Extract precision and recall values for the category
        precision_at_cat = precision[:, :, idx, 0, -1].flatten()
        recall_at_cat = recalls[:, idx, 0, -1]

        # Filter out invalid values
        precision_at_cat = precision_at_cat[precision_at_cat > -1]
        recall_at_cat = recall_at_cat[recall_at_cat > -1]

        if len(precision_at_cat) > 0 and len(recall_at_cat) > 0:
            pr_data[category_names[cat_id]["name"]] = {
                'precision': precision_at_cat,
                'recall': recall_at_cat
            }
    
    return pr_data


def save_precision_confidence_curve(coco_evaluator, category_ids, category_names, out_dir, title):
    precision = coco_evaluator.coco_eval['bbox'].eval['precision']
    print(f'precision : {precision}')
    recalls = coco_evaluator.coco_eval['bbox'].eval['scores']
    print(f'recalls : {recalls}')
    
    plt.figure(figsize=(10, 8))

    # Precision is a (T, R, K, A, M) array.
    # T: number of IoU thresholds.
    # R: number of recall thresholds.
    # K: number of categories.
    # A: number of areas (this is usually 1).
    # M: number of max detections (this is usually 1).

    print(category_names[0])
    
    # We will plot the precision-recall curve for the first area and max detection.
    for idx, cat_id in enumerate(category_ids):
        if cat_id == 0:  # Skip background class
            continue

        # Adjust index to skip background class
        idx = cat_id - 1    
        
        precision_at_cat = precision[:, :, idx, 0, -1]
        
        scores_at_cat = recalls[:, idx, 0, -1]
        
        ap = precision_at_cat.mean(axis=1)  # average precision over IoU thresholds

        ap = ap[ap > -1]  # filter out invalid values

        if len(ap) > 0:
            plt.plot(scores_at_cat, ap, label=f'{category_names[cat_id]["name"]}')

    plt.xlabel('Confidence')
    plt.ylabel('Average Precision')
    plt.title(title)
    plt.legend()
    
    plt.savefig(f"{out_dir}/{title}.png")
    plt.close()    

def save_precision_recall_curve(pr_data, out_dir, title):
    plt.figure(figsize=(10, 8))

    # Plot each category's precision-recall curve
    for cat_name, data in pr_data.items():
        precision = data['precision']
        recall = data['recall']
        
        # Ensure the lengths match for plotting
        min_len = min(len(precision), len(recall))
        precision = precision[:min_len]
        recall = recall[:min_len]

        plt.plot(recall, precision, label=f'{cat_name} (AP={precision.mean():.3f})')
    
    # Plot the combined precision-recall curve
    all_precisions = np.concatenate([data['precision'] for data in pr_data.values()])
    all_recalls = np.concatenate([data['recall'] for data in pr_data.values()])

    min_len = min(len(all_precisions), len(all_recalls))
    all_precisions = all_precisions[:min_len]
    all_recalls = all_recalls[:min_len]

    plt.plot(all_recalls, all_precisions, label=f'all classes (mAP@0.5={all_precisions.mean():.3f})', linewidth=2, color='blue')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    
    plt.savefig(f"{out_dir}/{title}.png")
    plt.show()





def save_loss_plot(
    OUT_DIR, 
    train_loss_list, 
    x_label='iterations',
    y_label='train loss',
    save_name='train_loss_iter'
):
    """
    Function to save both train loss graph.
    
    :param OUT_DIR: Path to save the graphs.
    :param train_loss_list: List containing the training loss values.
    """
    figure_1 = plt.figure(figsize=(10, 7), num=1, clear=True)
    train_ax = figure_1.add_subplot()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel(x_label)
    train_ax.set_ylabel(y_label)
    figure_1.savefig(f"{OUT_DIR}/{save_name}.png")
    print('SAVING PLOTS COMPLETE...')
    # plt.close('all')

def save_mAP(OUT_DIR, map_05, map):
    """
    Saves the mAP@0.5 and mAP@0.5:0.95 per epoch.

    :param OUT_DIR: Path to save the graphs.
    :param map_05: List containing mAP values at 0.5 IoU.
    :param map: List containing mAP values at 0.5:0.95 IoU.
    """
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(
        map_05, color='tab:orange', linestyle='-', 
        label='mAP@0.5'
    )
    ax.plot(
        map, color='tab:red', linestyle='-', 
        label='mAP@0.5:0.95'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')
    ax.legend()
    figure.savefig(f"{OUT_DIR}/map.png")
    # plt.close('all')

def visualize_mosaic_images(boxes, labels, image_resized, classes):
    print(boxes)
    print(labels)
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
    for j, box in enumerate(boxes):
        color = (0, 255, 0)
        classn = labels[j]
        cv2.rectangle(image_resized,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color, 2)
        cv2.putText(image_resized, classes[classn], 
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                    2, lineType=cv2.LINE_AA)
    cv2.imshow('Mosaic', image_resized)
    cv2.waitKey(0)

def save_model(
    epoch, 
    model, 
    optimizer, 
    train_loss_list,
    train_loss_list_epoch, 
    val_map,
    val_map_05,
    OUT_DIR,
    config,
    model_name
):
    """
    Function to save the trained model till current epoch, or whenever called.
    Saves many other dictionaries and parameters as well helpful to resume training.
    May be larger in size.

    :param epoch: The epoch number.
    :param model: The neural network model.
    :param optimizer: The optimizer.
    :param optimizer: The train loss history.
    :param train_loss_list_epoch: List containing loss for each epoch.
    :param val_map: mAP for IoU 0.5:0.95.
    :param val_map_05: mAP for IoU 0.5.
    :param OUT_DIR: Output directory to save the model.
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_list': train_loss_list,
                'train_loss_list_epoch': train_loss_list_epoch,
                'val_map': val_map,
                'val_map_05': val_map_05,
                'config': config,
                'model_name': model_name
                }, f"{OUT_DIR}/last_model.pth")

def save_model_state(model, OUT_DIR, config, model_name):
    """
    Saves the model state dictionary only. Has a smaller size compared 
    to the the saved model with all other parameters and dictionaries.
    Preferable for inference and sharing.

    :param model: The neural network model.
    :param OUT_DIR: Output directory to save the model.
    """
    torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'model_name': model_name
                }, f"{OUT_DIR}/last_model_state.pth")

def denormalize(x, mean=None, std=None):
    # Shape of x here should be [B, 3, H, W].
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    # Returns tensor of shape [B, 3, H, W].
    return torch.clamp(x, 0, 1)

def save_validation_results(images, detections, counter, out_dir, classes, colors):
    """
    Function to save validation results.
    :param images: All the images from the current batch.
    :param detections: All the detection results.
    :param counter: Step counter for saving with unique ID.
    """
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    image_list = [] # List to store predicted images to return.
    for i, detection in enumerate(detections):
        image_c = images[i].clone()
        # image_c = denormalize(image_c, IMG_MEAN, IMG_STD)
        image_c = image_c.detach().cpu().numpy().astype(np.float32)
        image = np.transpose(image_c, (1, 2, 0))

        image = np.ascontiguousarray(image, dtype=np.float32)

        scores = detection['scores'].cpu().numpy()
        labels = detection['labels']
        bboxes = detection['boxes'].detach().cpu().numpy()
        boxes = bboxes[scores >= 0.5].astype(np.int32)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Get all the predicited class names.
        pred_classes = [classes[i] for i in labels.cpu().numpy()]
        for j, box in enumerate(boxes):
            class_name = pred_classes[j]
            color = colors[classes.index(class_name)]
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, 2, lineType=cv2.LINE_AA
            )
            cv2.putText(image, class_name, 
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                    2, lineType=cv2.LINE_AA)
        cv2.imwrite(f"{out_dir}/image_{i}_{counter}.jpg", image*255.)
        image_list.append(image[:, :, ::-1])
    return image_list

def set_infer_dir():
    """
    This functions counts the number of inference directories already present
    and creates a new one in `outputs/inference/`. 
    And returns the directory path.
    """
    if not os.path.exists('outputs/inference'):
        os.makedirs('outputs/inference')
    num_infer_dirs_present = len(os.listdir('outputs/inference/'))
    next_dir_num = num_infer_dirs_present + 1
    new_dir_name = f"outputs/inference/res_{next_dir_num}"
    os.makedirs(new_dir_name, exist_ok=True)
    return new_dir_name

def set_training_dir(dir_name=None):
    """
    This functions counts the number of training directories already present
    and creates a new one in `outputs/training/`. 
    And returns the directory path.
    """
    if not os.path.exists('outputs/training'):
        os.makedirs('outputs/training')
    if dir_name:
        new_dir_name = f"outputs/training/{dir_name}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name
    else:
        num_train_dirs_present = len(os.listdir('outputs/training/'))
        next_dir_num = num_train_dirs_present + 1
        new_dir_name = f"outputs/training/res_{next_dir_num}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name