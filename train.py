import argparse
import os
import random
from typing import Sequence
import numpy as np
from tqdm import tqdm

#Hàm tự định nghĩa
from aug_helper.BatchWiseAug import BatchWiseAug
# from utils.MetricCal import MetricCal
from utils.MetricCalV2 import MetricCalV2
from learning_rate_helper.learning_rate import PiecewiseScheduler, WarmupCosineScheduler
from model_builder.model import Model
from dataset_helper.DatasetLoader import DatasetLoader
from utils.Utilities import Get_Max_Acc, Loading_Checkpoint, Saving_Best, Saving_Checkpoint, Saving_Metric2, YAML_Reader, get_mean_std
# from CBAM_Resnet import Model as CBAM_Resnet
from loss_helper.SaliencyGuidedLoss import SaliencyGuidedLoss

import torch
import torch.nn as nn
import torch.optim as optim

from timm.loss.cross_entropy import SoftTargetCrossEntropy

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argparse example")
    
    # Add arguments
    parser.add_argument(
    "--cfg",
    type=str,
    default="config/DTLHerb_config.yaml",
    help="Config file used to train the model (default: config/default_config.yaml)"
    )
    args = parser.parse_args()
    config = YAML_Reader(args.cfg)
    return config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

features = []
def hook_fn(module, input, output):
    # features.append(output)
    module.feature_maps = output

# feature_maps = None
# def hook_fn(module, input, output):
#     nonlocal feature_maps
#     feature_maps = output

def train(epoch: int, end_epoch: int, batchWiseAug, model, loader, criterion, optimizer, device, num_classes, ):
    model.train()
    metrics = MetricCalV2(num_classes=num_classes, device=device)
    for inputs, masks, targets, has_masks in tqdm(loader, total=len(loader), desc="Training epoch [{0}/{1}]".
                                format(epoch, end_epoch)):

        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if batchWiseAug != None:
            inputs, masks, targets, has_masks = batchWiseAug(inputs, masks, targets)
        has_masks = has_masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        # features.clear()
        outputs = model(inputs)
        # feature_maps = features[0]
        feature_maps = model.get_feature_maps()
        total_loss, cls_loss, bce_align_loss, dice_loss = criterion(outputs, targets, feature_maps, masks, has_masks) #SaliencyGuideLoss trả về 4 tham số
        total_loss.backward()
        optimizer.step()

        metrics.update_train(cls_loss=cls_loss,
                             bce_loss=bce_align_loss,
                             dice_loss=dice_loss,
                             has_masks=has_masks, 
                             outputs=outputs, 
                             targets=targets, 
                             type="soft" if batchWiseAug != None else "hard")
    return metrics

def validate(epoch, end_epoch, model, loader, criterion, device, num_classes):
    model.eval()
    metrics = MetricCalV2(num_classes=num_classes, device=device)
    with torch.no_grad():
        for inputs, _, targets, _ in tqdm(loader, total=len(loader), desc="Validating epoch [{0}/{1}]".
                                format(epoch, end_epoch)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            metrics.update_test(loss=loss, outputs=outputs, targets=targets, type="hard")
    return metrics

def main():
    config = parse_args()


    #Data parameters
    root_path = config["DATASET"]["ROOT_FOLDER"]
    train_path = config["DATASET"]["TRAIN_FOLDER"]
    test_path = config["DATASET"]["TEST_FOLDER"]
    train_mask_path = config["DATASET"]["TRAIN_MASK_FOLDER"]
    test_mask_path = config["DATASET"]["TEST_MASK_FOLDER"]
    CLASSES = sorted([i for i in os.listdir(root_path)])
    mean: Sequence[float] = config["TRAIN"]["DATA"]["MEAN"]
    std: Sequence[float] = config["TRAIN"]["DATA"]["STD"]
    batch_size = config["TRAIN"]["DATA"]["BATCH_SIZE"]
    

    #Training parameters
    img_size = config["TRAIN"]["DATA"]["IMAGE_SIZE"]
    enabled_transform = config["TRAIN"]["TRANSFORM"]
    enabled_batchwise_transform = config["TRAIN"]["AUG"]["ENABLED"]
    begin_epoch = config["TRAIN"]["TRAIN_PARA"]["BEGIN_EPOCH"] 
    end_epoch = config["TRAIN"]["TRAIN_PARA"]["END_EPOCH"]
    resume = config["TRAIN"]["TRAIN_PARA"]["RESUME"]
    early_stopping = int(config["TRAIN"]["TRAIN_PARA"]["EARLY_STOPPING"])
    patience = config["TRAIN"]["TRAIN_PARA"]["PATIENCE"]
    epochs_no_improve = 0
    model_type = int(config["TRAIN"]["TRAIN_PARA"]["MODEL_TYPE"])
    if model_type == 8:
        img_size = [224, 224]

    #Learning_rate
    if model_type not in [8, 9]:
        Learning_rate_para = config["TRAIN"]["LEARNING_RATE"]["PieceWise"]
    else:
        Learning_rate_para = config["TRAIN"]["LEARNING_RATE"]["WarmupCosine"]

    #Optional
    save_checkpoint = config["TRAIN"]["OPTIONAL"]["SAVE_CHECKPOINT"]
    save_best = config["TRAIN"]["OPTIONAL"]["SAVE_BEST"]
    save_metrics = config["TRAIN"]["OPTIONAL"]["SAVE_METRICS"]
    checkpoint_path = config["TRAIN"]["OPTIONAL"]["CHECKPOINT_PATH"]
    best_path = config["TRAIN"]["OPTIONAL"]["BEST_PATH"]
    metrics_path = config["TRAIN"]["OPTIONAL"]["METRICS_PATH"]
    
    set_seed()
    
    if mean is None or std is None:
        print("Calculating mean and std")
        mean, std = get_mean_std(train_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data = DatasetLoader(
        img_path=train_path,
        mask_path=train_mask_path,
        std=std,
        mean=mean,
        img_size=img_size,
        batch_size=batch_size,
        transform = enabled_transform)
    test_data = DatasetLoader(
        img_path=test_path,
        mask_path=test_mask_path,
        std=std,
        mean=mean,
        img_size=img_size,
        batch_size=batch_size)

    training_loader = train_data.dataset_loader("train")
    testing_loader = test_data.dataset_loader("test")

    batchWiseAug = None
    if enabled_batchwise_transform:
        batchWiseAug = BatchWiseAug(config=config, num_classes=len(CLASSES))

    model = Model(len(CLASSES), model_type).to(device)
    # hook_handle = model.model.layer4.register_forward_hook(hook_fn)
    # model.model.layer4.feature_maps = None
    # hook_handle = model.model.layer4.register_forward_hook(hook_fn)
    hook_handle = model.register_hook(hook_fn)

    eval_criterion = nn.CrossEntropyLoss()
    # train_criterion = nn.CrossEntropyLoss()
    # if enabled_batchwise_transform:
    #     train_criterion = SoftTargetCrossEntropy()
    train_criterion = SaliencyGuidedLoss(type="train", 
                                         enabled_batchwise_transform=enabled_batchwise_transform, 
                                         alpha=1,
                                         beta=0.05,
                                         gamma=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=Learning_rate_para["MAX_LR"], weight_decay=1e-2)

    if model_type not in [8, 9]:
        lr_schedule = PiecewiseScheduler(
            start_lr=Learning_rate_para["START_LR"],
            max_lr=Learning_rate_para["MAX_LR"],
            min_lr=Learning_rate_para["MIN_LR"],
            rampup_epochs=Learning_rate_para["RAMPUP_EPOCHS"],
            sustain_epochs=Learning_rate_para["SUSTAIN_EPOCHS"],
            exp_decay=Learning_rate_para["EXP_DECAY"]
        )
        print("Training using PiecewiseScheduler")
    else:
        lr_schedule = WarmupCosineScheduler(
            start_lr=Learning_rate_para["START_LR"],
            max_lr=Learning_rate_para["MAX_LR"],
            min_lr=Learning_rate_para["MIN_LR"],
            rampup_epochs=Learning_rate_para["RAMPUP_EPOCHS"],
            sustain_epochs=Learning_rate_para["SUSTAIN_EPOCHS"],
            total_epochs=end_epoch
        )
        print("Training using WarmupCosineScheduler")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    best_acc = 0

    if resume == True:
        begin_epoch = Loading_Checkpoint(path=checkpoint_path,
                                         model=model,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         device=device)
        best_acc = Get_Max_Acc(metrics_path)

    for epoch in range(begin_epoch, end_epoch):
        train_metrics = train(epoch, 
                                end_epoch, 
                                batchWiseAug=batchWiseAug,
                                model=model, 
                                loader=training_loader, 
                                criterion=train_criterion, 
                                optimizer=optimizer, 
                                device=device,
                                num_classes=len(CLASSES))
        train_loss, train_acc = train_metrics.overall_loss(alpha=1,beta=0.05,gamma=0.01), train_metrics.avg_accuracy
        scheduler.step()
        print()
        hook_handle.remove() #Vô hiệu hóa hook khi validate và tái khởi động khi train
        val_metrics = validate(epoch, end_epoch, model, testing_loader, eval_criterion, device, num_classes=len(CLASSES))
        hook_handle = model.register_hook(hook_fn)
        val_loss, val_acc = val_metrics.avg_cls_loss, val_metrics.avg_accuracy
        print()

        if save_checkpoint == True:
            Saving_Checkpoint(epoch=epoch, 
                            model=model, 
                            optimizer=optimizer, 
                            scheduler=scheduler,
                            last_epoch=epoch, 
                            path=checkpoint_path)

        print("Epoch [{0}/{1}]: Training loss: {2}, Training Acc: {3}%".
            format(epoch, end_epoch, train_loss, round(train_acc * 100.0, 2)))
        print("Epoch [{0}/{1}]: Validation loss: {2}, Validation Acc: {3}%".
            format(epoch, end_epoch, val_loss, round(val_acc * 100.0, 2)))
        if val_acc > best_acc:
            if save_best == True:
                print("Validation accuracy increase from {0}% to {1}% at epoch {2}. Saving best result".
                    format(round(best_acc * 100.0, 2), round(val_acc * 100.0, 2),  epoch))
                Saving_Best(model, best_path)
            else:
                print("Validation accuracy increase from {0}% to {1}% at epoch {2}".
                    format(round(best_acc * 100.0, 2), round(val_acc * 100.0, 2),  epoch))
            best_acc = val_acc
            epochs_no_improve = 0  # reset patience
        else:
            epochs_no_improve += 1
        if save_metrics:
            Saving_Metric2(epoch=epoch, 
                           train_cls_loss=train_metrics.avg_cls_loss,
                           train_bce_loss=train_metrics.avg_bce_loss,
                           train_dice_loss=train_metrics.avg_dice_loss,
                           train_overall_loss=train_loss,
                           train_acc=train_acc,
                           train_precision=train_metrics.precision_macro,
                           train_recall=train_metrics.recall_macro,
                           train_f1=train_metrics.f1_macro, 
                           val_loss=val_loss,
                           val_acc=val_acc,
                           val_precision=val_metrics.precision_macro,
                           val_recall=val_metrics.recall_macro,
                           val_f1=val_metrics.f1_macro, 
                           path=metrics_path)
        if epochs_no_improve >= patience and early_stopping == True:
            print("Early stopping triggered at epoch {0}".format(epoch))
            break
        print()
    hook_handle.remove()

    
if __name__ == '__main__':
    main()
