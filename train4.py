# Dùng riêng cho việc kết hợp Mask và Hier

import argparse
import os
import random
from typing import Sequence
import numpy as np
from tqdm import tqdm

#Hàm tự định nghĩa
from aug_helper.Aug_Hier.BatchWiseAug import BatchWiseAug
from loss_helper.HierarchyGuidedLoss import HierarchyGuidedLoss
# from utils.MetricCal import MetricCal
from utils.MetricCal_Hier import MetricCal_Hier
from learning_rate_helper.learning_rate import PiecewiseScheduler, WarmupCosineScheduler
# from model_builder.baseline import Model
from model_builder.baseline_Hier import Model
from dataset_helper.DatasetLoaderHierMask import DatasetLoader
from utils.Utilities import Get_Max_Acc, Loading_Checkpoint, Saving_Best, Saving_Checkpoint, Saving_Metric3, YAML_Reader, get_mean_std
# from CBAM_Resnet import Model as CBAM_Resnet

import torch
import torch.nn as nn
import torch.optim as optim


# num_classes = {'Species':200 , 'Genus': 125, 'Family': 36, 'Order': 13}
# consistent_list = ["Order2Family", "Family2Genus", "Genus2Species"]
def parse_args():
    parser = argparse.ArgumentParser(description="A simple argparse example")
    
    # Add arguments
    parser.add_argument(
    "--cfg",
    type=str,
    default="config/DTLHerb_Hier_config.yaml",
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



def train(epoch: int, end_epoch: int, batchWiseAug, model, loader, criterion, optimizer, device, num_classes, hier_matrixs, consistent_list):
    model.train()
    metrics = MetricCal_Hier(num_classes=num_classes,consistent_list=consistent_list , device=device)
    for inputs, targets in tqdm(loader, total=len(loader), desc="Training epoch [{0}/{1}]".
                                format(epoch, end_epoch)):

        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if batchWiseAug != None:
            inputs, targets= batchWiseAug(inputs, targets)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        classification_loss, each_classification_loss, consistent_loss, each_consistent_loss, total_loss = criterion(outputs, targets, hier_matrixs)
        total_loss.backward()
        optimizer.step()

        metrics.update_train(each_cls_loss=each_classification_loss, 
                     total_cls_loss=classification_loss, 
                     each_consistent_loss=each_consistent_loss, 
                     total_consistent_loss=consistent_loss, 
                     outputs=outputs, 
                     targets=targets, 
                     type="soft" if batchWiseAug != None else "hard")
    return metrics

def validate(epoch, end_epoch, model, loader, criterion, device, num_classes, consistent_list):
    model.eval()
    metrics = MetricCal_Hier(num_classes=num_classes, consistent_list=consistent_list, device=device)
    with torch.no_grad():
        for inputs, targets in tqdm(loader, total=len(loader), desc="Validating epoch [{0}/{1}]".
                                format(epoch, end_epoch)):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets[:,0]
            outputs = model(inputs)
            outputs = outputs["Species"]
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
    hierarchy_label_root = config["DATASET"]["HIERARCHY_LABEL_ROOT"]
    hierarchy_columns = config["DATASET"]["HIERARCHY_COLUMNS"]
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

    #Learning_rate
    if model_type not in [0]:
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
        hierarchy_label_root=hierarchy_label_root,
        hierarchy_columns= hierarchy_columns,
        std=std,
        mean=mean,
        img_size=img_size,
        batch_size=batch_size,
        transform = enabled_transform
    )

    test_data = DatasetLoader(
        img_path=test_path,
        mask_path=test_mask_path,
        hierarchy_label_root=hierarchy_label_root,
        hierarchy_columns= hierarchy_columns,
        std=std,
        mean=mean,
        img_size=img_size,
        batch_size=batch_size
    )

    training_loader = train_data.dataset_loader("train")
    testing_loader = test_data.dataset_loader("test")

    num_classes = train_data.num_classes
    consistent_list, hier_matrixs = train_data.Create_Consistent_Matrix(device)
  

    batchWiseAug = None
    if enabled_batchwise_transform:
        batchWiseAug = BatchWiseAug(config=config, num_classes=num_classes)

    model = Model(model_type=model_type, num_classes=num_classes).to(device)
    # model = Resnet50_Hierarchy(num_classes, 0, False).to(device)

    eval_criterion = nn.CrossEntropyLoss()
    train_criterion = HierarchyGuidedLoss(
        num_classes=num_classes, 
        type="train", 
        enabled_batchwise_transform=enabled_batchwise_transform)
    optimizer = optim.AdamW(model.parameters(), lr=Learning_rate_para["MAX_LR"], weight_decay=1e-2)

    if model_type not in [0]:
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
                                num_classes=num_classes, 
                                hier_matrixs=hier_matrixs,
                                consistent_list=consistent_list)
        train_loss, train_acc = train_metrics.overall_loss(weights=0.5), train_metrics.avg_accuracy("Species")
        scheduler.step()
        print()
        val_metrics = validate(epoch, end_epoch, model, testing_loader, eval_criterion, device, num_classes=num_classes, consistent_list=consistent_list)
        val_loss, val_acc = val_metrics.avg_cls_loss, val_metrics.avg_accuracy("Species")
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
            #Lưu mấy cái loss chung của train trước
            metric_row = {
                "train_overall_loss": train_loss,
                "train_cls_loss": train_metrics.avg_cls_loss,
                "train_consistent_loss": train_metrics.avg_consistent_loss
            }
            #Lưu loss của từng phân cấp
            for key in num_classes.keys():
                metric_row.update({
                    f"train_cls_loss_{key}": train_metrics.avg_each_cls_loss(key)
                })
            for key in consistent_list:
                metric_row.update({
                    f"train_consistent_loss_{key}": train_metrics.avg_each_consistent_loss(key)
                })
            metric_row["val_loss"] = val_loss

            for key in num_classes.keys():
                metric_row.update({
                    f"train_acc_{key}": train_metrics.avg_accuracy(key),
                    # f"val_acc_{key}": val_metrics.avg_accuracy(key),
                    f"train_precision_{key}": train_metrics.precision_macro(key),
                    # f"val_precision_{key}": val_metrics.precision_macro(key),
                    f"train_recall_{key}": train_metrics.recall_macro(key),
                    # f"val_recall_{key}": val_metrics.recall_macro(key),
                    f"train_f1_{key}": train_metrics.f1_macro(key), 
                    # f"val_f1_{key}": val_metrics.f1_macro(key), 
                    f"train_MCC_{key}": train_metrics.MCC(key),
                    # f"val_MCC_{key}": val_metrics.MCC(key),
                    f"train_FMI_{key}": train_metrics.FMI(key),
                    # f"val_FMI_{key}": val_metrics.FMI(key),
                    f"train_Cohen_Kappa_{key}": train_metrics.cohen_kappa(key)
                    # f"val_Cohen_Kappa_{key}": val_metrics.cohen_kappa(key)
                })
            metric_row.update({
                "val_acc": val_acc,
                "val_precision": val_metrics.precision_macro("Species"),
                "val_recall": val_metrics.recall_macro("Species"),
                "val_f1": val_metrics.f1_macro("Species"), 
                "val_MCC": val_metrics.MCC("Species"),
                "val_FMI": val_metrics.FMI("Species"),
                "val_Cohen_Kappa": val_metrics.cohen_kappa("Species"),
            })
            Saving_Metric3(epoch=epoch, metric_row=metric_row, path=metrics_path)
        if epochs_no_improve >= patience and early_stopping == True:
            print("Early stopping triggered at epoch {0}".format(epoch))
            break
        print()

    #Cần debug lại để chắc chắn đúng
if __name__ == '__main__':
    main()
