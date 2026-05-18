import torch
#Quên cái vụ Consistent loss có số loss ít hơn class, nên sửa lại (Đã sửa)
#Bữa nào rãnh viết lại cái reset cho tiết kiệm dung lượng
class MetricCal_Hier():
    def __init__(self, num_classes, consistent_list, device) -> None:
        self.num_classes = num_classes #Dây là một dict
        self.consistent_list = consistent_list #Đây là một list, số lượng của nó = len(num_classes) - 1
        self.device = device
        self.reset()
    def reset(self):
        self.total_cls_loss = torch.zeros(1, device=self.device)
        self.each_cls_loss = {
            key: torch.zeros(1, device=self.device) 
            for key in self.num_classes.keys()
        } #Lưu loss của từng cấp
        
        self.total_consistent_loss = torch.zeros(1, device=self.device)
        self.each_consistent_loss = {
            key: torch.zeros(1, device=self.device) 
            for key in self.consistent_list
        } #Lưu loss của từng cấp
        
        # self.total_overall_loss = torch.zeros(1, device=self.device)
        self.correct = {
            key: torch.zeros(1, device=self.device) 
            for key in self.num_classes.keys()
        } #Lưu correct của từng cấp
        self.total = torch.zeros(1, device=self.device)


        self.cm = {
            key: torch.zeros(
                (value, value),
                dtype=torch.int64,
                device=self.device
            )
            for key, value in self.num_classes.items()
        } #Lưu confusion matrix của từng cấp
        
        self.tp_per_class = {
            key: torch.zeros(
            value,
            device=self.device
            )
            for key, value in self.num_classes.items()
        }
        self.fp_per_class = {
            key: torch.zeros(
            value,
            device=self.device
            )
            for key, value in self.num_classes.items()
        }
        self.fn_per_class = {
            key: torch.zeros(
            value,
            device=self.device
            )
            for key, value in self.num_classes.items()
        }
        # self.fp_per_class = torch.zeros(self.num_classes, device=self.device)
        # self.fn_per_class = torch.zeros(self.num_classes, device=self.device)

    @torch.no_grad()
    def update_test(self, loss, outputs, targets, type="soft"):
        #Dùng để tính classification loss
        batch_size = targets.size(0)
        
        self.total_cls_loss += loss.detach() * batch_size
        self.total += batch_size

        
        if type == "soft":
            pred_class = outputs.argmax(dim=1)
            true_class = targets.argmax(dim=1)
        else:
            _, pred_class = outputs.max(1)
            true_class = targets
        
        self.correct["Species"] += (pred_class == true_class).sum()

        
        # pred = pred_class.detach().cpu()
        # true = true_class.detach().cpu()
        
        # Confusion matrix update
        cm = torch.bincount(
            self.num_classes["Species"] * true_class + pred_class,
            minlength=self.num_classes["Species"] ** 2
        ).reshape(self.num_classes["Species"], self.num_classes["Species"])
        
        self.cm["Species"] += cm

        self.tp_per_class["Species"] += cm.diag()
        self.fp_per_class["Species"] += cm.sum(dim=0) - cm.diag()
        self.fn_per_class["Species"] += cm.sum(dim=1) - cm.diag()

    @torch.no_grad()
    def update_train(self, 
                     each_cls_loss, 
                     total_cls_loss, 
                     each_consistent_loss, 
                     total_consistent_loss, 
                     outputs, 
                     targets, 
                     type="soft"):
        #Dùng để tính classification loss
        if type == "hard":
            batch_size = targets.size(0)
        else:
            batch_size = next(iter(targets.values())).size(0)


        self.total_cls_loss += total_cls_loss.detach() * batch_size
        self.total_consistent_loss += total_consistent_loss.detach() * batch_size
        
        pred_class = dict()
        true_class = dict()
        for index, key in enumerate(self.num_classes.keys()):
            self.each_cls_loss[key] = each_cls_loss[key].detach() * batch_size
            if type == "soft":
                pred_class[key] = outputs[key].argmax(dim=1)
                # true_class[key] = targets[:, index].argmax(dim=1)
                true_class[key] = targets[key].argmax(dim=1)
            else:
                _, pred_class[key] = outputs[key].max(1)
                true_class[key] = targets[:, index]
            
            self.correct[key] += (pred_class[key] == true_class[key]).sum()

            # Confusion matrix update
            cm = torch.bincount(
                self.num_classes[key] * true_class[key] + pred_class[key],
                minlength=self.num_classes[key] ** 2
            ).reshape(self.num_classes[key], self.num_classes[key])
            
            self.cm[key] += cm

            self.tp_per_class[key] += cm.diag()
            self.fp_per_class[key] += cm.sum(dim=0) - cm.diag()
            self.fn_per_class[key] += cm.sum(dim=1) - cm.diag()

        for key in self.consistent_list:
            self.each_consistent_loss[key] = each_consistent_loss[key].detach() * batch_size

        self.total += batch_size



    @property
    def avg_cls_loss(self):
        """Average loss over all accumulated batches."""
        return (self.total_cls_loss / self.total).item() if self.total > 0 else 0.0

    def avg_each_cls_loss(self, key):
        return (self.each_cls_loss[key] / self.total).item() if self.total > 0 else 0.0

    @property
    def avg_consistent_loss(self):
        return (self.total_consistent_loss / self.total).item() if self.total > 0 else 0.0
   
    def avg_each_consistent_loss(self, key):
        return (self.each_consistent_loss[key] / self.total).item() if self.total > 0 else 0.0

    def overall_loss(self, weights):
        loss = self.avg_cls_loss + weights * self.avg_consistent_loss
        return loss

    def avg_accuracy(self, key):
        """Accuracy (%) over all accumulated batches."""
        return (self.correct[key] / self.total).item() if self.total > 0 else 0.0

    def precision(self, key):
        """Per-class precision"""
        denom = self.tp_per_class[key] + self.fp_per_class[key]
        prec = torch.where(denom > 0, self.tp_per_class[key] / denom, torch.zeros_like(denom))
        return prec


    def recall(self, key):
        """Per-class recall"""
        denom = self.tp_per_class[key] + self.fn_per_class[key]
        rec = torch.where(denom > 0, self.tp_per_class[key] / denom, torch.zeros_like(denom))
        return rec

    def f1_score(self, key):
        """Per-class F1-score"""
        prec = self.precision(key)
        rec = self.recall(key)
        denom = prec + rec
        f1 = torch.where(denom > 0, 2 * prec * rec / denom, torch.zeros_like(denom))
        return f1

    def precision_macro(self, key):
        return self.precision(key).mean().item()


    def recall_macro(self, key):
        return self.recall(key).mean().item()


    def f1_macro(self, key):
        return self.f1_score(key).mean().item()
    
    # Matthews Correlation Coefficient(MCC)
    def MCC(self, key):
        eps = 1e-12
        cm = self.cm[key].detach().float()

        t_sum = cm.sum(dim=1)      # true counts per class
        p_sum = cm.sum(dim=0)      # predicted counts per class

        n_correct = torch.trace(cm)
        n_samples = cm.sum()

        cov_ytyp = n_correct * n_samples - torch.dot(t_sum, p_sum)

        cov_ypyp = n_samples**2 - torch.dot(p_sum, p_sum)
        cov_ytyt = n_samples**2 - torch.dot(t_sum, t_sum)

        denom = torch.sqrt(cov_ytyt * cov_ypyp).clamp_min(eps)

        return (cov_ytyp / denom).item()

    # Fowlkes–Mallows Index
    def FMI(self, key):
        eps = 1e-12

        tp = self.tp_per_class[key].detach().float()
        fp = self.fp_per_class[key].detach().float()
        fn = self.fn_per_class[key].detach().float()

        denom = torch.sqrt((tp + fp) * (tp + fn)).clamp_min(eps)

        fmi_per_class = tp / denom

        return fmi_per_class.mean().item()

    def cohen_kappa(self, key):
        eps = 1e-12
        cm = self.cm[key].float()

        n = cm.sum()

        po = torch.trace(cm) / (n + eps)

        true_sum = cm.sum(dim=1)
        pred_sum = cm.sum(dim=0)

        pe = torch.dot(true_sum, pred_sum) / ((n * n) + eps)

        return ((po - pe) / (1 - pe + eps)).item()