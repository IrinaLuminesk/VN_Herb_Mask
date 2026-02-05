import torch
class MetricCal():
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes
        self.reset()
    def reset(self):
        self.total_loss = 0.0
        self.correct = 0
        self.correct_top5 = 0
        self.total = 0

        self.tp_per_class = torch.zeros(self.num_classes)
        self.fp_per_class = torch.zeros(self.num_classes)
        self.fn_per_class = torch.zeros(self.num_classes)

    def update(self, loss, outputs, targets, type="soft"):
        batch_size = targets.size(0)
        self.total_loss += loss.item() * batch_size
        if type == "soft":
            pred_class = outputs.argmax(dim=1)
            true_class = targets.argmax(dim=1)
        else:
            _, pred_class = outputs.max(1)
            true_class = targets
        self.correct += (pred_class == true_class).sum().item()
        self.total += batch_size

        # for c in range(self.num_classes):
        #     tp = ((pred_class == c) & (true_class == c)).sum().item()
        #     fp = ((pred_class == c) & (true_class != c)).sum().item()
        #     fn = ((pred_class != c) & (true_class == c)).sum().item()
        #     self.tp_per_class[c] += tp
        #     self.fp_per_class[c] += fp
        #     self.fn_per_class[c] += fn
        
        pred = pred_class.detach().cpu()
        true = true_class.detach().cpu()
        
        # Confusion matrix update
        cm = torch.bincount(
            self.num_classes * true + pred,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.tp_per_class += cm.diag().float()
        self.fp_per_class += (cm.sum(dim=0) - cm.diag()).float()
        self.fn_per_class += (cm.sum(dim=1) - cm.diag()).float()

    @property
    def avg_loss(self):
        """Average loss over all accumulated batches."""
        return self.total_loss / self.total if self.total > 0 else 0.0

    @property
    def avg_accuracy(self):
        """Accuracy (%) over all accumulated batches."""
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def precision(self):
        """Per-class precision"""
        denom = self.tp_per_class + self.fp_per_class
        prec = torch.where(denom > 0, self.tp_per_class / denom, torch.zeros_like(denom))
        return prec

    @property
    def recall(self):
        """Per-class recall"""
        denom = self.tp_per_class + self.fn_per_class
        rec = torch.where(denom > 0, self.tp_per_class / denom, torch.zeros_like(denom))
        return rec

    @property
    def f1_score(self):
        """Per-class F1-score"""
        prec = self.precision
        rec = self.recall
        denom = prec + rec
        f1 = torch.where(denom > 0, 2 * prec * rec / denom, torch.zeros_like(denom))
        return f1

    @property
    def precision_macro(self):
        return self.precision.mean().item()

    @property
    def recall_macro(self):
        return self.recall.mean().item()

    @property
    def f1_macro(self):
        return self.f1_score.mean().item()