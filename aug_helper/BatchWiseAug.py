from torchvision.transforms import v2
from Aug.FMix import FMix

class BatchWiseAug:
    def __init__(self, config, num_classes):
        self.config = config 
        self.num_classes = num_classes

    def make_params(self):
        aug_map = {
            "CUTMIX": lambda: v2.CutMix(num_classes=self.num_classes, alpha=2.0),
            "MIXUP": lambda: v2.MixUp(num_classes=self.num_classes, alpha=2.0),
            "FMIX":  lambda: FMix(number_of_class=self.num_classes, alpha=0.6, decay_power=6),
        }

        Augs = []
        Probabilities = []
        all_augs = self.config["TRAIN"]["AUG"]

        for aug_name, aug_fn in aug_map.items():
            enabled, prob = all_augs[aug_name]
            if enabled:
                Augs.append(aug_fn())       # ← instantiated *only if enabled*
                Probabilities.append(prob)
        return v2.RandomChoice(Augs, p=Probabilities)
    def __call__(self, images, labels):
        return self.forward(images=images, labels=labels)

    def forward(self, images, labels):
        onBatchAugs = self.make_params()
        images, labels = onBatchAugs(images, labels)
        return images, labels