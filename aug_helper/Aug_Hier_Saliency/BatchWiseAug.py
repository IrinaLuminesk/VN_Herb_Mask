from torchvision.transforms import v2
from aug_helper.Aug_Hier_Saliency.CutMixHierSaliency import CutMixHierSaliency
from aug_helper.Aug_Hier_Saliency.FMixHierSaliency import FMixHierSaliency
from aug_helper.Aug_Hier_Saliency.MixUpHierSaliency import MixUpHierSaliency

class BatchWiseAug:
    def __init__(self, config, num_classes):
        self.config = config 
        self.num_classes = num_classes
        self.onBatchAugs = self.make_params()
    def make_params(self):
        aug_map = {
            "CUTMIX": lambda: CutMixHierSaliency(number_of_class=self.num_classes, alpha=2.0),
            "MIXUP": lambda: MixUpHierSaliency(number_of_class=self.num_classes, alpha=2.0),
            "FMIX":  lambda: FMixHierSaliency(number_of_class=self.num_classes, alpha=0.6, decay_power=6),
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
    def __call__(self, images, masks, labels):
        return self.forward(images=images, masks=masks, labels=labels)

    def forward(self, images, masks, labels):
        images, masks, labels, has_masks = self.onBatchAugs(images, masks, labels)
        return images, masks, labels, has_masks