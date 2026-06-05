class ApplyToImageOnly:
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, img, mask):
        return self.tf(img), mask


class ApplyToBoth:
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, img, mask):
        return self.tf(img, mask)