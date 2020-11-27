augmented = aug(images.to('cuda:0'))  # in device cuda:0
augmented = aug(images.to('cuda:1'))  # in device cuda:1
