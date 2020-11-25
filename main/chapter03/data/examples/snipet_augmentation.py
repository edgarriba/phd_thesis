class DataAugmentationPipeline(torch.nn.Module):
    """Module to perform data augmentation using Kornia."""
    def __init__(self, apply_color_jitter: bool = True) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter
        self.transforms = torch.nn.Sequential(
            K.augmentation.RandomVerticalFlip(
                p=0.5, return_transform=True
            ),
        )
        self.jitter = K.augmentation.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, x: torch.Tensor):
        x_out, trans = self.transforms(x)
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out, trans

# load image in numpy using OpenCV
img: np.ndarray = cv2.imread("panda.jpg", cv2.IMREAD_COLOR)

# load image in torch.Tensor
img_bgr: torch.Tensor = K.image_to_tensor(img)
img_rgb = K.bgr_to_rgb(img_bgr)
img_rgb = K.normalize(img_rgb.float(), 0., 255.)

# create transform and apply
aug = DataAugmentationPipeline()
img_aug, transform = aug(img_rgb)
