# load image in numpy using OpenCV
img: np.ndarray = cv2.imread("wally.jpg", cv2.IMREAD_COLOR)

# load image in torch.Tensor
img_bgr: torch.Tensor = K.image_to_tensor(img, keepdim=False)
img_rgb = K.bgr_to_rgb(img_bgr)
img_rgb = K.normalize(img_rgb.float(), 0., 255.)

# extract tensor patches
patches = K.extract_tensor_patches(
    img_rgb, window_size=32, stride=32
)
