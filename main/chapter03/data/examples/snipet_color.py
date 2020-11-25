# load image in numpy using OpenCV
img: np.ndarray = cv2.imread("simba.png", cv2.IMREAD_COLOR)

# load image in torch.Tensor
img_bgr: torch.Tensor = K.image_to_tensor(img)
img_rgb = K.bgr_to_rgb(img_bgr)
img_rgb = K.normalize(img_rgb, 0., 255.)

# apply color transforms
img_gray = K.rgb_to_grayscale(img_rgb)
