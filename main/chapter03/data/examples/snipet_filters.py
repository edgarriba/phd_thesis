# load image in numpy using OpenCV
img: np.ndarray = cv2.imread("goku.png", cv2.IMREAD_COLOR)

# load image in torch.Tensor
img_bgr: torch.Tensor = K.image_to_tensor(img, keepdim=False)
img_rgb = K.bgr_to_rgb(img_bgr).float() / 255.
img_gray = K.rgb_to_grayscale(img_rgb)

# apply a gaussian blur
img_edge = K.sobel(img_gray)
img_blur = K.gaussian_blur2d(img_rgb, (11, 11), (10.5, 10.5))
