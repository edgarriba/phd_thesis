# the source points are the region to crop corners
points_src = torch.tensor([[
    [125., 150.], [562., 40.], [562., 282.], [54., 328.],
]])

# the destination points are the image vertexes
h, w = img_rgb.shape[-2:]  # destination size
points_dst = torch.tensor([[
    [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.],
]])

# compute perspective transform
M: torch.tensor = K.get_perspective_transform(
    points_src, points_dst)

# warp the original image by the found transform
img_warp: torch.tensor = K.warp_perspective(
    img_rgb, M, dsize=(h, w))
