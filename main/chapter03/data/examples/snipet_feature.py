# Define local deature detector and descriptor
PS = 32  # patch size
detector = K.ScaleSpaceDetector(
    num_features=2000,
    resp_module=K.BlobHessian(),
    nms_module=K.ConvQuadInterp3d(10),
    scale_pyr_module=K.ScalePyramid(3, 1.6, PS, double_image=False),
    ori_module=K.LAFOrienter(19),
    aff_module=K.LAFAffineShapeEstimator(19),
    mr_size=6.0)

descriptor = K.HardNet(pretrained=True)

# detect and extract patches
lafs, resps = detector(timg_gray)
patches =  K.extract_patches_from_pyramid(timg_gray, lafs, 32)
B, N, CH, H, W = patches.size()
descs = descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)

# Matching
tentatives, scores = K.match_smnn(descs[0], descs[1], 0.95)
kps = KF.laf.get_laf_center(lafs)
kps_tent1 = kps[0:1,tentatives[:,0]]
kps_tent2 = kps[1:2,tentatives[:,1]]

# Finding homography
H = K.find_homography_dlt_iterated(
    kps_tent1, kps_tent2, 1-scores.view(1,-1)
)
