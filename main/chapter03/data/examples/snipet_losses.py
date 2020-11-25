# segmentation
loss = kornia.focal_loss(predictions, labels)
loss = kornia.dice_loss(predictions, labels)
loss = kornia.tversky_loss(predictions, labels)

# reconstruction
loss = kornia.psnr_loss(img1, img2)
loss = kornia.ssim_loss(img1, img2, window_size=5)
