from PIL import Image
import requests
from io import BytesIO
import torch
from torch import allclose
from torch.testing import assert_allclose

import kornia
from kornia.constants import Resample
from kornia.color import *
from kornia import augmentation as K
import kornia.augmentation.functional as F
import kornia.augmentation.random_generator as rg
from torchvision.transforms import functional as tvF
from torchvision.transforms import transforms

import matplotlib.pyplot as plt
import numpy as np

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

def tensor_pre_transform_wrapper(input: torch.Tensor):
    """ A wrapper that tried to reproduce the actual output from:
        - transforms.ToPILImage()
        - transforms.ToTensor()
        For each image, simply (img * 255).int() // 255
    """
    return torch.round(input * 255).to(torch.uint8) / 255.

response = requests.get("https://tinypng.com/images/social/website.jpg")
img = Image.open(BytesIO(response.content)).resize((255, 128))
img = to_tensor(img).unsqueeze(0).numpy()

import kornia.augmentation as K; import torch;
torch.manual_seed(42)
p = lambda x: torch.nn.Parameter(torch.tensor(x))
images = torch.tensor(img, requires_grad=True)
jitter = K.ColorJitter(p([0.8, 0.8]), p([0.7, 0.7]), p([0.6, 0.6]), p([0.1, 0.1]))
out = jitter(images)
loss = torch.nn.MSELoss()(out, images)
optimizer_img = torch.optim.SGD([images], lr=1e+5)
optimizer_param = torch.optim.SGD(jitter.parameters(), lr=0.1)
loss.backward()
optimizer_img.step()
optimizer_param.step()

to_pil(torch.cat([torch.tensor(img).squeeze(0), out.squeeze(0), images.squeeze(0)], dim=2))