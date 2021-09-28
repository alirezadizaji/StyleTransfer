""" based on paper https://arxiv.org/pdf/1508.06576.pdf """

import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.models import vgg16
from torch.nn import functional as F
from typing import Tuple, List
from torch.optim import Adam, LBFGS
import math
import numpy as np
import IPython.display

layer = {}
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def read_img(pth: str, resize=512) -> np.ndarray:
    img = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (resize, resize))
    img = img[np.newaxis, ...].transpose((0, 3, 1, 2))
    return img

def plot_img(img: torch.Tensor):
    img = img.detach().cpu().numpy().squeeze().transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()

def plot_all_imgs(imgs: List[torch.Tensor], cols=5):
    rows = len(imgs) // cols
    plt.figure(figsize=(14,4))
    
    for i,img in enumerate(imgs):
        img = img.detach().cpu().numpy().squeeze().transpose((1, 2, 0))
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)

def get_content_layers():
    return [26]

def get_style_layers():
    return [0, 5, 10, 17, 24]

def get_layer(name):
    def hook(model, input, output):
        layer[name] = output.detach()
    return hook
    
def get_vgg16_model() -> torch.nn.Module:
    model = vgg16(pretrained=True) 
    model.to(device)
    
    style_layers = get_style_layers()
    content_layers = get_content_layers()
    feature_layers = style_layers + content_layers
    
    for param in model.parameters():
        param.requires_grad = False
    for l in feature_layers:
        model.features[l].register_forward_hook(get_layer(f"features.{l}"))

    return model

def get_style_content_features(model: torch.nn.Module, 
                               x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    style_layers = get_style_layers()
    num_style_layers = len(style_layers)
    content_layers = get_content_layers()
    feature_layers = style_layers + content_layers

    model(x)
    features = []
    for l in feature_layers:
        print
        features.append(layer[f"features.{l}"])

    style_features = features[:num_style_layers]
    content_features = features[num_style_layers:]
    return style_features, content_features

def cal_content_loss(pred: torch.Tensor,
                     target: torch.Tensor) -> torch.Tensor:

    return F.mse_loss(pred, target)

def cal_gram_matrix(x: torch.Tensor) -> torch.Tensor:
    B, C, _, _ = x.size()
    x = x.view(-1, C)
    gram = torch.matmul(torch.t(x), x)
    return gram / B

def cal_style_loss(pred: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:

    pred = cal_gram_matrix(pred)
    target = cal_gram_matrix(target)
    return F.mse_loss(pred, target)

def cal_total_loss(model: torch.nn.Module,
                    x: torch.Tensor,
                    target_content_features: List[torch.Tensor], 
                    target_style_features: List[torch.Tensor], 
                    content_score: int,
                    style_score: int) -> Tuple[torch.Tensor]:

    content_loss = 0
    style_loss = 0
    style_features, content_features = get_style_content_features(model, x)

    style_weights = [1./len(style_features)] * len(style_features)
    content_weights = [1./len(content_features)] * len(content_features)

    for w, pred, target in zip(style_weights, style_features, target_style_features):
        style_loss = style_loss + cal_style_loss(pred, target) * w

    for w, pred, target in zip(content_weights, content_features, target_content_features):
        content_loss = content_loss + cal_content_loss(pred, target) * w

    style_loss *= style_score
    content_loss *= content_score

    total_loss = style_loss + content_loss

    return total_loss, style_loss, content_loss


def run(style_pth: str,
        content_pth: str,
        style_score: int,
        content_score: int,
        num_iters: int,
        ):
    
    model = get_vgg16_model()
    style_img = read_img(style_pth)
    content_img = read_img(content_pth)
    style_img = torch.tensor(style_img, device=device, dtype=torch.float)
    content_img = torch.tensor(content_img, device=device, dtype=torch.float)
    base_img = content_img.clone().detach()

    target_style_features, _ = get_style_content_features(model, style_img)
    _, target_content_features = get_style_content_features(model, content_img)

    optimizer = Adam(params=model.parameters(), lr=5, betas=(0.99, 0.999), eps=1e-1)
    
    all_imgs = []
    best_loss, best_img = math.inf, None
    for i in range(num_iters):
        optimizer.zero_grad()
        total_loss, style_loss, content_loss = \
                cal_total_loss(model, 
                               base_img, 
                               target_content_features,
                               target_style_features, 
                               content_score, 
                               style_score)
        total_loss.backward()
        optimizer.step()

        if total_loss < best_loss:
            best_loss = total_loss
            best_img = base_img

        if i % 100 == 0:
            print(f"Iteration {i}: total_loss {total_loss} style_loss {style_loss} content_loss {content_loss}")  
            all_imgs.append(base_img)

    print("DONE :)")
    plot_all_imgs(all_imgs)

run(style_pth = "A.jpg",
    content_pth = "B.jpg",
    style_score = 1,
    content_score = 1e-3,
    num_iters=1000)