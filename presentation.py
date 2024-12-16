import torch
import segmentation_models_pytorch as smp


def create_model():
    return smp.Unet(
        encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                      # model output channels (number of classes in your dataset)
    )

def create_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=5e-1)