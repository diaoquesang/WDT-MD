from config import config
from torchvision import transforms
import cv2 as cv


class resize():
    def __call__(self, img):
        img = cv.resize(img, (config.width, config.height))
        return img


class padding():
    def __call__(self, img):
        img = cv.copyMakeBorder(
            src=img,
            top=(config.width - config.height) // 2,
            bottom=(config.width - config.height) // 2,
            left=0,
            right=0,
            borderType=cv.BORDER_CONSTANT,
            value=0
        )
        return img


myTransform = {
    'wdtTransform1': transforms.Compose([
        resize(),
        padding(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
}
