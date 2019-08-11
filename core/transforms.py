from torchvision import transforms
from getbbox import getbbox
from PIL import Image
from config import INPUT_SIZE


def my_transforms1(img):
    """训练集"""
    _, box, _, _, _ = getbbox(img)  # 得到borderbox
    if len(box) > 0:
        img = img.crop(box)  # 围绕boderbox裁剪
        img = transforms.Resize((600, 600), Image.BILINEAR)(img)
        img = transforms.RandomCrop(INPUT_SIZE)(img)
        img = transforms.RandomHorizontalFlip()(img)
    else:
        img = transforms.Resize((600, 600), Image.BILINEAR)(img)
        img = transforms.CenterCrop(INPUT_SIZE)(img)

    transforms.ToTensor()
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return img

def my_transform2(img):
    """测试集"""
    _, box, _, _, _ = getbbox(img)  # 得到borderbox
    if len(box) > 0:
        img = img.crop(box)  # 围绕boderbox裁剪
        img = transforms.Resize((600, 600), Image.BILINEAR)(img)
        img = transforms.CenterCrop(INPUT_SIZE)(img)
    else:
        img = transforms.Resize((600, 600), Image.BILINEAR)(img)
        img = transforms.CenterCrop(INPUT_SIZE)(img)

    transforms.ToTensor()
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return img
