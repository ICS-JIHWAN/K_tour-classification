import cv2
import numpy as np
from PIL import Image


# Log images
def tensor2numpy(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return var.astype('uint8')