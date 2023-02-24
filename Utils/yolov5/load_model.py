
import sys
from pathlib import Path
import torch
############ DO NOT DELETE OR CHANGE ORDER OF THIS ############
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
from models.experimental import attempt_load
###############################################################


def load_model(weights_file=None,  # model.pt path(s)
               half=True,  # use FP16 half-precision inference
               ):

    assert weights_file is not None, "load_model() requires as input the name of the pretrained model .pt file."

    # Set device cpu or gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # weights is .pt file stored by yolov5 github implementation in pytorch
    model = attempt_load(weights_file, map_location=device)  # load FP32 model

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()

    # Configure
    # model.eval()

    return model
