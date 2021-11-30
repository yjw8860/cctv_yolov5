import os
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from PIL import ImageDraw
from torchvision import transforms as T

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_test_dataset, check_file, check_img_size, \
    non_max_suppression_test, set_logging, increment_path
from utils.torch_utils import select_device, time_synchronized
from utils.utils import json_load
import warnings
warnings.filterwarnings("ignore")


def test(data,
         dataset_name,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         augment=False,
         model=None,
         dataloader=None,
        ):

    prediction_save_path = os.path.join('./results', dataset_name)
    os.makedirs(prediction_save_path, exist_ok=True)

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device('', batch_size=batch_size)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    with open(data, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_test_dataset(data)  # check


    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test']
        dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), single_cls=False, pad=0.5, rect=True)[0]

    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    img_idx = 0
    for batch_i, (img, _, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        images = img.to(device, non_blocking=True)
        images = images.half() if half else images.float()  # uint8 to fp16/32
        images /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = images.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            inf_out, train_out = model(images, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            outputs = non_max_suppression_test(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
            t1 += time_synchronized() - t
            images = list(img.to(device) for img in images)
            for output, img in zip(outputs, images):
                img = T.ToPILImage()(img).convert('RGB')
                output = output.detach().to('cpu').numpy()
                for pred in output:
                    draw = ImageDraw.Draw(img)
                    xmin, ymin, xmax, ymax, score, _ = pred
                    if score > 0.5:
                        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=2)
                    else:
                        break
                # img_save_path = os.path.join(prediction_save_path, f'{img_idx}.jpg')
                # img_idx += 1
                # img.save(img_save_path)
                img.show()

if __name__ == '__main__':
    config = json_load('./config.json')
    test_options = config["test_options"]
    root_dir = os.path.join(test_options["root_dir"], test_options["dataset_name"])
    dataset_name = test_options['dataset_name']
    data = os.path.join(root_dir, test_options["data"])
    batch_size = test_options["batch_size"]
    img_size = test_options["img_size"]
    conf_thres = test_options["conf_thres"]
    iou_thres = test_options["iou_thres"]
    last_save_folder_dir = os.path.join('./runs',test_options["dataset_name"], 'weights')
    weights = os.path.join(last_save_folder_dir, 'best.pt')

    data = check_file(data)  # check file

    test(data = data,
         dataset_name = dataset_name,
         weights = weights,
         batch_size= batch_size,
         imgsz=img_size,
         conf_thres=conf_thres,
         iou_thres=iou_thres,
         )
