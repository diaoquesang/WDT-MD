from config import config
import numpy as np

from dataset import myTestDataset
from transform import myTransform
from torch.utils.data import DataLoader
from diffusers import LCMScheduler
from tqdm import tqdm
import cv2 as cv
import torch
import time
import os
from monai.utils import set_determinism
from pytorch_wavelets import DWTForward, DWTInverse
from model import mySiTModel

set_determinism(42)


def DWT(img, device="cpu"):
    dwt = DWTForward(wave='db6', J=1, mode="periodization").to(device)
    return torch.cat((dwt(img)[0], torch.squeeze(dwt(img)[1][0], dim=1)), dim=1)


def IDWT(img, device="cpu"):
    idwt = DWTInverse(wave='db6', mode="periodization").to(device)
    return idwt((torch.unsqueeze(img[:, 0, :, :], dim=1), [torch.unsqueeze(img[:, 1:, :, :], dim=1)]))


def eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_image_path = "output1210e"

    img_path = os.path.join("e-ophtha_MA", "image")

    model = torch.load("e-ophtha_MA-2025-07-24-myWDT-12-n10.pth").to(device).eval()

    test_file_list = "e-ophtha_MA_testset.txt"

    myTestSet = myTestDataset(test_file_list, img_path,
                              myTransform['wdtTransform1'])
    myTestLoader = DataLoader(myTestSet, batch_size=1, shuffle=False)

    noise_scheduler = LCMScheduler(num_train_timesteps=config.num_train_timesteps)
    noise_scheduler.set_timesteps(config.num_infer_timesteps)

    with torch.no_grad():
        progress_bar = tqdm(enumerate(myTestLoader), total=len(myTestLoader), ncols=100)
        total_start = time.time()
        for step, batch in progress_bar:
            _, filename, v = batch[0].to(device=device, non_blocking=True).float(), batch[1][0], batch[2].to(
                device=device, non_blocking=True).float()
            v_dwt = DWT(v, device)

            noise_scheduler = LCMScheduler(num_train_timesteps=config.num_train_timesteps)
            noise_scheduler.set_timesteps(config.num_infer_timesteps)

            noise = torch.randn_like(v_dwt).to(device)
            sample = torch.cat((noise, v_dwt), dim=1).to(device)  # BCHW

            for j, t in tqdm(enumerate(noise_scheduler.timesteps)):
                # print(j,t)
                residual = model(sample, torch.Tensor((t,)).to(device).long()).to(device)
                sample = noise_scheduler.step(residual, t, sample).prev_sample
                sample = torch.cat((sample[:, :4], v_dwt), dim=1)  # BCHW

                if not config.use_server:
                    v_show = np.array(sample[:, 0].detach().to("cpu"))
                    v_show = np.squeeze(v_show)  # HW
                    v_show = v_show * 0.5 + 0.5
                    v_show = np.clip(v_show, 0, 1)
                    cv.imshow("v_show", v_show)
                    cv.waitKey(1)

            img = cv.imread(os.path.join(img_path, filename))

            hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            h, s, v_real = cv.split(hsv_img)

            v_recon = IDWT(sample[:, :4], device)
            v_recon = np.array(v_recon.detach().to("cpu"))
            v_recon = np.squeeze(v_recon)  # HW
            v_recon = v_recon * 0.5 + 0.5

            v_recon = np.clip(v_recon, 0, 1)

            v_recon = np.multiply(v_recon, 255).astype(np.uint8)
            v_recon_cropped = v_recon[50:-50, :]

            v_recon_cropped[v_recon_cropped < 10] = v_real[v_recon_cropped < 10]

            source_hist, bins1 = np.histogram(v_recon_cropped, bins=256, range=[0, 256])
            target_hist, bins2 = np.histogram(v_real, bins=256, range=[0, 256])
            source_cdf = source_hist.cumsum()
            source_cdf = (source_cdf / source_cdf[-1]).astype(np.float32)
            target_cdf = target_hist.cumsum()
            target_cdf = (target_cdf / target_cdf[-1]).astype(np.float32)
            matched_cdf = np.interp(source_cdf, target_cdf, range(256)).astype(np.uint8)
            v_recon_cropped = matched_cdf[v_recon_cropped]

            if not config.use_server:
                cv.imshow("v_recon", v_recon_cropped)
                cv.waitKey(1)

            pred = cv.merge([h, s, v_recon_cropped])
            pred = cv.cvtColor(pred, cv.COLOR_HSV2BGR)
            cv.imwrite(os.path.join(output_image_path, filename), pred)

        total_time = time.time() - total_start
        print(f"Total time: {total_time}.")


if __name__ == "__main__":
    eval()
