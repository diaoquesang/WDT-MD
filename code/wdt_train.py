from matplotlib import pyplot as plt
from config import config
from dataset import myTrainDataset
from transform import myTransform
from torch.utils.data import DataLoader
from model import mySiTModel
from diffusers import LCMScheduler
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from datetime import date

import torch.nn.functional as F
import torch
import time
from pytorch_wavelets import DWTForward, DWTInverse

from monai.utils import set_determinism

# Setting random seeds
set_determinism(42)


def DWT(img, device):
    dwt = DWTForward(wave='db6', J=1, mode="periodization").to(device)
    return torch.cat((dwt(img)[0], torch.squeeze(dwt(img)[1][0], dim=1)), dim=1)


def IDWT(img, device):
    idwt = DWTInverse(wave='db6', mode="periodization").to(device)
    return idwt((torch.unsqueeze(img[:, 0, :, :], dim=1), [torch.unsqueeze(img[:, 1:, :, :], dim=1)]))


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运行环境

    train_file_list = config.current_dataset + "_trainset.txt"  # Text file storing training set filenames
    val_file_list = config.current_dataset + "_valset.txt"  # Text file storing validation set filenames

    img_path = config.current_dataset + "/image"  # Image folder path
    mask_path = config.current_dataset + "/mask"  # Mask folder path

    # Set the training set
    myTrainSet = myTrainDataset(train_file_list, img_path, mask_path,
                                myTransform['wdtTransform1'])
    # Set the validation set
    myValSet = myTrainDataset(val_file_list, img_path, mask_path,
                              myTransform['wdtTransform1'])

    myTrainLoader = DataLoader(myTrainSet, batch_size=config.batch_size, shuffle=True)  # Data loader of training set
    myValLoader = DataLoader(myValSet, batch_size=config.batch_size, shuffle=True)  # Data loader of validation set

    print("Number of batches in train set:", len(myTrainLoader))  # Number of batch in training set
    print("Training set size:", len(myTrainSet))  # Training set size
    print("Number of batches in validation set:", len(myValLoader))  # Number of batch in validation set
    print("Validation set size:", len(myValSet))  # Validation set size

    model = mySiTModel.to(device).train()  # Diffusion Transformer Noise Estimator Network

    # Setting noise scheduler
    noise_scheduler = LCMScheduler(num_train_timesteps=config.num_train_timesteps)
    noise_scheduler.set_timesteps(config.num_infer_timesteps)

    # Setting the dynamic learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.initial_learning_rate, eps=1e-6)
    milestones = [x * len(myTrainLoader) for x in config.milestones]
    optimizer_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    train_losses = []
    val_losses = []
    plt_train_loss_epoch = []
    plt_val_loss_epoch = []

    train_epoch_list = list(range(0, config.epoch_number))
    val_epoch_list = list(range(0, int(config.epoch_number / config.val_epoch_interval)))

    print(time.strftime("%H:%M:%S", time.localtime()), "----------Begin Training----------")
    for epoch in range(config.epoch_number):
        model.train()
        for i, batch in tqdm(enumerate(myTrainLoader)):
            mask, v, v_enhanced = batch[1].to(device), batch[3].to(device), batch[4].to(device)
            v = DWT(v, device)
            v_enhanced = DWT(v_enhanced, device)

            if config.noised_condition:
                noise_c = torch.randn_like(v).to(device)
                timesteps_c = torch.randint(0, config.noised_timesteps, (v.shape[0],), device=device).long()
                v = noise_scheduler.add_noise(v, noise_c, timesteps_c)

            cat = torch.cat((v_enhanced, v), dim=-3)

            # Noise selection
            if config.offset_noise:
                noise = torch.randn_like(v).to(device) + config.offset_noise_coefficient * torch.randn(
                    v.shape[0], v.shape[1], 1,
                    1).to(device)
            else:
                noise = torch.randn_like(v).to(device)

            blank = torch.zeros_like(v).to(device)
            noise = torch.cat((noise, blank), dim=-3)

            # Randomly sample a timestep for each image in a batch
            timesteps = torch.randint(0, config.num_train_timesteps, (v.shape[0],), device=device).long()

            # Adding noise to an image based on the noise schedule
            noisy_images = noise_scheduler.add_noise(cat, noise, timesteps)

            # Getting the model's noise prediction
            noise_pred = model(noisy_images, timesteps)

            # Calculation of loss
            loss = F.mse_loss(noise_pred[:, :4].float(), noise[:, :4].float())

            # Loss backward
            loss.backward()
            train_losses.append(loss.item())

            # Update optimizer
            optimizer.step()
            optimizer.zero_grad()
            optimizer_scheduler.step()

        # Recording training loss
        train_loss_epoch = sum(train_losses[-len(myTrainLoader):]) / len(myTrainLoader)
        print(time.strftime("%H:%M:%S", time.localtime()), f"Epoch:{epoch},train losses:{train_loss_epoch}")
        plt_train_loss_epoch.append(train_loss_epoch)

        if (epoch + 1) % config.val_epoch_interval == 0:
            model.eval()
            print(time.strftime("%H:%M:%S", time.localtime()), "----------Stop Training----------")
            print(time.strftime("%H:%M:%S", time.localtime()), "----------Begin Validation----------")
            with torch.no_grad():
                for i, batch in tqdm(enumerate(myValLoader)):
                    mask, v, v_enhanced = batch[1].to(device), batch[3].to(device), batch[4].to(device)
                    v = DWT(v, device)
                    v_enhanced = DWT(v_enhanced, device)

                    if config.noised_condition:
                        noise_c = torch.randn_like(v).to(device)
                        timesteps_c = torch.randint(0, config.noised_timesteps, (v.shape[0],),
                                                    device=device).long()
                        v = noise_scheduler.add_noise(v, noise_c, timesteps_c)

                    cat = torch.cat((v_enhanced, v), dim=-3)

                    # Noise selection
                    if config.offset_noise:
                        noise = torch.randn_like(v).to(device) + config.offset_noise_coefficient * torch.randn(
                            v.shape[0],
                            v.shape[1], 1,
                            1).to(device)
                    else:
                        noise = torch.randn_like(v).to(device)

                    blank = torch.zeros_like(v).to(device)
                    noise = torch.cat((noise, blank), dim=-3)

                    # Randomly sample a timestep for each image in a batch
                    timesteps = torch.randint(0, config.num_train_timesteps, (v.shape[0],),
                                              device=device).long()

                    # Adding noise to an image based on the noise schedule
                    noisy_images = noise_scheduler.add_noise(cat, noise, timesteps)

                    # Getting the model's noise prediction
                    noise_pred = model(noisy_images, timesteps)

                    # Calculation of loss
                    loss = F.mse_loss(noise_pred[:, :4].float(), noise[:, :4].float())
                    val_losses.append(loss.item())

                # Recording validation loss
                val_loss_epoch = sum(val_losses[-len(myValLoader):]) / len(myValLoader)
                print(time.strftime("%H:%M:%S", time.localtime()), f"Epoch:{epoch},validation losses:{val_loss_epoch}")
                plt_val_loss_epoch.append(val_loss_epoch)

                print(time.strftime("%H:%M:%S", time.localtime()), "----------End Validation----------")
                print(time.strftime("%H:%M:%S", time.localtime()), "----------Continue to Train----------")
    print(time.strftime("%H:%M:%S", time.localtime()), "----------End Training Successfully----------")

    # View loss curves
    f, ([ax1, ax2]) = plt.subplots(1, 2)
    ax1.plot(train_epoch_list, plt_train_loss_epoch, color="red")
    ax1.set_title('Training loss')
    ax2.plot(val_epoch_list, plt_val_loss_epoch, color="blue")
    ax2.set_title('Validation loss')
    plt.savefig(
        "loss-" + config.current_dataset + "-" + str(config.num_DiT_blocks) + "-n" + str(
            config.noised_timesteps) + ".png")  # Save loss curves
    torch.save(model,
               config.current_dataset + "-" + str(date.today()) + "-myWDT-" + str(config.num_DiT_blocks) + "-n" + str(
                   config.noised_timesteps) + ".pth")


if __name__ == "__main__":
    train()
