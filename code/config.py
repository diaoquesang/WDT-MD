from dataclasses import dataclass


@dataclass
class config():
    use_server = True
    width = 300
    height = 200
    val_epoch_interval = 10

    batch_size = 4
    epoch_number = 600
    initial_learning_rate = 1e-4
    milestones = [300, 400, 500]
    num_train_timesteps = 1000
    num_infer_timesteps = 50

    offset_noise = False
    offset_noise_coefficient = 0.1
    num_DiT_blocks = 12
    noised_condition = True
    inpaint = True
    noised_timesteps = 10

    current_dataset = "e-ophtha_MA"
