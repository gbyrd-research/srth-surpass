

import sys
sys.path.append("$PATH_TO_YAY_ROBOT/src")  # to import aloha
import torch
import pickle
import threading
from queue import Queue


from copy import deepcopy

## TODO: merge load_merged_data and load_data_dvrk
from utils import load_data_dvrk
from policy import ACTPolicy
from aloha_pro.aloha_scripts.utils import (
    initialize_model_and_tokenizer,
    # encode_text,
    # crop_resize,
    # is_multi_gpu_checkpoint,
    # modify_real_time,
    # visualize_language_correction,
    # create_dataset_path,
    # memory_monitor,
    # save_trajectory,
)
# from instructor.train import build_instructor

CROP_TOP = True  # for aloha pro, whose top camera is high
CKPT = 0  # 0 for policy_last, otherwise put the ckpt number here
AUDIO = False
option = 0
intervention_needed = threading.Event()  # flag to signal an intervention
recorded_commands = Queue()

def forward_pass(data, policy, no_qpos=False):
    if len(data) == 5:  # use_language with qpos
        image_data, qpos_data, action_data, is_pad, command_embedding = data
        if command_embedding is None:
            raise ValueError("command_embedding is None in forward_pass with len(data)==5")
        command_embedding = command_embedding.cuda()
    elif no_qpos and len(data) == 4:  # use_language without qpos
        image_data, action_data, is_pad, command_embedding = data
        qpos_data = None
        if command_embedding is None:
            raise ValueError("command_embedding is None in forward_pass with no_qpos")
        command_embedding = command_embedding.cuda()
    elif len(data) == 4:  # no language, with qpos
        image_data, qpos_data, action_data, is_pad = data
        command_embedding = None
    else:
        raise ValueError(f"Unexpected data length: {len(data)}, no_qpos={no_qpos}")
    
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda() if qpos_data is not None else None,
        action_data.cuda(),
        is_pad.cuda(),
    )
    
    return policy(qpos_data, image_data, command_embedding=command_embedding)


def main():

    ckpt_path = "/home/grayson/surpass/srth-surpass/ckpt_srth_grasp_only/policy_epoch_6000_seed_0.ckpt"

    # General config -- Hardcoded for now
    is_eval = True
    ckpt_dir = "./ckpt_dir/surpass_grasp_only"
    policy_class = "ACT"
    onscreen_render = False
    task_name = "surpass_grasp_only"
    batch_size_train = 12
    batch_size_val = 12
    num_epochs = 20000
    log_wandb = True
    commands = []  # no --command provided
    use_language = True
    language_encoder = "distilbert"
    multi_gpu = False
    instructor_path = None  # not provided in your overrides
    history_len = None      # not provided in your overrides
    history_step_size = None  # not provided in your overrides
    hl_margin = None        # not provided in your overrides
    policy_level = "low"
    
    # Dataset config
    from dvrk_scripts.constants_dvrk import TASK_CONFIGS
    task_config = TASK_CONFIGS[task_name]
    dataset_dirs = []
    num_episodes_list = []
    task_configs_list = []  # Store all task configs for multi-dataset training
    max_episode_len = 0
    dataset_dirs.append(task_config["dataset_dir"])
    num_episodes_list.append(task_config["num_episodes"])
    task_configs_list.append(task_config)  # Store the full config
    max_episode_len = max(max_episode_len, task_config["episode_len"])
    camera_names = task_config["camera_names"]
    save_frequnecy = task_config['save_frequency']
    if task_config.get('no_qpos'):
        no_qpos = task_config['no_qpos']
    else:
        no_qpos = False

    # ACT Policy Configuration
    state_dim = 20 # changed from 14 to 20 for dvrk  
    lr_backbone = 1e-5
    lr = 1e-5
    chunk_size = 60
    kl_weight = 10
    hidden_dim = 512
    dim_feedforward = 3200
    image_encoder = "efficientnet_b3film"

    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {
        "lr": lr,
        "num_queries": chunk_size,
        "action_dim": 20,
        "kl_weight": kl_weight,
        "hidden_dim": hidden_dim,
        "dim_feedforward": dim_feedforward,
        "lr_backbone": lr_backbone,
        "backbone": image_encoder,
        "enc_layers": enc_layers,
        "dec_layers": dec_layers,
        "nheads": nheads,
        "camera_names": camera_names,
        "multi_gpu": multi_gpu,
    }

    print(f"\n=== Single-dataset training ===")
    train_dataloader, val_dataloader, stats, _ = load_data_dvrk(
        dataset_dirs[0],
        num_episodes_list[0],
        camera_names,
        batch_size_train,
        batch_size_val,
        task_configs_list[0],
        chunk_size=chunk_size,
        use_language=use_language
    )

    if use_language:
        tokenizer, model = initialize_model_and_tokenizer(language_encoder)
        assert tokenizer is not None and model is not None

    # load policy and stats
    policy = ACTPolicy(policy_config)
    model_state_dict = torch.load(ckpt_path)["model_state_dict"]
    loading_status = policy.deserialize(model_state_dict)
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = "/home/grayson/surpass/srth-surpass/ckpt_srth_grasp_only/dataset_stats.pkl"
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
    
    # inference testing
    policy.cuda()
    with torch.inference_mode():
        policy.eval()
        for batch_idx, data in enumerate(val_dataloader):
            action_hat_norm = forward_pass(data, policy, no_qpos=no_qpos)
            action_hat_raw = post_process(action_hat_norm)
            pass

if __name__=="__main__":
    # main()
    from low_level_policy import LowLevelPolicy
    from omegaconf import OmegaConf

    lr_backbone = 1e-5
    lr = 1e-5
    chunk_size = 60
    kl_weight = 10
    hidden_dim = 512
    dim_feedforward = 3200
    image_encoder = "efficientnet_b3film"

    enc_layers = 4
    dec_layers = 7
    nheads = 8

    args = OmegaConf.create({
        "ckpt_dir": "/home/grayson/surpass/srth-surpass/ckpt_srth_grasp_only/policy_epoch_6000_seed_0.ckpt",
        "policy_class": "ACT",
        "task_name": "surpass_grasp_only",
        "seed": 42,
        "use_language": True,
        "num_epochs": 20000,
        "lr": lr,
        "chunk_size": chunk_size,
        "action_dim": 20,
        "kl_weight": kl_weight,
        "hidden_dim": hidden_dim,
        "dim_feedforward": dim_feedforward,
        "lr_backbone": lr_backbone,
        "backbone": image_encoder,
        "enc_layers": enc_layers,
        "dec_layers": dec_layers,
        "nheads": nheads,
    })
    policy = LowLevelPolicy(args)
    policy.run()
    main()