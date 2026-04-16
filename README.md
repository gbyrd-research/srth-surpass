# DEBUG INFERENCE BRANCH

This branch was used to verify that the refactored code performs the same as the old code. This is the old code.

To determine if the performance was the same, we conducted the following experiment:

1. Train a model on the same, grasp only dataset using both the old and refactored code.
2. Create a deterministic dataset using this repository by running the `src/srth_new/low_level_policy/train_debug_collect_dataset.py` file. You must provide the checkpoint to your trained model using this repository:

```bash
train.resume_checkpoint=/home/grayson/surpass/srth-new/saved_runs/new_compare_grasp_only_dataset/checkpoints/train_step_3950.ckpt
```
3. This will create a deterministic dataset where each sample contains:

  1. Images
  2. Qpos
  3. Ground Truth Actions
  4. Padding
  5. Command Text
  6. Predicted Actions from the refactored model

4. Next, copy the root location of that dataset and run the code in the old repository (srth-surpass) using the below file and example arguments:

```bash
python src/act/inference_debug.py \
  --ckpt_dir /home/grayson/surpass/srth-surpass/src/act/ckpt_dir/surpass_grasp_only/policy_epoch_4000_seed_0.ckpt \
  --policy_class ACT \
  --task_name surpass_grasp_only \
  --seed 42 \
  --num_epochs 20000 \
  --root_dir /home/grayson/surpass/srth-new/outputs/low_level_policy/train/2026-04-16/10-33-07/deterministic_comparison_dataset
```

5. This will print out the total loss for the training and validation splits of the deterministic dataset for both the new and the old code. If both repositories are similar in performance, that means the refactor is likely to be successful and correspondent to the old code. For example, here is our output from when we ran the experiment:

```bash
Comparing split: val
val raw action debug - | NEW: First Pred: 0.011569 Total: 0.722174 | OLD: First Pred: 0.020387 Total: 0.800820 | COMPARE: First Pred: 0.009975 Total: 0.239514
```

The NEW and OLD are very similar, giving us strong confidence that the refactor was completed successfully.

# SRT-H (Surgical Robot Transformer - Hierarchy)


This is the repo for Hierarchical Surgical Robot Transformer.


If you encountered any issue, feel free to contact jchen396 (at) jh (dot) edu

## Installation
1. Clone this repository
```bash
git clone git@github.com:JuoTungChen/srth.git
cd srth
```

2. Create a virtual environment
```bash 
conda create -n srth python=3.8.10 
conda activate srth
```

3. Install packages
```bash
pip install -r requirements_ll.txt
```

(Optional)
1. Install [Whisper](https://github.com/openai/whisper)
```bash
sudo apt update && sudo apt install ffmpeg
```

2.  Install package for audio recording
```bash
sudo apt install portaudio19-dev python3-pyaudio
```


## Adding Path variables in bashrc
To avoid Python module import errors, please add the following path variables to the `~/.bashrc` file. This is useful for adding `sys.path.append("$PATH_TO_YAY_ROBOT/src")` to avoid Python module import errors.

export PATH_TO_YAY_ROBOT=[path to the current path]
export PATH_TO_SKAY_ROBOT=[path to the current path]
export PATH_TO_DATASET=[path to dataset folders]
export YOUR_CKPT_PATH="$PATH_TO_YAY_ROBOT/model_ckpts"

source /home/iulian/anaconda3/bin/activate aloha

# Training and Evaluation

## Train Low-Level Policy

### Relevant files
 ``` 
.
├── src                                  # main packages
|   └── act                              # where the low level policy code are stored
|   |   ├── dvrk_scripts              
|   |   |   ├── constants_dvrk.py        # the task configs for training
|   |   ├── generic_dataset.py           # dataset class
|   |   ├── auto_training_suturing.py    # script for initiating training
|   |   ├── imitate_episodes.py          # training class and functions
|   |   └── img_aug.py                   # class for image data augementations
├── script                               # useful scripts
|   └── chole                            # useful scripts we wrote for chole or other projects 
|   |   ├── calculate_std_mean.py        # script for calculating norm stats
|   └── encode_instruction.py                  # generate candidate text embeddings before training
...
 ``` 

### Data format
To train a policy, make sure your data are stored in following structure:


 ``` 
 $PATH_TO_DATASET
├── [DATASET_NAME]       # the dataset base dir
|   └── tissue_1                      # data subset
|   |   ├── 1_[task_name]             # task name
|   |   |   ├── [episode]             # should be timestamp when the data was recorded
|   |   |   |      ├── left_img_dir   # left endoscope cam images (frame000000_left.jpg)
|   |   |   |      ├── right_img_dir  # right endoscope cam images (frame000000_right.jpg)
|   |   |   |      ├── endo_psm1      # right wrist cam images (frame000000_psm1.jpg)
|   |   |   |      ├── endo_psm2      # left wrist cam images (frame000000_psm2.jpg)
|   |   |   |      └── ee_csv.csv     # kinematics
|   └── tissue_2                      # data subset
|   |   ├── 1_[task_name]             # task name
|   |   |   ├── [episode]             # should be timestamp when the data was recorded
|   |   |   |      ├── left_img_dir   # left endoscope cam images (frame000000_left.jpg)
|   |   |   |      ├── right_img_dir  # right endoscope cam images (frame000000_right.jpg)
|   |   |   |      ├── endo_psm1      # right wrist cam images (frame000000_psm1.jpg)
|   |   |   |      ├── endo_psm2      # left wrist cam images (frame000000_psm2.jpg)
|   |   |   |      └── ee_csv.csv     # kinematics
...
 ``` 

### Training procedure
After making sure the folder structure are correct, follow the following steps to train your low-level policy
1. run the following file to generate candidate embeddings before training
```
python encode_instruction.py --dataset_dir $PATH_TO_DATASET/[DATASET_NAME] --encoder distilbert --from_count
```
This should generate a json file called "candidate_embeddings_distilbert.json" containing the embeddings for the task names as well as embeddings for the direction correction commands.

2. calcuate the std and mean using the following script:
```
python script/chole/calculate_std_mean.py
```
Make sure to specify the tissue ids and data_dir before running the script.
The result will be shown on the terminal, and also stored in script/chole/std_mean.txt 

3. Next, copy and paste the max, min, std, and mean to the corresponding task config you created in src/act/dvrk_scripts/constants_dvrk.py 

4. Set the task configs in src/act/dvrk_scripts/constants_dvrk.py.
In the task config, there are several things you need to specify:

    a. __`dataset_dir`__: which dataset you want to use and where it's stored.

    b. __`num_episodes`__: total num of episodes, should be the same as printed by calculate_std_mean.py

    c. __`use_auto_label`__: if set to True, we will use directional instruction (i.e, "move left arm to the right") when sampling from recovery folders.

    d. __`tissue_samples_ids`__: the tissue ids you want to train on.

    e. __`camera_file_suffixes`__: the suffixes of the image files, should be the same size as the camera names.

    f. __`camera_names`__: the cameras you want to use (possible option: left, right, left_wrist, right_wrist).

    g. __`episode_len`__: don't need to worry about this.

    h. __`cutting_action_pad_size`__: this is for the cutting task in chole where we augment the kinematics to be able to close the scissor. not relevant for other tasks.

    i. __`recovery_ratio`__: the amount of recovery demos versus perfect demos you want to use. Typically set to 1.0 to use all of the recovery demos we have.

    j. __`action_mode`__: action representation and the norm stats. possible action mode: "hybrid", "ego", "relative_endoscope". We typically use hybrid.

    k. __`norm_scheme`__: the normalization scheme to use. possible options: "std", "min_max". We typically use "std".

    l. __`save_frequency`__: the frequency you wish to save the checkpoints.

    m. __`merging_subtasks`__: not relevant.

    n. __`phantom`__: not relevant.

    o. __`no_qpos`__: this is only set to True when you set the policy_class to "SRT" (a ACT variant with decoder only structure).

    Task specific settings:
    For suturing task you can also set the following configs:

    a. __`goal_condition_style`__ - "plot" means it will plot the labeled needle insertion point on the image during training.
 
5. create your own or modify auto_training.py. (example can be seen in auto_training_suturing.py). This file will initiate a subprocess for training and if the training is interupted by segmentation fault it will wait for a few seconds then continue from the last checkpoint. You can set some training parameters:

    a. __`task_name`__: should be corresponding to the task config you created in src/act/dvrk_scripts/constants_dvrk.py.

    b. __`ckpt_dir`__: the checkpoint name.

    c. __`policy_class`__: typically use "ACT". possible options: ["ACT", "SRT", "Diffusion"].

    d. __`batch_size`__: typically set to 16, which takes around 21GB of GPU memory on RTX4090.

    e. __`num_epochs`__: total num of epochs to train for.

    f. __`use_language`__: if set, use FiLM for vision backbones.

    g. __`language_encoder`__: we typically use "distilbert". possible options: ["distilbert", "clip"].

    h. __`image_encoder`__: we typically use "efficientnet_b3film". default is 'resnet18'. possible options: ['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b3', 'resnet18film', 'resnet34film', 'resnet50film','efficientnet_b0film', 'efficientnet_b3film', 'efficientnet_b5film'].

    i. __`gpu`__: which gpu to use (if you have multiple gpu).

    j. __`multi_gpu`__: if set to True will use all the CUDA_VISIBLE_DEVICES for training.

    k. __`policy_level`__: should use "low" for training low level policy.




## Train High-Level Policy
### Data
- Set DATA_DIR to the correct dataset folder in skay_jhu_private/src/instructor/constants_daVinci.py
    - There you can also define same training constants – e.g. the camera folder names, val/test tissues, ..
- Labels will be directly read from the directory names (orientate on the directory structure from the SRT-H chole dataset
- Compute the dataset mean and std for standardization using skay_jhu_private/script/chole/dataset_rgb_mean_std.py
    - Alternatively use the mean and std from imagenet – might work little worse

### Data Curation:
- Check if all recordings look good:
    - Create a video with all demonstrations concatenated for one tissue and check if all demonstrations look good and are in the correct task directory script/chole/concatenate_all_tissue_demos.py (sometimes it happens that a demonstration is saved in the previous task folder (or vice versa) 
    - Specifically check that the demonstrations are complete. If the demonstrations are incomplete (started to late or ended to early), then concatenating the task recordings is erroneous
- If a task recording started too early or is too long, you can add a “indices_curated.json” in the demonstration directory with the keys “start” and/or “end” giving them the frame index of the curated start/end

### Files 
Models:	
- model_daVinci.py -> Contains all temporal model code (Transformer, ..)
- backbone_models_daVinci.py 🡪 Backbone models that can be selected in models – e.g. ResNet, SwinT, ..
- dataset_daVinci.py contains the dataset/dataloader code (loads defined data, applies augmentations, ..)
- Concatenates the recordings of neighboring tasks ..
- Start training using example commands in the training_configs directory (this will call the train() function in train.py with example parameters
- Inference via instructor_pipeline.py
    - Select the ckpt you just trained

### Ignore:
future_frame_predictor_model.py
hl_correction_publisher_ui_w_whisper.py
temporal_models.py -> here only TCN included (not the Transformer architecture that was used in the end)

Note: Many features were experimental so the dataset/model code contains many features that can be removed -> to clean/simplify the code 
