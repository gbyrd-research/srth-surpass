"""
Example usage:

# Training full model
python src/instructor/train_daVinci.py --dataset_names base_chole_clipping_cutting base_chole_clipping_cutting_amos --ckpt_dir $YOUR_CKPT_PATH/hl/swin-t_second_split --gpu 2 --recovery_probability 0.85 --batch_size 16 --num_epochs 2000 --lr 4e-4 --min_lr 5e-5 --lr_cycle 50 --warmup_epochs 5 --weight_decay 0.05 --validation_interval 4 --prediction_offset 15 --history_len 4 --history_step_size 20 --one_hot_flag --early_stopping_interval 200 --seed 5 --load_best_ckpt_flag --plot_val_images_flag --max_num_images 2 --cameras_to_use left_img_dir --backbone_model swin-t --model_init_weights imagenet --image_dim 224 224 --freeze_backbone_until none --multitask_loss_weight 0.4 --uniform_sampling_flag --extra_repeated_phase_last_frame_sampling_flag --extra_repeated_phase_last_frame_sampling_probability 0.15 --add_center_crop_view_flag --log_wandb --extra_corrections_sampling_flag --extra_corrections_sampling_probability 0.15 --val_split_number 0
"""

import os
import argparse
import threading
import sys
import logging

import torch
import cv2
import torch.optim as optim
import numpy as np
import wandb
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms as transforms
from torchvision.transforms import v2
import albumentations as A
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score

# import aloha 
path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")
# from aloha_pro.aloha_scripts.utils import crop_resize, random_crop, initialize_model_and_tokenizer, encode_text
from aloha_pro.aloha_scripts.utils import memory_monitor
from instructor.dataset_daVinci import load_merged_data, SequenceDataset
from instructor.model_daVinci import Instructor
from instructor.utils import set_seed, lr_lambda, multitask_criterion_fct, phase_criterion_fct
from instructor.constants_daVinci import DATASET_CONFIGS # get dataset parameters


def train(model, dataloader, optimizer, criterion, multitask_criterion, device, ckpt_dir, current_epoch, max_num_images=10, multitask_loss_weight=0.2):
    model.train()
    total_loss = total_next_phase_pred_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        # Get the data from the batch
        if args.use_jaw_values_flag:
            images, _, commands, psm2_psm1_jaw_values, phase_history, multitask_label_indices_dict = batch 
        else:
            images, _, commands, phase_history, multitask_label_indices_dict = batch
            psm2_psm1_jaw_values = None
    
        # Prepare model input
        images = images.to(device)
        if args.use_jaw_values_flag:
            psm2_psm1_jaw_values = psm2_psm1_jaw_values.to(device)
        output_phase_flag = args.use_phase_history_flag or args.use_phase_history_for_moving_direction_and_corr_pred_flag
        if output_phase_flag:
            phase_history_indexed = [[model.history_phase_to_index[phase_command_list[batch_idx]] for batch_idx in range(len(phase_command_list))] for phase_command_list in phase_history]
            phase_history_indexed = torch.tensor(phase_history_indexed, device=device)

        # Forward pass
        optimizer.zero_grad()
        if args.use_jaw_values_flag and output_phase_flag:
            logits, _, multitask_logits_dict, temperature = model(images, psm2_psm1_jaw_values, phase_history_indexed)
        elif args.use_jaw_values_flag:
            logits, _, multitask_logits_dict, temperature = model(images, psm2_psm1_jaw_values=psm2_psm1_jaw_values)
        elif output_phase_flag:
            logits, _, multitask_logits_dict, temperature = model(images, phase_history=phase_history_indexed)
        else:
            logits, _, multitask_logits_dict, temperature = model(images)

        # Convert ground truth command strings to indices using the pre-computed dictionary
        commands_idx = torch.tensor([model.command_to_index[cmd] for cmd in commands], device=device)
        multitask_label_indices_dict = {multitask: label_indices.to(device) for multitask, label_indices in multitask_label_indices_dict.items()}

        # Compute the loss
        next_phase_pred_loss = criterion(logits, commands_idx)
        if multitask_logits_dict:
            multitask_loss = multitask_criterion(multitask_logits_dict, multitask_label_indices_dict) 
            loss = next_phase_pred_loss * (1-multitask_loss_weight) + multitask_loss * multitask_loss_weight
        else:
            loss = next_phase_pred_loss
        loss.backward()
        optimizer.step()

        total_next_phase_pred_loss += next_phase_pred_loss.item()
        total_loss += loss.item()

        if args.log_wandb:
            wandb.log({"Train Loss": loss.item(), "Train (Phase Pred Loss)": next_phase_pred_loss.item(), "Temperature": temperature.item()})
            
        # Save images from the last batch (to see, e.g., the augmentation applied)
        saved_img_cnt = 0
        rnd_batch_idx = np.random.randint(0, len(dataloader))
        if batch_idx == rnd_batch_idx:
            for input_idx in range(len(images)):
                if saved_img_cnt >= max_num_images:
                    break
                
                gt = commands[input_idx]
                pred = model.decode_logits(logits[input_idx].unsqueeze(0), temperature)[0]
                pred_prob = torch.nn.functional.softmax(logits[input_idx], dim=0)[model.command_to_index[pred]].item()

                save_path = os.path.join(ckpt_dir, "training_images", f"epoch_{current_epoch=}_{batch_idx=}_{input_idx=}.jpg")
                if multitask_label_indices_dict:
                    multitask_gt = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(multitask_label_indices_dict, batch_wise=True, logits_flag=False, img_idx=input_idx)
                    multitask_preds = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(multitask_logits_dict, batch_wise=True, logits_flag=True, img_idx=input_idx)
                    multitask_probs = {multitask: torch.nn.functional.softmax(multitask_logits[input_idx], dim=0)[torch.argmax(torch.nn.functional.softmax(multitask_logits[input_idx], dim=0))].item() for multitask, multitask_logits in multitask_logits_dict.items()} if multitask_logits_dict else None
                else:
                    multitask_gt = multitask_preds = multitask_probs = None
                curr_psm2_psm1_jaw_values = psm2_psm1_jaw_values[input_idx] if args.use_jaw_values_flag else None
                curr_phase_history = phase_history[input_idx] if output_phase_flag else None
                log_combined_image(images[input_idx], gt, pred, save_path=save_path, pred_prob=pred_prob, multitask_gt=multitask_gt, multitask_preds=multitask_preds, multitask_probs=multitask_probs, 
                                   psm2_psm1_jaw_values=curr_psm2_psm1_jaw_values, phase_history=curr_phase_history)
                
                if args.log_wandb:
                    wandb.log({f"Training Image {saved_img_cnt}": wandb.Image(save_path, caption=f"Epoch {current_epoch}, Batch {batch_idx}, Image {input_idx}")})

                saved_img_cnt += 1
            
    return total_loss / len(dataloader), total_next_phase_pred_loss / len(dataloader)


def evaluate(model, dataloader, criterion, multitask_criterion, device, multitask_loss_weight=0.2):
    model.eval()
    total_loss = total_next_phase_pred_loss = 0.0
    with torch.no_grad():
        for batch in dataloader: 
            # Get the data from the batch
            if args.use_jaw_values_flag:
                images, _, commands, psm2_psm1_jaw_values, phase_history, multitask_label_indices_dict = batch 
            else:
                images, _, commands, phase_history, multitask_label_indices_dict = batch
                psm2_psm1_jaw_values = None
        
            # Prepare model input
            images = images.to(device)
            if args.use_jaw_values_flag:
                psm2_psm1_jaw_values = psm2_psm1_jaw_values.to(device)
            output_phase_flag = args.use_phase_history_flag or args.use_phase_history_for_moving_direction_and_corr_pred_flag
            if output_phase_flag:
                phase_history_indexed = [[model.history_phase_to_index[phase_command_list[batch_idx]] for batch_idx in range(len(phase_command_list))] for phase_command_list in phase_history]
                phase_history_indexed = torch.tensor(phase_history_indexed, device=device)

            # Forward pass
            optimizer.zero_grad()
            if args.use_jaw_values_flag and output_phase_flag:
                logits, _, multitask_logits_dict, temperature = model(images, psm2_psm1_jaw_values, phase_history_indexed)
            elif args.use_jaw_values_flag:
                logits, _, multitask_logits_dict, temperature = model(images, psm2_psm1_jaw_values=psm2_psm1_jaw_values)
            elif output_phase_flag:
                logits, _, multitask_logits_dict, temperature = model(images, phase_history=phase_history_indexed)
            else:
                logits, _, multitask_logits_dict, temperature = model(images)

            # Convert ground truth command strings to indices using the pre-computed dictionary
            commands_idx = torch.tensor([model.command_to_index[cmd] for cmd in commands], device=device)
            multitask_label_indices_dict = {multitask: label_indices.to(device) for multitask, label_indices in multitask_label_indices_dict.items()}

            # Compute the loss
            next_phase_pred_loss = criterion(logits, commands_idx)
            if multitask_label_indices_dict:
                multitask_loss = multitask_criterion(multitask_logits_dict, multitask_label_indices_dict) 
                loss = next_phase_pred_loss * (1-multitask_loss_weight) + multitask_loss * multitask_loss_weight
            else:
                loss = next_phase_pred_loss
            total_next_phase_pred_loss += next_phase_pred_loss.item()
            total_loss += loss.item()

            if args.log_wandb:
                wandb.log({"Eval Loss": loss.item(), "Train (Phase Pred Loss)": next_phase_pred_loss.item(), "Temperature": temperature.item()})
        
    return total_loss / len(dataloader), total_next_phase_pred_loss / len(dataloader)


def test(model, dataloader, split_name, device, current_epoch, one_hot_flag, ckpt_dir,
         max_num_images = 10, plot_images_flag=True, log_wandb_flag=True):

    model.eval()

    all_commands_gt = []
    all_decoded_texts = []
    all_commands_last_pred = [] # To evaluate the phase transitions accuracy

    if not one_hot_flag:
        all_predicted_embeddings = []
        all_gt_embeddings = []

    if plot_images_flag:
        incorrect_img_cnt_phase = correct_img_cnt_phase = 0
        incorrect_img_cnt_correction = correct_img_cnt_correction = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Get the data from the batch
            if args.use_jaw_values_flag:
                images, command_embedding_gt, command_gt, psm2_psm1_jaw_values, phase_history, multitask_label_indices_dict = batch 
            else:
                images, command_embedding_gt, command_gt, phase_history, multitask_label_indices_dict = batch
                psm2_psm1_jaw_values = None
        
            # Prepare model input
            images = images.to(device)
            if args.use_jaw_values_flag:
                psm2_psm1_jaw_values = psm2_psm1_jaw_values.to(device)
            output_phase_flag = args.use_phase_history_flag or args.use_phase_history_for_moving_direction_and_corr_pred_flag
            if output_phase_flag:
                phase_history_indexed = [[model.history_phase_to_index[phase_command_list[batch_idx]] for batch_idx in range(len(phase_command_list))] for phase_command_list in phase_history]
                phase_history_indexed = torch.tensor(phase_history_indexed, device=device)

            # Forward pass + decode logits
            optimizer.zero_grad()
            if args.use_jaw_values_flag and output_phase_flag:
                logits, predicted_embedding, multitask_logits_dict, temperature = model(images, psm2_psm1_jaw_values, phase_history_indexed)
            elif args.use_jaw_values_flag:
                logits, predicted_embedding, multitask_logits_dict, temperature = model(images, psm2_psm1_jaw_values=psm2_psm1_jaw_values)
            elif output_phase_flag:
                logits, predicted_embedding, multitask_logits_dict, temperature = model(images, phase_history=phase_history_indexed)
            else:
                logits, predicted_embedding, multitask_logits_dict, temperature = model(images)
            decoded_texts = model.decode_logits(logits, temperature) 

            # Store the ground truth and predicted commands for the confusion matrix
            all_commands_gt.extend(command_gt)
            all_decoded_texts.extend(decoded_texts)
            # Store the last predicted command for the phase transitions
            all_commands_last_pred.extend(phase_history[-1])
            
            if not one_hot_flag:
                all_predicted_embeddings.extend(predicted_embedding.cpu().numpy())
                all_gt_embeddings.extend(command_embedding_gt.cpu().numpy())

            if batch_idx == 0:
                all_multitask_gt_labels_dict = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(multitask_label_indices_dict, batch_wise=True, logits_flag=False)
                all_multitask_preds_dict = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(multitask_logits_dict, batch_wise=True, logits_flag=True)
            else:
                curr_batch_multitask_gt_labels_dict = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(multitask_label_indices_dict, batch_wise=True, logits_flag=False)
                curr_batch_multitask_preds_dict = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(multitask_logits_dict, batch_wise=True, logits_flag=True)
                for multitask in multitask_label_indices_dict:
                    all_multitask_gt_labels_dict[multitask].extend(curr_batch_multitask_gt_labels_dict[multitask])
                    all_multitask_preds_dict[multitask].extend(curr_batch_multitask_preds_dict[multitask])

            # Plot correct and incorrect phase predictions inputs for current batch
            if plot_images_flag and (incorrect_img_cnt_phase < max_num_images or correct_img_cnt_phase < max_num_images):
                rnd_indices = list(torch.randperm(len(images))) # Randomly shuffle the indices - to get random images
                plot_images_for_task(images, logits, command_gt, decoded_texts, ckpt_dir, current_epoch, batch_idx, rnd_indices, 
                                    multitask_label_indices_dict, multitask_logits_dict, psm2_psm1_jaw_values, phase_history, 
                                    task_name="phase_predictions", model=model, plot_images_flag=plot_images_flag, 
                                    max_num_images=max_num_images, args=args, correct_img_cnt=correct_img_cnt_phase, 
                                    incorrect_img_cnt=incorrect_img_cnt_phase, device=device)

            # Plot correct and incorrect is_correction predictions inputs for current batch
            if plot_images_flag and (incorrect_img_cnt_correction < max_num_images or correct_img_cnt_correction < max_num_images):
                rnd_indices = list(torch.randperm(len(images))) # Randomly shuffle the indices - to get random images
                plot_images_for_task(images, multitask_logits_dict['is_correction'], all_multitask_gt_labels_dict['is_correction'], 
                                    all_multitask_preds_dict['is_correction'], ckpt_dir, current_epoch, batch_idx, rnd_indices, 
                                    multitask_label_indices_dict, multitask_logits_dict, psm2_psm1_jaw_values, phase_history, 
                                    task_name="is_correction", model=model, plot_images_flag=plot_images_flag, 
                                    max_num_images=max_num_images, args=args, correct_img_cnt=correct_img_cnt_correction, 
                                    incorrect_img_cnt=incorrect_img_cnt_correction, device=device)
                               
    # Visualize embeddings
    # TODO: Fix if required
    # if not one_hot_flag:
    #     # Save the t-SNE visualization of the embeddings of the last batch
    #     tnse_plots_folder_path = os.path.join(ckpt_dir, "tsne_plots")
    #     if not os.path.exists(tnse_plots_folder_path):
    #         os.makedirs(tnse_plots_folder_path, exist_ok=True)
    #     save_path = os.path.join(tnse_plots_folder_path, f"embeddings_tsne_epoch_{epoch}.png")
    #     log_tsne_plot(candidate_embeddings, candidate_texts, all_predicted_embeddings, all_decoded_texts, all_gt_embeddings, current_epoch, save_path)

    # Create and save confusion matrix
    conf_matrix_folder_path = os.path.join(ckpt_dir, "confusion_matrices")
    task_name = "phase_predictions"
    conf_matrix_phase_pred_folder_path = os.path.join(conf_matrix_folder_path, task_name)
    if not os.path.exists(conf_matrix_phase_pred_folder_path):
        os.makedirs(conf_matrix_phase_pred_folder_path, exist_ok=True)
    save_path = os.path.join(conf_matrix_phase_pred_folder_path, f"{split_name}_confusion_matrix_epoch_{current_epoch}.png")
    log_confusion_matrix(task_name, all_commands_gt, all_decoded_texts, candidate_texts, split_name, current_epoch, save_path, log_wandb_flag)
    
    # Save confusion matrix for corrections and moving direction
    for multitask in ["is_correction", "dominant_moving_direction"]:
        conf_matrix_multitask_folder_path = os.path.join(conf_matrix_folder_path, multitask)
        if not os.path.exists(conf_matrix_multitask_folder_path):
            os.makedirs(conf_matrix_multitask_folder_path, exist_ok=True)
        save_path = os.path.join(conf_matrix_multitask_folder_path, f"{split_name}_confusion_matrix_{multitask}_epoch_{current_epoch}.png")
        multitask_possible_labels = SequenceDataset.get_all_multitask_labels(multitask)
        log_confusion_matrix(multitask, all_multitask_gt_labels_dict[multitask], all_multitask_preds_dict[multitask], multitask_possible_labels, split_name, current_epoch, save_path, log_wandb_flag)    
    
    # Compute metrics 
    logger.info("")
    compute_metrics(current_epoch, all_commands_gt, all_decoded_texts, all_commands_last_pred, args, logger, all_multitask_gt_labels_dict, all_multitask_preds_dict)
    
# ----------------------------

def plot_images_for_task(images, logits, command_gt, decoded_texts, ckpt_dir, current_epoch, batch_idx, 
                         rnd_indices, multitask_label_indices_dict, multitask_logits_dict, psm2_psm1_jaw_values, 
                         phase_history, task_name, model, plot_images_flag, max_num_images, args, correct_img_cnt, 
                         incorrect_img_cnt, device):
    
    for rnd_input_idx in rnd_indices:            
        gt, pred = command_gt[rnd_input_idx], decoded_texts[rnd_input_idx]
        all_multitasks = SequenceDataset.get_all_multitask_names()
        if task_name in all_multitasks:
            pred_idx = SequenceDataset.get_multitask_index(task_name, pred)
            pred_prob = torch.nn.functional.softmax(multitask_logits_dict[task_name][rnd_input_idx], dim=0)[pred_idx].item()

        else:
            pred_prob = torch.nn.functional.softmax(logits[rnd_input_idx], dim=0)[model.command_to_index[pred]].item()
        
        # Save incorrect prediction 
        if pred != gt and incorrect_img_cnt < max_num_images:
            incorrect_img_cnt += 1
            save_path = os.path.join(ckpt_dir, "predictions", task_name, "incorrect", f"{current_epoch=}_incorrect_{batch_idx=}_{rnd_input_idx}.jpg")
            folder_path = os.path.join(ckpt_dir, "predictions", task_name, "incorrect")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            if multitask_label_indices_dict:
                multitask_gt = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(multitask_label_indices_dict, batch_wise=True, logits_flag=False, img_idx=rnd_input_idx)
                multitask_preds = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(multitask_logits_dict, batch_wise=True, logits_flag=True, img_idx=rnd_input_idx)
                multitask_probs = {multitask: torch.nn.functional.softmax(multitask_logits[rnd_input_idx], dim=0)[torch.argmax(torch.nn.functional.softmax(multitask_logits[rnd_input_idx], dim=0))].item() for multitask, multitask_logits in multitask_logits_dict.items()} if multitask_logits_dict else None
            else:
                multitask_gt = multitask_preds = multitask_probs = None
            curr_psm2_psm1_jaw_values = psm2_psm1_jaw_values[rnd_input_idx] if args.use_jaw_values_flag else None
            curr_phase_history = phase_history[rnd_input_idx] if phase_history is not None else None
            log_combined_image(images[rnd_input_idx], gt, pred, save_path=save_path, pred_prob=pred_prob, multitask_gt=multitask_gt, multitask_preds=multitask_preds, multitask_probs=multitask_probs, 
                            psm2_psm1_jaw_values=curr_psm2_psm1_jaw_values, phase_history=curr_phase_history)
            if args.log_wandb:
                wandb.log({f"Incorrect Prediction ({task_name})": wandb.Image(save_path, caption=f"Epoch {current_epoch}, Batch {batch_idx}, Image {rnd_input_idx}")})
        
        # Save correct prediction
        if pred == gt and correct_img_cnt < max_num_images:
            correct_img_cnt += 1
            save_path = os.path.join(ckpt_dir, "predictions", task_name, "correct", f"epoch_{current_epoch}_correct_{batch_idx}_{rnd_input_idx}.jpg")
            folder_path = os.path.join(ckpt_dir, "predictions", task_name, "correct")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            if multitask_label_indices_dict:
                multitask_gt = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(multitask_label_indices_dict, batch_wise=True, logits_flag=False, img_idx=rnd_input_idx)
                multitask_preds = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(multitask_logits_dict, batch_wise=True, logits_flag=True, img_idx=rnd_input_idx)
                multitask_probs = {multitask: torch.nn.functional.softmax(multitask_logits[rnd_input_idx], dim=0)[torch.argmax(torch.nn.functional.softmax(multitask_logits[rnd_input_idx], dim=0))].item() for multitask, multitask_logits in multitask_logits_dict.items()} if multitask_logits_dict else None
            else:
                multitask_gt = multitask_preds = multitask_probs = None
            curr_psm2_psm1_jaw_values = psm2_psm1_jaw_values[rnd_input_idx] if args.use_jaw_values_flag else None
            curr_phase_history = phase_history[rnd_input_idx] if phase_history is not None else None
            log_combined_image(images[rnd_input_idx], gt, pred, save_path=save_path, pred_prob=pred_prob, multitask_gt=multitask_gt, multitask_preds=multitask_preds, multitask_probs=multitask_probs, 
                            psm2_psm1_jaw_values=curr_psm2_psm1_jaw_values, phase_history=curr_phase_history)
            if args.log_wandb:
                wandb.log({f"Correct Prediction ({task_name})": wandb.Image(save_path, caption=f"Epoch {current_epoch}, Batch {batch_idx}, Image {rnd_input_idx}")})


def compute_metrics(current_epoch, all_commands_gt, all_decoded_texts, all_commands_last_pred, args, logger, all_multitask_gt_labels_dict=None, all_multitask_preds_dict=None):
    # Compute the success rate -> accuracy
    accuracy_curr_epoch = accuracy_score(all_commands_gt, all_decoded_texts)
    if args.log_wandb:
        wandb.log({"Accuracy": accuracy_curr_epoch})
        
    # Compute the (macro) F1 score
    f1_score_curr_epoch = f1_score(all_commands_gt, all_decoded_texts, average='macro')
    if args.log_wandb:
        wandb.log({"F1 Score": f1_score_curr_epoch})
        
    logger.info(f"Epoch {current_epoch}: Accuracy = {accuracy_curr_epoch * 100:.2f}% - F1 Score = {f1_score_curr_epoch * 100:.2f}%")
        
    # ---------- Metrics at phase transitions (so only where gt != last pred) ----------
    
    # Keep only the transition inputs
    transition_filter = [gt != pred for gt, pred in zip(all_commands_gt, all_commands_last_pred)]
    all_commands_gt_filtered = [gt for gt, filter_val in zip(all_commands_gt, transition_filter) if filter_val]
    all_decoded_texts_filtered = [pred for pred, filter_val in zip(all_decoded_texts, transition_filter) if filter_val]
    
    # Compute the success rate -> accuracy
    accuracy_curr_epoch_transitions = accuracy_score(all_commands_gt_filtered, all_decoded_texts_filtered)
    if args.log_wandb:
        wandb.log({"Accuracy (at transitions)": accuracy_curr_epoch_transitions})
    
    # Compute the (macro) F1 score
    f1_score_curr_epoch_transitions = f1_score(all_commands_gt_filtered, all_decoded_texts_filtered, average='macro')
    if args.log_wandb:
        wandb.log({"F1 Score (at transitions)": f1_score_curr_epoch_transitions})

    logger.info(f"Epoch {current_epoch}: Accuracy (at transitions) = {accuracy_curr_epoch_transitions * 100:.2f}% - F1 Score (at transitions) = {f1_score_curr_epoch_transitions * 100:.2f}%\n")


    # ------------------------- Multi-task metrics -------------------------
    
    if all_multitask_gt_labels_dict and all_multitask_preds_dict:
        # Compute the multi-task accuracy and F1 score
        multitask_accuracy, multitask_f1_score = {}, {}
        for multitask in all_multitask_gt_labels_dict:
            multitask_accuracy[multitask] = accuracy_score(all_multitask_gt_labels_dict[multitask], all_multitask_preds_dict[multitask])
            multitask_f1_score[multitask] = f1_score(all_multitask_gt_labels_dict[multitask], all_multitask_preds_dict[multitask], average='macro')

            if args.log_wandb:
                wandb.log({f"{multitask} Accuracy": multitask_accuracy[multitask]})
                wandb.log({f"{multitask} F1 Score": multitask_f1_score[multitask]})

            logger.info(f"Epoch {current_epoch}: {multitask} Accuracy = {multitask_accuracy[multitask] * 100:.2f}% - {multitask} F1 Score = {multitask_f1_score[multitask] * 100:.2f}%")
    

def log_combined_image(images, gt_text, pred_text, save_path=None, pred_prob=None, multitask_gt=None, multitask_preds=None, multitask_probs=None, 
                       psm2_psm1_jaw_values=None, phase_history=None):

    num_ts, num_cams, num_channels = images.shape[:3]
        
    # Use the segmentation mask as another camera
    if num_channels > 3:
        additional_channels = num_channels - 3
        if additional_channels == 3:
            images = torch.cat([images[:, :, :3], images[:, :, 3:]], dim=1)
            num_cams += 1
        elif additional_channels == 1 and len(args.seg_mask_objs) == 3:
            seg_mask_list = []
            for class_id in range(1, len(args.seg_mask_objs)+1):
                seg_mask = (images[:, :, 3] == class_id).float()
                seg_mask_list.append(seg_mask)
            seg_masks = torch.stack(seg_mask_list, dim=2)
            images = torch.cat([images[:, :, :3], seg_masks], dim=1)
            num_cams += 1
        else:
            images = images[:, :, :3] # NOTE: Currently only possible to work with 3 seg masks (as currently only required)
    
    if num_ts <= 6:
        # Extract frames for all timesteps and concatenate across width
        for t in range(num_ts):
            combined_image = torch.cat([images[t, cam_idx] for cam_idx in range(images.shape[1])], dim=-1)
            if t == 0: 
                combined_image_all = combined_image
            else:
                combined_image_all = torch.cat([combined_image_all, combined_image], dim=-2)
        combined_image = combined_image_all
    else:
        # Extract last frame and concatenate across width
        combined_image = torch.cat([images[-1, cam_idx] for cam_idx in range(images.shape[1])], dim=-1)

    # Convert to PIL image
    combined_image_pil = transforms.ToPILImage()(combined_image)

    # [Changed] Calculate the additional width needed for multitask predictions
    multitask_text_width = 1000  # Adjust based on expected text length
    total_width = combined_image_pil.width + multitask_text_width

    # [Changed] Create a blank canvas to add text and multitask predictions on the right side
    canvas = Image.new("RGB", (total_width, combined_image_pil.height), "black")
    canvas.paste(combined_image_pil, (0, 0))

    # Add GT and predicted text on the side
    draw = ImageDraw.Draw(canvas)
    num_ts_images = num_ts if num_ts <= 6 else 1
    font_size = 1.9 * num_ts_images * max((num_ts_images-2), 1)
    font = ImageFont.load_default(size=font_size)
    y_text_factor = 1.5  # Factor to increase the y_text position for each new line
    x_text = combined_image_pil.width + 10
    y_text = 3
    if multitask_gt and "dominant_moving_direction" in multitask_gt:
        gt_written_text = f"{gt_text} ({multitask_gt['dominant_moving_direction']})"
    else:
        gt_written_text = gt_text
    draw.text((x_text, y_text), "GT: " + gt_written_text, font=font, fill="white")
    y_text += y_text_factor*font_size
    pred_text_with_extras = f"{pred_text} ({pred_prob*100:.1f}%)" if pred_prob is not None else pred_text
    if gt_text != pred_text:
        draw.text((x_text, y_text), "Pred: " + pred_text_with_extras, font=font, fill="red")
    else:
        draw.text((x_text, y_text), "Pred: " + pred_text_with_extras, font=font, fill="green")

    # Add jaw values if provided
    if psm2_psm1_jaw_values is not None:
        y_text += 2*y_text_factor*font_size
        draw.text((x_text, y_text), f"Jaw Values (PSM2 - PSM1):\n", font=font, fill="white")
        y_text += y_text_factor*font_size
        for jaw_values in psm2_psm1_jaw_values:
            draw.text((x_text, y_text), f"{jaw_values.tolist()}", font=font, fill="white")
            y_text += y_text_factor*font_size
            
    # Add phase history if provided
    if phase_history is not None:
        y_text += 2*y_text_factor*font_size
        draw.text((x_text, y_text), f"Phase History:\n", font=font, fill="white") 
        y_text += y_text_factor*font_size
        for phase in phase_history:
            draw.text((x_text, y_text), f"{phase}", font=font, fill="white")
            y_text += font_size   

    # Add multitask predictions and probabilities if provided
    if multitask_preds is not None and multitask_probs is not None:
        y_text += y_text_factor*font_size  # Add extra space before multitask section
        for task_name, task_pred in multitask_preds.items():
            task_prob = multitask_probs.get(task_name, None)
            task_gt = multitask_gt.get(task_name, None) if multitask_gt is not None else None
            
            task_text = f"{task_name}: {task_pred} ({task_prob*100:.1f}%)" if task_prob is not None else f"{task_name}: {task_pred}"
            
            # Determine the color based on GT match
            if task_gt is not None:
                color = "green" if task_pred == task_gt else "red"
            else:
                color = "white"
                
            draw.text((x_text, y_text), task_text, font=font, fill=color)
            y_text += y_text_factor*font_size    

    # Save the combined image and text
    if save_path is not None:
        canvas.save(save_path)
        

def log_confusion_matrix(name, y_true, y_pred, classes, split_name=None, epoch=None, save_path=None, log_wandb_flag=True):
    """
    Compute the confusion matrix for each criteria.
    
    Args:
        y_true_all_criteria (torch.Tensor): True labels for all criteria
        y_pred_all_criteria (torch.Tensor): Predicted labels for all criteria
        split_name (str): Name of the split (e.g., "train", "val")
        epoch (int): Current epoch - If None, no epoch is logged (for final confusion matrix after training)
    """
    
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        
        # Create a new fig
        fig = plt.figure(figsize=(8, 8))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar(shrink=0.7)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        return fig
    
    # Log the confusion matrix with WandB
    if epoch is not None:
        fig = plot_confusion_matrix(confusion_matrix(y_true, y_pred, labels=classes), classes=classes, title=f"Confusion Matrix (Epoch {epoch})")
    else:
        fig = plot_confusion_matrix(confusion_matrix(y_true, y_pred, labels=classes), classes=classes, title=f"Confusion Matrix")
    if log_wandb_flag:
        if split_name is None:
            wandb.log({f"confusion_matrix_{name}": fig})
        else:
            wandb.log({f"{split_name=}_confusion_matrix_{name}": fig})
    
    # Save with the epoch in the filename
    plt.savefig(save_path)
    plt.close()


def log_tsne_plot(candidate_embeddings, candidate_commands, predicted_embeddings, predicted_commands, gt_embeddings, epoch, save_path):
    # TODO: If later required - fix it 

    # Convert lists to numpy arrays
    candidate_embeddings = np.array(candidate_embeddings)
    gt_embeddings = np.array(gt_embeddings)
    predicted_embeddings = np.array(predicted_embeddings)

    # Check that all predicted commands are within the candidate commands
    all_unique_commands_set = set(candidate_commands)
    all_unique_predicted_commands_set = set(predicted_commands)
    if not all_unique_predicted_commands_set.issubset(all_unique_commands_set):
        logger.info(f"\nCommands that are not in the candidate commands: {all_unique_predicted_commands_set - all_unique_commands_set}")
        raise ValueError("All predicted commands should be within the candidate commands")

    # Generate a color palette
    base_colors = sns.color_palette("husl", len(candidate_commands))
    color_map = {command: color for command, color in zip(candidate_commands, base_colors)}

    # Stack embeddings and apply t-SNE
    all_embeddings = np.vstack([predicted_embeddings, gt_embeddings, candidate_embeddings])
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Split the 2D embeddings back (not interested in the gt_embeddings, only for stability of the t-SNE)
    predicted_2d = embeddings_2d[: len(predicted_embeddings)]
    candidate_2d = embeddings_2d[len(predicted_embeddings):]

    # Plot the results
    plt.figure(figsize=(12, 10))

    # Plot candidate embeddings
    for i, command in enumerate(candidate_commands):
        plt.scatter(candidate_2d[i, 0], candidate_2d[i, 1], color=color_map[command], alpha= 1, label=f"{command}" if command not in candidate_commands[:i] else "")

    # Plot predicted embeddings
    for i, command in enumerate(predicted_commands):
        plt.scatter(predicted_2d[i, 0], predicted_2d[i, 1], color=color_map[command], alpha=0.5)

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(f"t-SNE Visualization of Embeddings (Epoch {epoch})")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Save with the epoch in the filename
    plt.savefig(save_path)

    # Log the image to wandb if logging is enabled
    if args.log_wandb:
        wandb.log({"t-SNE Visualization": [wandb.Image(save_path, caption=f"Epoch {epoch}")]})
    plt.close()

# -----------------------------

def build_instructor(history_len, history_step_size, prediction_offset, candidate_embeddings, candidate_texts, device, one_hot_flag, camera_names,
                     backbone_model_name, model_init_weights, freeze_backbone_until, global_pool_image_features_flag, use_jaw_values_flag, use_phase_history_flag, 
                     phase_history_len, temporal_mode, phase_to_instruction_mapping, phase_history_only_phase_switches_flag,
                     camera_dropout_prob=0.2, jaw_values_dropout_prob=0.2, phase_history_dropout_prob=0.2, image_dim=(224, 224),
                     llava_anyres_flag=False, no_llava_anyres_global_image_flag=False, wrist_images_rel_width=3/4, llava_anyres_rel_width=0.5,
                     selected_multitasks=[], use_seg_masks_input_flag=False, seg_mask_objs=["clips", "left_tube", "right_tube"], merge_seg_masks_flag=False,
                     seg_masks_dropout_prob=0.5, add_center_crop_view_flag=False, merge_global_and_center_embs_flag=False, distance_from_border_y=0.1,
                     distance_from_border_x=0.25, y_offset=-0.1, use_phase_history_for_moving_direction_and_corr_pred_flag=False, moving_direction_and_corr_history_len=2,
                     use_separate_backbones_flag=False, dataset_mean_std_file_names=None, num_transformer_heads=8, num_transformer_layers=6):
                     
    # Map command texts to indices
    command_to_index = {command: index for index, command in enumerate(candidate_texts)}

    # Build model
    candidate_embeddings = candidate_embeddings.to(device)
    model = Instructor(
        device=device,
        history_len=history_len,
        history_step_size=history_step_size,
        prediction_offset=prediction_offset,
        candidate_embeddings=candidate_embeddings,
        candidate_texts=candidate_texts,
        command_to_index=command_to_index,
        one_hot_flag=one_hot_flag,
        camera_names=camera_names,
        backbone_model_name=backbone_model_name,
        model_init_weights=model_init_weights,
        freeze_backbone_until=freeze_backbone_until,
        global_pool_image_features_flag=global_pool_image_features_flag,
        use_jaw_values_flag=use_jaw_values_flag,
        use_phase_history_flag=use_phase_history_flag,
        phase_history_len=phase_history_len,
        temporal_mode=temporal_mode,
        phase_to_instruction_mapping=phase_to_instruction_mapping,
        phase_history_only_phase_switches_flag=phase_history_only_phase_switches_flag,
        camera_dropout_prob=camera_dropout_prob,
        jaw_values_dropout_prob=jaw_values_dropout_prob,
        phase_history_dropout_prob=phase_history_dropout_prob,
        image_dim=image_dim,
        llava_anyres_flag=llava_anyres_flag,
        no_llava_anyres_global_image_flag=no_llava_anyres_global_image_flag,
        wrist_images_rel_width=wrist_images_rel_width,
        llava_anyres_rel_width=llava_anyres_rel_width,
        selected_multitasks=selected_multitasks,
        use_seg_masks_input_flag=use_seg_masks_input_flag,
        seg_mask_objs=seg_mask_objs,
        merge_seg_masks_flag=merge_seg_masks_flag,
        seg_masks_dropout_prob=seg_masks_dropout_prob,
        add_center_crop_view_flag=add_center_crop_view_flag,
        merge_global_and_center_embs_flag=merge_global_and_center_embs_flag, 
        distance_from_border_y=distance_from_border_y,
        distance_from_border_x=distance_from_border_x,
        y_offset=y_offset,
        use_phase_history_for_moving_direction_and_corr_pred_flag=use_phase_history_for_moving_direction_and_corr_pred_flag,
        moving_direction_and_corr_history_len=moving_direction_and_corr_history_len,
        use_separate_backbones_flag=use_separate_backbones_flag,
        dataset_mean_std_file_names=dataset_mean_std_file_names,
        num_heads=num_transformer_heads,
        num_layers=num_transformer_layers,
    ).to(device)
    return model

def best_checkpoint(ckpt_dir):
    """
    Returns the best checkpoint file from the given directory (if exists best).
    """

    # Starts with "best_val_loss_" and ends with ".ckpt" - could be multiple from different ckpt runs - take the last one
    best_val_ckpt_name_list = [
        file_name
        for file_name in os.listdir(ckpt_dir)
        if file_name.startswith("best_val_loss_") and file_name.endswith(".ckpt")
    ]
    
    epoch_numbers = [int(file_name.split("=")[1].split(".")[0]) for file_name in best_val_ckpt_name_list]

    # If no valid checkpoints are found, return None
    if not epoch_numbers:
        return None, None

    latest_best_idx = max(epoch_numbers)
    next_idx = latest_best_idx + 1
    return os.path.join(ckpt_dir, f"best_val_loss_epoch={latest_best_idx}.ckpt"), next_idx

def latest_checkpoint(ckpt_dir):
    """
    Returns the latest checkpoint file from the given directory.
    """
    all_ckpts = [
        f
        for f in os.listdir(ckpt_dir)
        if f.startswith("epoch=") and f.endswith(".ckpt")
    ]
    epoch_numbers = [int(file_name.split("=")[1].split(".")[0]) for file_name in all_ckpts]

    # If no valid checkpoints are found, return None
    if not epoch_numbers:
        return None, None

    latest_idx = max(epoch_numbers)
    return os.path.join(ckpt_dir, f"epoch={latest_idx}.ckpt"), latest_idx


if __name__ == "__main__":    
    threading.Thread(target=memory_monitor, daemon=True).start()

    parser = argparse.ArgumentParser(description="Train and evaluate command prediction model using CLIP.")
    parser.add_argument('--dataset_names', nargs='+', type=str, help='List of dataset names', required=True)
    parser.add_argument('--gpu', action='store', type=int, help='gpu', default=0)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--weight_decay', action='store', type=float, help='weight_decay', default=0.01)
    parser.add_argument('--lr_cycle', action='store', type=int, help='lr_cycle', default=50)
    parser.add_argument('--min_lr', action='store', type=float, help='min_lr', default=5e-5)
    parser.add_argument('--backbone_lr_for_transformer_training', action='store', type=float, help='backbone_lr_for_transformer_training', default=1e-5)
    parser.add_argument('--warmup_epochs', action='store', type=int, help='lr_warmup_epochs', default=5)
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--test_only_flag', action='store_true', help='Test the model using the latest checkpoint and exit')
    parser.add_argument('--validation_interval', action='store', type=int, help='validation_interval', default=3)
    parser.add_argument('--save_ckpt_interval', action='store', type=int, help='save_ckpt_interval', default=50)
    parser.add_argument('--early_stopping_interval', action='store', type=int, help='early_stopping_interval', default=None)
    parser.add_argument('--load_best_ckpt_flag', action='store_true', help='Use the best checkpoint based on the validation loss if continue training on available checkpoint')
    parser.add_argument('--plot_val_images_flag', action='store_true', help='Plot images for correct and incorrect predictions')
    parser.add_argument('--max_num_images', action='store', type=int, help='Maximum number of images to plot for correct and incorrect predictions', default=2)
    parser.add_argument('--uniform_sampling_flag', action='store_true', default=True, help='Use uniform sampling for the dataset')
    parser.add_argument('--freeze_backbone_until', action='store', type=str, help='freeze_backbone_until', default="all") 
    parser.add_argument('--prediction_offset', action='store', type=int, help='prediction_offset', default=15)
    parser.add_argument('--recovery_probability', action='store', type=float, help='recovery_probability', default=0.2)
    parser.add_argument('--camera_dropout_prob', action='store', type=float, help='camera_dropout_prob', default=0)
    parser.add_argument('--jaw_values_dropout_prob', action='store', type=float, help='jaw_values_dropout_prob', default=0.2)
    parser.add_argument('--phase_history_dropout_prob', action='store', type=float, help='phase_history_dropout_prob', default=0.2)
    # ---- Architecture parameters ----
    parser.add_argument('--history_len', action='store', type=int, help='history_len', default=3)
    parser.add_argument('--history_step_size', action='store', type=int, help='history_step_size', default=30)
    parser.add_argument('--one_hot_flag', action='store_true', help='Use one hot encoding for the commands')
    parser.add_argument('--cameras_to_use', nargs='+', type=str, help='List of camera names to use', default=["endo_psm2", "left_img_dir", "endo_psm1"]) # "right_img_dir"
    parser.add_argument('--reduced_base_instruction_set_flag', action='store_true', help='Use a reduced set of base classes')
    parser.add_argument('--use_separate_backbones_flag', action='store_true', help='Use separate backbones for each camera')
    parser.add_argument('--backbone_model_name', action='store', type=str, help='backbone_model_name', default="clip")
    parser.add_argument('--global_pool_image_features_flag', action='store_true', help='Should the extracted image features be pooled from (B, N, D) over N to get (B, D) features?')
    parser.add_argument('--image_dim', nargs='+', type=int, help='image_dim', default=[224, 224]) # 336 possible for clip with imagenet weights - (H, W) 
    parser.add_argument('--model_init_weights', action='store', type=str, help='model init weigths like imagenet or even the own pretrained model weights via inputting to the path state dict', default=None)
    # gsvit - possible weights: general | cholecystectomy | imagenet
    # resnet - possible weights: imagenet | mocov2 | simclr | swav | dino 
    # endovit - possible weights: imagenet | endo700k
    # clip - possible weights: imagenet | sda
    # ---
    parser.add_argument('--temporal_mode', action='store', type=str, default=None, help='Select from transformer, LSTM and TCN')
    parser.add_argument('--use_jaw_values_flag', action='store_true', help='Use the jaw values as input to the model')
    parser.add_argument('--use_phase_history_flag', action='store_true', help='Use the history as input to the model')
    parser.add_argument('--phase_history_len', action='store', type=int, help='phases_history_len', default=1)
    parser.add_argument('--phase_history_only_phase_switches_flag', default=True, action='store_true', help='Use only the phase switches in the history')
    parser.add_argument('--prediction_step_size', action='store', type=int, help='prediction_step_size (in number of frames)', default=30)
    # ---
    parser.add_argument('--llava_anyres_flag', action='store_true', help='Use the LLAVA AnyRes images')
    parser.add_argument('--no_llava_anyres_global_image_flag', action='store_true', help='Use the global image instead of the LLAVA AnyRes image')
    parser.add_argument('--wrist_images_rel_width', action='store', type=float, help='wrist_images_rel_width', default=0.75)
    parser.add_argument('--llava_anyres_rel_width', action='store', type=float, help='llava_anyres_rel_width', default=0.5)
    # ---
    default_selected_multitasks = SequenceDataset.get_all_multitask_names()
    parser.add_argument('--selected_multitasks', nargs='*', type=str, help='List of multitasks to use', default=default_selected_multitasks)
    parser.add_argument('--multitask_loss_weight', action='store', type=float, help='multitask_loss_weight', default=0.2)
    parser.add_argument('--use_phase_history_for_moving_direction_and_corr_pred_flag', action='store_true', help='Use the phase history for the moving direction') # TODO: To be removed
    parser.add_argument('--moving_direction_and_corr_history_len', action='store', type=int, help='moving_direction_and_corr_history_len', default=2) # TODO: To be removed
    # ---
    parser.add_argument('--use_seg_masks_input_flag', action='store_true', help='Use the segmentation masks as input to the model')
    parser.add_argument('--seg_mask_objs', nargs='+', type=str, help='List of segmentation mask objects to use. Note: When merging order is expressing the priority, so what will be chosen when two masks overlap.', default=["clips", "left_tube", "right_tube"])
    parser.add_argument('--merge_seg_masks_flag', action='store_true', help='Merge the segmentation masks')
    parser.add_argument('--seg_masks_dropout_prob', action='store', type=float, help='seg_masks_dropout_prob', default=0.5)
    # ---
    parser.add_argument('--use_kinematic_indices_flag', action='store_true', help='Use the kinematic indices as input to the model', default=True)
    # --- Extra sampling parameters ---
    parser.add_argument('--extra_corrections_sampling_flag', action='store_true', help='Use the go back correction prediction')
    parser.add_argument('--extra_corrections_sampling_probability', action='store', type=float, help='extra_corrections_sampling_probability', default=0.2)
    parser.add_argument('--extra_repeated_phase_last_frame_sampling_flag', action='store_true', help='Use the extra going back clipping cutting sampling')
    parser.add_argument('--extra_repeated_phase_last_frame_sampling_probability', action='store', type=float, help='extra_repeated_phase_last_frame_sampling_probability', default=0.2)
    # ---
    parser.add_argument('--add_center_crop_view_flag', action='store_true', help='Add the center crop view to the images')
    parser.add_argument('--merge_global_and_center_embs_flag', action='store_true', help='Merge the global and center embeddings')
    parser.add_argument('--distance_from_border_y', action='store', type=int, help='distance_from_border_y', default=0.1)
    parser.add_argument('--distance_from_border_x', action='store', type=int, help='distance_from_border_x', default=0.25)
    parser.add_argument('--y_offset', action='store', type=int, help='y_offset', default=-0.1)
    # ---
    parser.add_argument('--train_on_all_data_flag', action='store_true', help='Train on all data')
    parser.add_argument('--val_split_number', action='store', type=int, help='Number between 0 to 2 defining which train/val split to use', default=0)
    parser.add_argument('--dataset_mean_std_file_names', action='store', nargs='+', type=str, help='One path per desired camera type', default=None)
    # --- Model parameters ---
    parser.add_argument('--num_transformer_heads', action='store', type=int, help='num_heads', default=4)
    parser.add_argument('--num_transformer_layers', action='store', type=int, help='num_layers', default=2) 

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Set the image dimensions - for better processing (e.g., resizing operations)
    args.image_dim = tuple(args.image_dim)

    # Configure local logging (into ckpt folder)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_output_log_file_path = os.path.join(args.ckpt_dir, "output.log")
    logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(ckpt_output_log_file_path),
                    logging.StreamHandler()
                ])
    logger = logging.getLogger(__name__)

    # Device setting
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nDevice: {device}\n")
    if args.selected_multitasks:
        logger.info(f"Selected multitasks: {args.selected_multitasks}")

    dataset_dirs = []
    num_episodes_list = []

    for dataset in args.dataset_names:
        dataset_config = DATASET_CONFIGS[dataset]
        dataset_dirs.append(dataset_config["dataset_dir"])
        num_episodes_list.append(dataset_config["num_episodes"])
    camera_names = [camera_name for camera_name in dataset_config["camera_names"] if camera_name in args.cameras_to_use]
    camera_file_suffixes = [camera_file_suffix for camera_file_suffix, camera_name in zip(dataset_config["camera_file_suffixes"], dataset_config["camera_names"]) if camera_name in args.cameras_to_use]

    # ---------------------- Define dataloaders and model ----------------------

    # Define transforms/augmentations (resize transformation already applied in __getitem__ method)
    torch_input_transforms = []
    
    # NOTE: Automatic augmentations
    if not args.use_seg_masks_input_flag:
        torch_input_transforms.append(transforms.RandAugment())
        # torch_input_transforms.append(transforms.TrivialAugmentWide())
        # torch_input_transforms.append(transforms.AugMix())
    
    # NOTE: Manual augmentations
    # Torch augmentations
    torch_input_transforms.append(transforms.RandomResizedCrop(args.image_dim, scale=(0.8, 1.0), antialias=True))
    if args.use_seg_masks_input_flag:
        torch_input_transforms.append(transforms.RandomRotation(degrees=[-5.0, 5.0]))
        torch_input_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)))
    # torch_color_input_transforms.append(v2.RandomPerspective(p=0.5))
    
    # Torch Color augmentations
    torch_color_input_transforms = []
    if args.use_seg_masks_input_flag:
        torch_color_input_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.5, hue=0.08))
    # torch_color_input_transforms.append(v2.RandomPosterize(bits=7, p=0.25))
    # torch_color_input_transforms.append(v2.RandomAdjustSharpness(2, p=0.25))
    # torch_color_input_transforms.append(transforms.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.75))
    # torch_color_input_transforms.append(v2.RandomPhotometricDistort(p=0.8))
    # torch_color_input_transforms.append(transforms.RandomGrayscale(p=0.2))
    
    # Albumentations augmentations
    albumentation_input_transforms = []
    min_height, min_width = max(1, args.image_dim[0] // 40), max(1, args.image_dim[1] // 40)
    max_height, max_width = min(args.image_dim[0] // 30, args.image_dim[0]), min(args.image_dim[1] // 30, args.image_dim[1]) 
    albumentation_input_transforms.append(A.CoarseDropout(max_holes=128, max_height=max_height, max_width=max_width, min_holes=1, min_height=min_height, 
                    min_width=min_width, fill_value=0, p=0.5))
    
    # Store the transforms in a dictionary
    torch_input_transforms = transforms.Compose(torch_input_transforms)
    torch_color_input_transforms = transforms.Compose(torch_color_input_transforms)
    num_patches = SequenceDataset.get_num_patches(camera_names, args.add_center_crop_view_flag, args.llava_anyres_flag, args.no_llava_anyres_global_image_flag)
    num_input_images = num_patches * (args.history_len + 1)
    albumentations_additional_targets = dict(zip([f"image{i}" for i in range(num_input_images)], ["image"] * num_input_images))    
    albumentation_input_transforms = A.Compose(albumentation_input_transforms, additional_targets=albumentations_additional_targets)
    input_transforms = {"torch": torch_input_transforms, "albumentations": albumentation_input_transforms, "torch_color": torch_color_input_transforms}

    # Data loading
    output_phase_flag = args.use_phase_history_for_moving_direction_and_corr_pred_flag or args.use_phase_history_flag
    if not args.test_only_flag:
        train_dataloader, val_dataloader, ds_metadata_dict = load_merged_data(
            dataset_dirs=dataset_dirs,
            num_episodes_list=num_episodes_list,
            camera_names=camera_names,
            camera_file_suffixes=camera_file_suffixes,
            batch_size_train=args.batch_size,
            batch_size_val=args.batch_size,
            history_len=args.history_len,
            prediction_offset=args.prediction_offset,
            history_step_size=args.history_step_size,
            test_only=args.test_only_flag,
            input_transforms=input_transforms,
            reduced_base_instruction_set_flag=args.reduced_base_instruction_set_flag,
            use_phase_history_flag=output_phase_flag,
            phase_history_len=args.phase_history_len,
            use_jaw_values_flag=args.use_jaw_values_flag,
            prediction_step_size=args.prediction_step_size,
            recovery_probability=args.recovery_probability,
            phase_history_only_phase_switches_flag=args.phase_history_only_phase_switches_flag,
            image_dim=args.image_dim,
            llava_anyres_flag=args.llava_anyres_flag,
            no_llava_anyres_global_image_flag=args.no_llava_anyres_global_image_flag,
            wrist_images_rel_width=args.wrist_images_rel_width,
            llava_anyres_rel_width=args.llava_anyres_rel_width,
            uniform_sampling_flag=args.uniform_sampling_flag,
            selected_multitasks=args.selected_multitasks,
            use_seg_masks_input_flag=args.use_seg_masks_input_flag,
            seg_mask_objs=args.seg_mask_objs,
            merge_seg_masks_flag=args.merge_seg_masks_flag,
            use_kinematic_indices_flag=args.use_kinematic_indices_flag,
            extra_corrections_sampling_flag=args.extra_corrections_sampling_flag,
            extra_corrections_sampling_probability=args.extra_corrections_sampling_probability,
            extra_repeated_phase_last_frame_sampling_flag=args.extra_repeated_phase_last_frame_sampling_flag,
            extra_repeated_phase_last_frame_sampling_probability=args.extra_repeated_phase_last_frame_sampling_probability,   
            add_center_crop_view_flag=args.add_center_crop_view_flag,
            train_on_all_data_flag=args.train_on_all_data_flag,
            distance_from_border_y=args.distance_from_border_y,
            distance_from_border_x=args.distance_from_border_x,
            y_offset=args.y_offset,
            val_split_number=args.val_split_number
        )
    else:
        # TODO: Move this after loading the ckpt and take the metadata from the checkpoint for init the dataloader - also regarding which gallbladders to eval on
        test_dataloader, ds_metadata_dict = load_merged_data(
            dataset_dirs=dataset_dirs,
            num_episodes_list=num_episodes_list,
            camera_names=camera_names,
            camera_file_suffixes=camera_file_suffixes,
            batch_size_train=args.batch_size,
            batch_size_val=args.batch_size,
            history_len=args.history_len,
            prediction_offset=args.prediction_offset,
            history_step_size=args.history_step_size,
            test_only=args.test_only_flag,
            input_transforms=input_transforms,
            reduced_base_instruction_set_flag=args.reduced_base_instruction_set_flag,
            use_phase_history_flag=output_phase_flag,
            phase_history_len=args.phase_history_len,
            use_jaw_values_flag=args.use_jaw_values_flag,
            prediction_step_size=args.prediction_step_size,
            recovery_probability=args.recovery_probability,
            phase_history_only_phase_switches_flag=args.phase_history_only_phase_switches_flag,
            image_dim=args.image_dim,
            llava_anyres_flag=args.llava_anyres_flag,
            no_llava_anyres_global_image_flag=args.no_llava_anyres_global_image_flag,
            wrist_images_rel_width=args.wrist_images_rel_width,
            llava_anyres_rel_width=args.llava_anyres_rel_width,
            uniform_sampling_flag=args.uniform_sampling_flag,
            selected_multitasks=args.selected_multitasks,
            use_seg_masks_input_flag=args.use_seg_masks_input_flag,
            seg_mask_objs=args.seg_mask_objs,
            merge_seg_masks_flag=args.merge_seg_masks_flag,
            use_kinematic_indices_flag=args.use_kinematic_indices_flag,
            extra_corrections_sampling_flag=args.extra_corrections_sampling_flag,
            extra_corrections_sampling_probability=args.extra_corrections_sampling_probability,
            extra_repeated_phase_last_frame_sampling_flag=args.extra_repeated_phase_last_frame_sampling_flag,
            extra_repeated_phase_last_frame_sampling_probability=args.extra_repeated_phase_last_frame_sampling_probability,   
            add_center_crop_view_flag=args.add_center_crop_view_flag,
            train_on_all_data_flag=args.train_on_all_data_flag,
            distance_from_border_y=args.distance_from_border_y,
            distance_from_border_x=args.distance_from_border_x,
            y_offset=args.y_offset,
            val_split_number=args.val_split_number
        )

    # Merge ds_metadata_dict with args (use as wandb config)
    wandb_metadata = ds_metadata_dict.copy()  # Create a copy to avoid modifying the original dict
    wandb_metadata.update(vars(args))
    
    # Saving the metadata locally in the ckpt_dir
    metadata_path = os.path.join(args.ckpt_dir, "metadata.txt")
    with open(metadata_path, "w") as f:
        for key, value in wandb_metadata.items():
            f.write(f"{key}: {value}\n")

    # WandB initialization
    if args.log_wandb:
        wandb_entity = os.getenv("WANDB_ENTITY")
        run_name = "instructor." + args.ckpt_dir.split("/")[-1] + f".{args.seed}"
        wandb_run_id_path = os.path.join(args.ckpt_dir, "wandb_run_id.txt")
        # check if it exists
        if os.path.exists(wandb_run_id_path): 
            with open(wandb_run_id_path, "r") as f:
                saved_run_id = f.read().strip()
            wandb.init(
                project="yay-surgical-robot", entity=wandb_entity, name=run_name, resume=saved_run_id, config=wandb_metadata
            )
        else:
            wandb.init(
                project="yay-surgical-robot",
                entity=wandb_entity,
                name=run_name,
                config=wandb_metadata,
                resume="allow",
            )
            # Ensure the directory exists before trying to open the file
            os.makedirs(os.path.dirname(wandb_run_id_path), exist_ok=True)
            with open(wandb_run_id_path, "w") as f:
                f.write(wandb.run.id)
        
    # Build the model
    candidate_embeddings = ds_metadata_dict["candidate_embeddings"]
    candidate_texts = ds_metadata_dict["candidate_texts"]  
    if args.reduced_base_instruction_set_flag:
        phase_to_instruction_mapping = ds_metadata_dict["phase_to_instruction_mapping"]
    else:
        phase_to_instruction_mapping = None
        
    logger.info(f"\nLanguage instructions: {candidate_texts}\n")  
    model = build_instructor(args.history_len, args.history_step_size, args.prediction_offset, candidate_embeddings, 
                             candidate_texts, device, args.one_hot_flag, camera_names, args.backbone_model_name,
                             args.model_init_weights, args.freeze_backbone_until, args.global_pool_image_features_flag,
                             args.use_jaw_values_flag, args.use_phase_history_flag, args.phase_history_len, 
                             args.temporal_mode, phase_to_instruction_mapping, args.phase_history_only_phase_switches_flag,
                             camera_dropout_prob=args.camera_dropout_prob, jaw_values_dropout_prob=args.jaw_values_dropout_prob, 
                             phase_history_dropout_prob=args.phase_history_dropout_prob, image_dim=args.image_dim,
                             llava_anyres_flag=args.llava_anyres_flag, no_llava_anyres_global_image_flag=args.no_llava_anyres_global_image_flag,
                             wrist_images_rel_width=args.wrist_images_rel_width, llava_anyres_rel_width=args.llava_anyres_rel_width,
                             selected_multitasks=args.selected_multitasks, use_seg_masks_input_flag=args.use_seg_masks_input_flag,
                             seg_mask_objs=args.seg_mask_objs, merge_seg_masks_flag=args.merge_seg_masks_flag, seg_masks_dropout_prob=args.seg_masks_dropout_prob,
                             add_center_crop_view_flag=args.add_center_crop_view_flag, merge_global_and_center_embs_flag=args.merge_global_and_center_embs_flag,
                             distance_from_border_y=args.distance_from_border_y, distance_from_border_x=args.distance_from_border_x, y_offset=args.y_offset,
                             use_phase_history_for_moving_direction_and_corr_pred_flag=args.use_phase_history_for_moving_direction_and_corr_pred_flag,
                             moving_direction_and_corr_history_len=args.moving_direction_and_corr_history_len, use_separate_backbones_flag=args.use_separate_backbones_flag,
                             dataset_mean_std_file_names=args.dataset_mean_std_file_names, num_transformer_heads=args.num_transformer_heads, num_transformer_layers=args.num_transformer_layers)
    
    if args.temporal_mode == "transformer": 
        backbone_params, other_params = model.get_backbone_and_other_params()
        optimizer = optim.AdamW([{"params": backbone_params, "lr": args.backbone_lr_for_transformer_training}, {"params": other_params}], lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    # Add cosine annealing learning rate scheduler with warm up epochs
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_lambda(epoch, args.warmup_epochs, args.lr_cycle, args.lr, args.min_lr)) 
    criterion = phase_criterion_fct()
    multitask_criterion = multitask_criterion_fct() if args.selected_multitasks else None

    # Load the most recent checkpoint if available
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        next_idx = 0
    else:
        # Load the most recent checkpoint if available
        if args.load_best_ckpt_flag:
            latest_ckpt, next_idx = best_checkpoint(args.ckpt_dir)
        else:
            latest_ckpt, next_idx = latest_checkpoint(args.ckpt_dir)
        if latest_ckpt:
            logger.info(f"\nLoading checkpoint: {latest_ckpt}")
            latest_ckpt_dict = torch.load(latest_ckpt, map_location=device).state_dict()
            model.load_state_dict(latest_ckpt_dict)
        else:
            logger.info("\nNo checkpoint found.")
            next_idx = 0

    # ---------------------- Training loop ----------------------
        
    # Create a directory to save training images for the current run
    training_images_dir = os.path.join(args.ckpt_dir, "training_images")
    if not os.path.exists(training_images_dir):
        os.makedirs(training_images_dir)

    # Test the model using the latest checkpoint - don't train
    if args.test_only_flag:
        latest_idx = next_idx-1
        test(model, test_dataloader, "test", device, latest_idx, args.one_hot_flag, args.ckpt_dir, 
             log_wandb_flag=args.log_wandb, max_num_images=args.max_num_images)
        exit()

    # Training loop
    pbar_epochs = tqdm(range(next_idx, args.num_epochs), desc="Epochs")
    val_loss, next_phase_pred_val_loss, best_val_loss = None, None, float("inf")
    first_iteration_flag = True
    if next_idx != 0:
        scheduler.step(epoch=next_idx)
    for epoch in pbar_epochs:
        if args.log_wandb:
            wandb.log({"Epoch": epoch})
        if epoch == 0:
            logger.info(f"Start warmup for {args.warmup_epochs} epochs")
        logger.info(f"Epoch {epoch}: learning rate = {scheduler.get_last_lr()[0]}, weight decay = {optimizer.param_groups[0]['weight_decay']}")
        
        train_loss, next_phase_pred_train_loss = train(model, train_dataloader, optimizer, criterion, multitask_criterion, device, args.ckpt_dir, epoch, max_num_images=args.max_num_images, multitask_loss_weight=args.multitask_loss_weight)
        if not args.train_on_all_data_flag: 
            val_loss, next_phase_pred_val_loss = evaluate(model, val_dataloader, criterion, multitask_criterion, device, multitask_loss_weight=args.multitask_loss_weight)
            
            # Test the model and log success rate every 200 epochs
            if (epoch + 1) % args.validation_interval == 0:
                test(model, val_dataloader, "val", device, epoch, args.one_hot_flag, args.ckpt_dir, plot_images_flag=args.plot_val_images_flag, 
                     log_wandb_flag=args.log_wandb, max_num_images=args.max_num_images)

        if val_loss is not None:
            pbar_epochs.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss, "Phase Train Loss": next_phase_pred_train_loss, "Phase Val Loss": next_phase_pred_val_loss})
            # Log the losses locally
            logger.info(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f} - Val Loss = {val_loss:.4f}\nPhase Train Loss = {next_phase_pred_train_loss:.4f} - Phase Val Loss = {next_phase_pred_val_loss:.4f}")
        else:
            pbar_epochs.set_postfix({"Train Loss": train_loss})
            logger.info(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}")

        if args.log_wandb:
            wandb.log({"Epoch Train Loss": train_loss})
            wandb.log({"Epoch Phase Train Loss": next_phase_pred_train_loss})
            if val_loss:
                wandb.log({"Epoch Eval Loss": val_loss})
                wandb.log({"Epoch Phase Eval Loss": next_phase_pred_val_loss})

        # -------------------------- Checkpoints --------------------------

        # Save a checkpoint every 100 epochs
        if epoch % args.save_ckpt_interval == 0 and epoch > 0:
            ckpt_name = f"epoch={epoch}.ckpt"
            ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
            torch.save(model, ckpt_path)

            # Pruning: this removes the checkpoint save_ckpt_interval epochs behind the current one
            # except for the ones at multiples of prune_freq epochs
            prune_freq = args.save_ckpt_interval * 3
            prune_epoch = epoch - args.save_ckpt_interval
            if prune_epoch % prune_freq != 0 and not args.train_on_all_data_flag:
                prune_path = os.path.join(args.ckpt_dir, f"epoch={prune_epoch}.ckpt")
                if os.path.exists(prune_path):
                    os.remove(prune_path)
                    
        # Save always the best performing model based on the validation loss
        if not args.train_on_all_data_flag and next_phase_pred_val_loss < best_val_loss:
            best_val_loss = next_phase_pred_val_loss
            if not first_iteration_flag:
                prev_best_val_epoch = best_val_epoch
            best_val_epoch = epoch
            best_ckpt_name = f"best_val_loss_{epoch=}.ckpt"
            best_ckpt_path = os.path.join(args.ckpt_dir, best_ckpt_name)
            torch.save(model, best_ckpt_path)
            # Remove the previous best checkpoint if it exists
            if not first_iteration_flag:
                prev_best_ckpt_path = os.path.join(args.ckpt_dir, f"best_val_loss_epoch={prev_best_val_epoch}.ckpt")
                if os.path.exists(prev_best_ckpt_path):
                    os.remove(prev_best_ckpt_path)
            first_iteration_flag = False
        elif args.train_on_all_data_flag:
            # Save the current 
            ckpt_name = f"epoch={epoch}.ckpt"
            ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
            torch.save(model, ckpt_path)
            # Remove the previous best checkpoint if it exists
            if not first_iteration_flag:
                prev_ckpt_path = os.path.join(args.ckpt_dir, f"epoch={epoch-1}.ckpt")
                if os.path.exists(prev_ckpt_path):
                    os.remove(prev_ckpt_path)
            first_iteration_flag = False
            
        # Early stopping: Stop training if the validation loss has not improved for specific number of epochs
        if args.early_stopping_interval is not None and not args.train_on_all_data_flag:
            if epoch - best_val_epoch >= args.early_stopping_interval:
                logger.info(f"\nEarly stopping at epoch {epoch}")
                break 
            
        # Step the scheduler
        scheduler.step()

    if args.log_wandb:
        wandb.finish()
