import random
import numpy as np
import torch
import math
import cv2

from torch.utils.data import Sampler

def randintgaussian(low, high, mean, std_dev):
    """
    Generate a random integer using a Gaussian distribution and clip it to the specified range.

    Parameters:
    low (int): The minimum value (inclusive).
    high (int): The maximum value (exclusive).
    mean (float): The mean of the Gaussian distribution.
    std_dev (float): The standard deviation of the Gaussian distribution.

    Returns:
    int: A random integer within the specified range.
    """
    # Generate a random number from a Gaussian distribution
    value = int(np.random.normal(mean, std_dev))

    # Clip the value to ensure it falls within the desired range
    value = np.clip(value, low, high - 1)

    return value

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def lr_lambda(current_epoch, warmup_epochs, lr_cycle, base_lr, min_lr):
    """
    Learning rate scheduler with warm-up and cosine annealing.

    Parameters:
    - current_epoch: Current training epoch.
    - warmup_epochs: Number of warm-up epochs.
    - lr_cycle: Number of epochs to complete a cosine annealing cycle (T_max).
    - base_lr: Initial learning rate before warm-up.
    - min_lr: Minimum learning rate during cosine annealing.

    Returns:
    - Learning rate multiplier.
    """
    if current_epoch < warmup_epochs:
        # Linear warm-up
        return (current_epoch+1) / warmup_epochs
    else:
        # Cosine annealing
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_epoch + 1 - warmup_epochs) / (lr_cycle - warmup_epochs)))
        return (1 - min_lr / base_lr) * cosine_decay + min_lr / base_lr

def phase_criterion_fct():
    """
    Create a loss function for the phase prediction task.

    Returns:
    - A loss function for the phase prediction task.
    """
    criterion = torch.nn.CrossEntropyLoss(reduction="none") 
    def loss_fct(logits, labels):
        loss = criterion(logits, labels)
        pred_phase_idx = torch.argmax(logits, dim=1)
        phase_difference = abs(pred_phase_idx - labels)
        scaled_loss = loss * (1 + phase_difference)
        return scaled_loss.mean() 
    return loss_fct

def multitask_criterion_fct():
    """
    Create a multi-task loss function that combines multiple loss functions.

    Returns:
    - A combined multi-task loss function.
    """
    
    criterion = torch.nn.CrossEntropyLoss()  # You could add class weights here if needed
    next_frame_criterion = torch.nn.MSELoss()  # For future frame prediction
    def loss_fct(multitask_logits_dict, multitask_label_indices_dict):
        loss = 0.0
        selected_multitasks = multitask_logits_dict.keys()
        
        for multitask in selected_multitasks:
            logits = multitask_logits_dict[multitask]
            labels = multitask_label_indices_dict[multitask]
            
            # Check for correct shape (logits should be [batch_size, num_classes], labels should be [batch_size])
            assert logits.shape[0] == labels.shape[0], f"Shape mismatch for task {multitask}"
            if multitask == "future_frame_prediction":
                loss += next_frame_criterion(logits, labels)
            loss += criterion(logits, labels)
        
        # scale the loss by the number of tasks
        scaled_loss = loss / len(selected_multitasks)
        
        return scaled_loss

    return loss_fct

def rotate_image(image, angle):
    """Rotate the image by the given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def shift_image(image, shift_x, shift_y):
    """Shift the image by the given x and y offsets."""
    (h, w) = image.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, M, (w, h))
    return shifted_image
