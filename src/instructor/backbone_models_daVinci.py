from pathlib import Path
import os
import sys

from clip import load
import torch
import torchvision.transforms as T
from transformers import SwinModel
import torch.nn as nn
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torchvision import models
from functools import partial
from timm.models.vision_transformer import VisionTransformer
from huggingface_hub import snapshot_download
from transformers import ViTForImageClassification, CLIPVisionModel

path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")

# --------------------------- Model init functions ----------------------------
  
def load_swin_fe(model_variant, model_init_weights, device, num_input_channels=3):
    """
    Load a pre-trained Swin Transformer model with customizable input channels and model variants.
    
    :param model_variant: str, specifies the Swin Transformer variant ('tiny', 'small', etc.)
    :param model_init_weights: str, specifies the weights to use ('imagenet', etc.)
    :param num_input_channels: int, the number of input channels, default is 3
    :return: model, num_features
    """
    
    # Define the model variant and Swin Transformer configuration
    if model_variant == 't':
        model_name = 'microsoft/swin-tiny-patch4-window7-224'
    elif model_variant == 's':
        model_name = 'microsoft/swin-small-patch4-window7-224'
    elif model_variant == 'b':
        model_name = 'microsoft/swin-base-patch4-window7-224'
    elif model_variant == 'l':
        model_name = 'microsoft/swin-large-patch4-window12-384'
    else:
        raise ValueError(f"Swin Transformer variant {model_variant} not supported!")
    
    # Load the pre-trained Swin Transformer model with the original configuration
    if model_init_weights == "imagenet":
        model = SwinModel.from_pretrained(model_name)
        
        # Replace the head of the model with a new classification head (if necessary)
        model.pooler = nn.Identity()  # Remove the pre-trained pooler head if present 
    elif os.path.exists(model_init_weights):
        # Init the ResNet model
        model = SwinModel.from_pretrained(model_name)
    
        # Replace the head of the model with a new classification head (if necessary)
        model.pooler = nn.Identity()  # Remove the pre-trained pooler head if present 
    
        # Load the weights from the model
        pretrained_dict = torch.load(model_init_weights, map_location=device).backbone_model.state_dict()        
        model.load_state_dict(pretrained_dict, strict=False)
    else:
        raise ValueError(f"Model weights {model_init_weights} not supported yet!")
    
    # Compute the number of features
    num_features = model.config.hidden_size

    # Adjust the input channels if num_input_channels > 3
    if num_input_channels > 3:
        # Extract the weights of the first convolutional layer
        conv1_weight = model.embeddings.patch_embeddings.projection.weight.data
        # Duplicate the Red channel weights for the additional channels
        num_additional_channels = num_input_channels - 3
        red_channel_weight = conv1_weight[:, 0:1, :, :]  # Extract Red channel (first channel) weights
        red_channel_repeated = red_channel_weight.repeat(1, num_additional_channels, 1, 1)
        new_conv1_weight = torch.cat([conv1_weight, red_channel_repeated], dim=1)
        model.embeddings.patch_embeddings.projection = torch.nn.Conv2d(num_input_channels, model.embeddings.patch_embeddings.projection.out_channels, kernel_size=model.embeddings.patch_embeddings.projection.kernel_size, stride=model.embeddings.patch_embeddings.projection.stride, padding=model.embeddings.patch_embeddings.projection.padding, bias=False)
        model.embeddings.patch_embeddings.projection.weight = torch.nn.Parameter(new_conv1_weight)

    return model, num_features
  
def load_gsvit_fe(model_init_weights, device):
    try:
        from instructor.submodules.gsvit_submodule.gsvit_ae_model import EfficientViTAutoEncoder
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "GSViT source is missing. Expected vendored code under "
            "'src/instructor/submodules/gsvit_submodule/gsvit'. "
            "If you do not intend to use the 'gsvit' backbone, choose a different "
            "backbone_model. If you do intend to use it, add or initialize the "
            "missing GSViT submodule/code."
        ) from exc

    # Load the EfficientViT model as feature extractor
    model = EfficientViTAutoEncoder()

    # Load all weights from the pretrained model (from its encoder)
    models_folder_path = Path(__file__).resolve().parent / "submodules" / "gsvit_submodule" / "models"
    if model_init_weights == "general":
        model_path = models_folder_path / "GSViT.pkl"
    elif model_init_weights == "chole":
        model_path = models_folder_path / "saved_network_cholesystectomy_0_41.pkl" # Use either "GSViT.pkl" or "saved_network_cholesystectomy_0_41.pkl"
    elif model_init_weights == "imagenet":
        pass # As imagenet weights are loaded by default
    else:
        raise ValueError(f"Model weights {model_init_weights} not supported yet!")
    
    if model_init_weights in ["general", "chole"]:
        # Load the weights from the model
        # model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(pretrained_dict)
    
    # # Check if model_dict and pretrained_dict keys are the same
    # common_keys = set(model_dict.keys()) & set(pretrained_dict.keys())
    # print("\nKeys present in both the model's encoder architecture and pretrained SSL weights:", common_keys)
    
    num_features = 384 
    
    return model, num_features
  
def load_endovit_fe(model_init_weights, img_size, device, patch_size = 16, embed_dim = 768, depth = 12, num_heads = 12,
                    mlp_ratio = 4, qkv_bias = True, norm_layer = partial(nn.LayerNorm, eps=1e-6)):
       
    if model_init_weights == "endo700k":
        # Define the huggingface repository and model filename
        repo_id = "egeozsoy/EndoViT"
        model_filename = "pytorch_model.bin"
        
        # Download model files
        model_path = snapshot_download(repo_id=repo_id, revision="main")
        model_init_weights_path = Path(model_path) / model_filename

        # Load model weights
        model_init_weights = torch.load(model_init_weights_path, map_location=device)['model']

        # Define the model (ensure this matches your model's architecture)
        model = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)

        # Load the weights into the model
        model.load_state_dict(model_init_weights, strict=False)             
        
        # Assuming the model is a vision transformer, we need to get the dimension of the last layer
        num_features = model.head.in_features # If not using GlobalAveragePooling * (img_size[0]//patch_size * img_size[1]//patch_size + 1)
        
        # Replace the head of the model with new classification heads for the specific tasks
        model.head = nn.Identity()  # Remove the pre-trained classification head if present
    elif model_init_weights == "imagenet":
        # Using huggingface imagenet pretrained weights (https://huggingface.co/google/vit-base-patch16-224)
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        
        # Set the number of features
        num_features = model.classifier.in_features
        
        # Replace the head of the model with new classification heads for the specific tasks
        model.classifier = nn.Identity()  # Remove the pre-trained classification head if present
    else:
        raise ValueError(f"Model weights {model_init_weights} not supported yet!")
    
    return model, num_features
    
def load_resnet_fe(model_init_weights, device, num_input_channels=3):   
    if model_init_weights is None:
        # Load the bare model
        model = models.resnet50()
        num_features = model.fc.in_features
        model.fc = torch.nn.Identity() # Remove the original classification head
    elif model_init_weights == "imagenet":
        model = models.resnet50(weights='IMAGENET1K_V2')
        num_features = model.fc.in_features
        model.fc = torch.nn.Identity() # Remove the original classification head
    # Load pretrained  SelfSupSurg weights
    elif model_init_weights in ["mocov2", "simclr", "swav", "dino"]:
        self_sup_surg_models_folder = Path(__file__).resolve().parent / "submodules" / "selfsupsurg" / "models"
        if model_init_weights == "mocov2":
            # Init the ResNet model
            model = models.resnet50()
            num_features = model.fc.in_features
            model.fc = torch.nn.Identity() # Remove the original classification head
            
            # Load the weights from the MoCoV2 model
            mocov2_model_path = self_sup_surg_models_folder / "model_final_checkpoint_moco_v2_surg.torch"
            base_model_checkpoint = torch.load(mocov2_model_path, map_location=device)["classy_state_dict"]["base_model"]["model"]["trunk"]
            pretrained_dict = {param_name.replace('_feature_blocks.', ''): param_weight for param_name, param_weight in base_model_checkpoint.items()}
            model.load_state_dict(pretrained_dict, strict=False)
        elif model_init_weights == "simclr":
            # Init the ResNet model
            model = models.resnet50()
            num_features = model.fc.in_features
            model.fc = torch.nn.Identity() # Remove the original classification head
            
            # Load the weights from the SimCLR model
            simclr_model_path = self_sup_surg_models_folder / "model_final_checkpoint_simclr_surg.torch"
            base_model_checkpoint = torch.load(simclr_model_path, map_location=device)["classy_state_dict"]["base_model"]["model"]["trunk"]
            pretrained_dict = {param_name.replace('_feature_blocks.', ''): param_weight for param_name, param_weight in base_model_checkpoint.items()}
            model.load_state_dict(pretrained_dict, strict=False)            
        elif model_init_weights == "swav":
            # Init the ResNet model
            model = models.resnet50()
            num_features = model.fc.in_features
            model.fc = torch.nn.Identity() # Remove the original classification head
            
            # Load the weights from the SwAV model
            swav_model_path = self_sup_surg_models_folder / "model_final_checkpoint_swav_surg.torch"
            base_model_checkpoint = torch.load(swav_model_path, map_location=device)["classy_state_dict"]["base_model"]["model"]["trunk"]
            pretrained_dict = {param_name.replace('_feature_blocks.', ''): param_weight for param_name, param_weight in base_model_checkpoint.items()}
            model.load_state_dict(pretrained_dict, strict=False)
        elif model_init_weights == "dino":
            # Init the ResNet model
            model = models.resnet50()
            num_features = model.fc.in_features
            model.fc = torch.nn.Identity() # Remove the original classification head
            
            # Load the weights from the DINO model
            dino_model_path = self_sup_surg_models_folder / "model_final_checkpoint_dino_surg.torch"
            base_model_checkpoint = torch.load(dino_model_path, map_location=device)["classy_state_dict"]["base_model"]["model"]["trunk"]
            pretrained_dict = {param_name.replace('_feature_blocks.', ''): param_weight for param_name, param_weight in base_model_checkpoint.items()}
            model.load_state_dict(pretrained_dict, strict=False)      
    elif os.path.exists(model_init_weights):
        # Init the ResNet model
        model = models.resnet50()
        num_features = model.fc.in_features
        model.fc = torch.nn.Identity() # Remove the original classification head
    
        # Load the weights from the model
        pretrained_dict = torch.load(model_init_weights, map_location=device).backbone_model.state_dict()        
        model.load_state_dict(pretrained_dict, strict=False)
    else:
        raise ValueError(f"Model weights {model_init_weights} not supported yet!")
    
    # Integrate the num_input_channels by duplicating the Red channel weights
    if num_input_channels > 3:
        conv1_weight = model.conv1.weight.data
        # Duplicate the Red channel weights for the additional channels
        num_additional_channels = num_input_channels - 3
        red_channel_weight = conv1_weight[:, 0:1, :, :]  # Extract Red channel (first channel) weights
        red_channel_repeated = red_channel_weight.repeat(1, num_additional_channels, 1, 1)
        new_conv1_weight = torch.cat([conv1_weight, red_channel_repeated], dim=1)
        model.conv1 = torch.nn.Conv2d(num_input_channels, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
        model.conv1.weight = torch.nn.Parameter(new_conv1_weight)
    
    return model, num_features

def load_clip_fe(model_init_weights, device, img_size):    
    if model_init_weights == "imagenet" and img_size == 224:
        # Load the CLIP model
        model = load("ViT-B/32", device=device)[0]
    
        # Set the number of features
        num_features = model.visual.output_dim
    elif model_init_weights == "sda" and img_size == 224:
        # Load the SDA-CLIP model
        model_weights_path = Path(__file__).resolve().parent / "submodules" / "clip" / "models" / "soft_task.pt"
        model = load("ViT-B/16", device=device)[0]
        sda_clip_state_dict = torch.load(model_weights_path, map_location=device, weights_only=True)["model_state_dict"]
        model.load_state_dict(sda_clip_state_dict)
        
        # Set the number of features
        num_features = model.visual.output_dim
    elif model_init_weights == "imagenet" and img_size == 336:
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
        
        # Set the number of features
        num_features = model.config.hidden_size
    else:
        raise ValueError(f"Model weights {model_init_weights} not supported yet!")
    
    return model, num_features
    
def init_feature_extractor_model(fe_model_name, model_init_weights, device, freeze_fe_until_layer, img_size=224, num_input_channels=3):
    
    if num_input_channels != 3 and fe_model_name in ["gsvit", "endovit", "clip"]:
        raise ValueError(f"Feature extractor model {fe_model_name} using segmentation mask as additional input is not supported yet!")
    
    # Load the desired feature extractor model
    img_size = img_size[0]  if img_size[0] == img_size[1] else img_size
    if fe_model_name == "gsvit" and img_size == 224:
        # Load a pre-trained GSViT model (either with SSL or ImageNet weights)
        fe, num_features = load_gsvit_fe(model_init_weights, device)
    elif fe_model_name == "endovit":
        # Load a pre-trained EndoViT model (either with SSL or ImageNet weights)
        fe, num_features = load_endovit_fe(model_init_weights, img_size=[img_size, img_size], device=device)
    elif fe_model_name == "resnet":
        # Load a pre-trained ResNet50 model (either with SelfSupSurg or ImageNet weights)
        fe, num_features = load_resnet_fe(model_init_weights, device, num_input_channels=num_input_channels)
    elif fe_model_name == "clip":
        # Load a pre-trained CLIP model
        fe, num_features = load_clip_fe(model_init_weights, device, img_size)
        if freeze_fe_until_layer != "all":
            freeze_fe_until_layer = "all"  # Freeze the whole CLIP model
            raise ValueError(f"Unfreezing the CLIP model is not supported yet! Leads to instability in training.")
    elif "swin" in fe_model_name:
        model_variant = fe_model_name.split("-")[1]
        # Load a pre-trained Swin Transformer model
        fe, num_features = load_swin_fe(model_variant, model_init_weights, device, num_input_channels=num_input_channels)
    else:
        raise ValueError(f"Feature extractor model {fe_model_name} with input size {img_size} and model init weights {model_init_weights} not supported yet!")
        
    # Freeze the feature extractor
    for layer_idx, param in enumerate(fe.parameters()):
        if freeze_fe_until_layer != "all" and (freeze_fe_until_layer == "none" or layer_idx == freeze_fe_until_layer):
            break
        param.requires_grad = False
        
    return fe, num_features

# ------------------------- Model preprocessing functions -------------------------
    
def preprocess_inputs_gsvit(images):    
    """
    Flip color channels, e.g., from RGB to BGR
    
    Args:
        images (torch.Tensor): Input images
    
    Returns:
        images (torch.Tensor): Images with flipped color channels
    """

    # Switch RGB to BGR as pretrained on BGR images
    tmp = images[:, 0, :, :].clone()
    images[:, 0, :, :] = images[:, 2, :, :]
    images[:, 2, :, :] = tmp
    return images
   
    
def preprocess_inputs(images, fe_model_name, dataset_mean, dataset_std):

    # Only take the RGB channels for preprocessing (e.g., when having additional seg mask dimension) - batch size x num channels x H x W
    num_channels = images.shape[1]
    if num_channels > 3:
        images_rgb_channels = images[:, :3, :, :]
        remaining_channels = images[:, 3:, :, :]
    else:
        images_rgb_channels = images

    # Normalize the images - zero mean and unit variance
    if dataset_mean is not None and dataset_std is not None:
        images_rgb_channels = T.Normalize(mean=dataset_mean, std=dataset_std)(images_rgb_channels)

    if fe_model_name == "gsvit":
        # Process the inputs
        images_rgb_channels = preprocess_inputs_gsvit(images_rgb_channels)
        
    # Concatenate the RGB channels with the remaining channels (e.g., segmentation mask)
    if num_channels > 3:
        images = torch.cat((images_rgb_channels, remaining_channels), dim=1)
    else:
        images = images_rgb_channels
        
    return images

# ------------------------- FE + classifier functions ----------------------------------

def extract_features(fe, fe_model_name, model_init_weights, img_size, x, pool_features: bool = False):
    """
    
    Outputs:
        features (torch.Tensor): encoded image features of shape (B, N, F) or (B, F) if pool_features.
    """
    
    img_size = img_size[0]  if img_size[0] == img_size[1] else img_size
    if fe_model_name == "gsvit" and img_size == 224:
        features = fe.evit(x)
        # Apply 2D Global Average Pooling
        batch_size, num_features = features.shape[:2]
        flattened_tensor = x.view(batch_size, num_features, -1)  # shape: (Batchsize, num_features, 4x4)
        # Compute average pooling across the last dimension (spatial dimensions)
        if pool_features:
            features = torch.mean(flattened_tensor, dim=-1)  # shape: (Batchsize, num_features)
    elif fe_model_name == "endovit" and model_init_weights == "endo700k" and img_size == 224:
        features = fe.forward_features(x)
        # Apply 1D Global Average Pooling
        if pool_features:
            features = torch.mean(features, dim=1)
    elif fe_model_name == "endovit" and model_init_weights == "imagenet" and img_size == 224:
        features = fe(x).logits
        # TODO: unclear what to do with pooling flag
    elif fe_model_name == "clip" and model_init_weights in ["imagenet", "sda"] and img_size == 224:
        features = fe.encode_image(x)
        # TODO: unclear what to do with pooling flag
    elif fe_model_name == "clip" and model_init_weights == "imagenet" and img_size == 336:
        features = fe(x).pooler_output
        # TODO: unclear what to do with pooling flag
    elif "swin" in fe_model_name:
        features = fe(x).last_hidden_state
        if pool_features:
        # Apply global average pooling if needed
            features = torch.mean(features, dim=1)
    else:
        features = fe(x)
        
    return features

# TODO: Maybe also add later -> output_logits = self.classifier(features.reshape(features.size(0), -1)) in instructor model + advanced_classifier_flag = False in instructor training
def init_classifier(num_features, num_outputs, advanced_classifier_flag, complexity_level=4):
    # Complexity level: 0, 1, 2, 3, 4 (-> 4 is the original complexity level)
    
    if advanced_classifier_flag:
        # Define a more complex classifier
        complexity_factor = 2**complexity_level # -> 1, 2, 4, 8, 16
        classifier = nn.Sequential(
            nn.Linear(num_features, 128*complexity_factor), 
            nn.ELU(),  # Apply ELU activation
            nn.LayerNorm(128*complexity_factor),  # Apply Layer Normalization
            nn.Dropout(p=0.1),  # Add dropout with 10% dropout chance
            nn.Linear(128*complexity_factor, 32*complexity_factor),
            nn.ELU(),  # Apply ELU activation
            nn.LayerNorm(32*complexity_factor),  # Apply Layer Normalization
            nn.Dropout(p=0.1),  # Add dropout with 10% dropout chance
            nn.Linear(32*complexity_factor, num_outputs),
            # No activation here, outputting logits
        )  
    else:
        # Define the classification head
        classifier = nn.Linear(num_features, num_outputs)

    return classifier
