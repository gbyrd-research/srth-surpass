import sys
import os
import json

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import random

# import aloha
path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")
from instructor.backbone_models_daVinci import extract_features, init_feature_extractor_model, preprocess_inputs
from instructor.dataset_daVinci import SequenceDataset
from instructor.future_frame_predictor_model import FrameDecoder
from instructor.temporal_models import SingleStageTCNModel

class Instructor(nn.Module):
    def __init__(
        self,
        device,
        history_len,
        history_step_size,
        prediction_offset,
        camera_names,
        output_dim=768, # DestillBert embedding space size
        hidden_dim=256, # For the MLP
        num_heads=4, # For the Transformer - 8
        num_layers=2, # For the Transformer - 6
        lstm_hidden_dim=256, # For the LSTM
        tcn_hidden_dim=256, # For the TCN
        tcn_num_layers=4, # For the TCN
        candidate_embeddings=None,
        candidate_texts=None,
        command_to_index=None,
        one_hot_flag=False,
        backbone_model_name="clip",
        model_init_weights=None,
        freeze_backbone_until="all",
        global_pool_image_features_flag=False,
        use_jaw_values_flag=False,
        jaw_values_output_dim=256,
        use_phase_history_flag=False,
        phase_history_len=6,
        phase_emb_dim=4,
        phase_history_output_dim=16,
        temporal_mode=None, # Options: "transformer", "tcn", "lstm", or None
        phase_to_instruction_mapping=None,
        phase_history_only_phase_switches_flag=False,
        camera_dropout_prob=0.1,
        jaw_values_dropout_prob=0.1,
        phase_history_dropout_prob=0.1,
        image_dim=224,
        llava_anyres_flag=False,
        no_llava_anyres_global_image_flag=False,
        wrist_images_rel_width=3/4,
        llava_anyres_rel_width=1/2,
        selected_multitasks=[],
        use_seg_masks_input_flag=False,
        merge_seg_masks_flag=False,
        seg_mask_objs=["clips", "left_tube", "right_tube"], # Possible objects: "clips", "left_tube", "right_tube", "flap" 
        seg_masks_dropout_prob=0.2,
        future_frame_prediction_flag=False,
        add_center_crop_view_flag=False,
        merge_global_and_center_embs_flag=False,
        distance_from_border_y=0.1,
        distance_from_border_x=0.25,
        y_offset=-0.1,
        use_phase_history_for_moving_direction_and_corr_pred_flag=False, # TODO: To be removed (if it works with longer history)
        moving_direction_and_corr_history_len=2, # TODO: to be removed (if it works with longer history)
        use_separate_backbones_flag=False,
        dataset_mean_std_file_names = None,
        dataset_mean_std_camera_dict = None,
    ):
        super().__init__()

        # Store the parameters (for logging)
        self.one_hot_flag = one_hot_flag
        self.history_len = history_len
        self.history_step_size = history_step_size
        self.prediction_offset = prediction_offset
        self.camera_names = camera_names
        self.candidate_embeddings = candidate_embeddings 
        self.candidate_texts = candidate_texts
        self.command_to_index = command_to_index 
        self.backbone_model_name = backbone_model_name
        self.model_init_weights = model_init_weights
        self.freeze_backbone_until = freeze_backbone_until
        self.global_pool_image_features_flag = global_pool_image_features_flag
        self.device = device
        self.use_jaw_values_flag = use_jaw_values_flag
        self.use_phase_history_flag = use_phase_history_flag
        if use_jaw_values_flag:
            self.jaw_values_output_dim = jaw_values_output_dim
        if use_phase_history_flag:
            self.phase_emb_dim = phase_emb_dim
            self.phase_history_output_dim = phase_history_output_dim
            self.phase_history_len = phase_history_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        if temporal_mode == "transformer":
            self.num_heads = num_heads
            self.num_layers = num_layers
        elif temporal_mode == "tcn":
            self.tcn_hidden_dim = tcn_hidden_dim
            self.tcn_num_layers = tcn_num_layers
        elif temporal_mode == "lstm":
            self.lstm_hidden_dim = lstm_hidden_dim
        self.phase_to_instruction_mapping = phase_to_instruction_mapping
        self.phase_history_only_phase_switches_flag = phase_history_only_phase_switches_flag
        self.image_dim = image_dim
        self.llava_anyres_flag = llava_anyres_flag
        self.no_llava_anyres_global_image_flag = no_llava_anyres_global_image_flag
        self.wrist_images_rel_width = wrist_images_rel_width
        self.llava_anyres_rel_width = llava_anyres_rel_width
        self.selected_multitasks = selected_multitasks
        self.num_input_channels = 3
        self.use_seg_masks_input_flag = use_seg_masks_input_flag
        self.add_center_crop_view_flag = add_center_crop_view_flag
        self.merge_global_and_center_embs_flag = merge_global_and_center_embs_flag
        self.distance_from_border_y = distance_from_border_y
        self.distance_from_border_x = distance_from_border_x
        self.y_offset = y_offset
        self.camera_patch_names = SequenceDataset.get_camera_patch_names(camera_names, add_center_crop_view_flag, llava_anyres_flag, no_llava_anyres_global_image_flag)
        self.temporal_mode = temporal_mode
        self.num_cameras = len(camera_names)
        self.use_phase_history_for_moving_direction_and_corr_pred_flag = use_phase_history_for_moving_direction_and_corr_pred_flag # TODO: To be removed (if it works with longer history)
        self.moving_direction_and_corr_history_len = moving_direction_and_corr_history_len # TODO: To be removed (if it works with longer history)
        self.use_separate_backbones_flag = use_separate_backbones_flag

        # Dropout probabilities
        if len(camera_names) <= 1 and camera_dropout_prob > 0:
            self.camera_dropout_prob = 0
            print(f"Set camera dropout probability to {self.camera_dropout_prob} as there is only one camera.")
        else:
            self.camera_dropout_prob = camera_dropout_prob
        if use_jaw_values_flag:
            self.jaw_values_dropout_prob = jaw_values_dropout_prob
        if use_phase_history_flag or use_phase_history_for_moving_direction_and_corr_pred_flag:
            self.phase_history_dropout_prob = phase_history_dropout_prob
        
        if use_seg_masks_input_flag and self.camera_patch_names != ["left_img_dir"]:
            print("Segmentation masks can currently only be used with the left_img_dir camera. Switching segmentation masks off.")
            self.use_seg_masks_input_flag = False
        
        if self.use_seg_masks_input_flag:
            self.merge_seg_masks_flag = merge_seg_masks_flag
            self.seg_mask_objs = seg_mask_objs
            self.seg_masks_dropout_prob = seg_masks_dropout_prob
            additional_input_channels = len(seg_mask_objs) if not merge_seg_masks_flag else 1
            self.num_input_channels += additional_input_channels

        # TODO: To be removed
        # if self.history_len < self.moving_direction_and_corr_history_len:
        #     print("The history length should be less than or equal to the moving direction and correction history length. Setting the history length to the moving direction and correction history length.")
        #     self.moving_direction_and_corr_history_len = self.history_len

        # Load the dataset mean and std
        self.dataset_mean_std_file_names = dataset_mean_std_file_names
        if dataset_mean_std_camera_dict: # E.g., for inference
            self.dataset_mean_std_camera_dict = dataset_mean_std_camera_dict
        elif dataset_mean_std_file_names: # If not given load via file (if available)
            instructor_folder_dir = os.path.dirname(os.path.abspath(__file__))
            self.dataset_mean_std_camera_dict = {}
            for camera_name, dataset_mean_std_file_name in zip(camera_names, dataset_mean_std_file_names):
                dataset_mean_std_json_path = os.path.join(instructor_folder_dir, dataset_mean_std_file_name)
                with open(dataset_mean_std_json_path, "r") as f:
                    dataset_mean_std = json.load(f)
                self.dataset_mean_std_camera_dict[camera_name] = (dataset_mean_std["total_dataset"]["mean"], dataset_mean_std["total_dataset"]["std"])
        else:
            self.dataset_mean_std_camera_dict = None

        # -------------
        
        # Load (pretrained) backbone model(s)
        if self.use_separate_backbones_flag:
            # Initialize separate backbone models for each camera
            self.backbone_models = nn.ModuleDict()
            self.backbone_output_dims = {}
            for camera_name in self.camera_names:
                backbone_model, backbone_output_dim = init_feature_extractor_model(backbone_model_name, model_init_weights, device, freeze_backbone_until, image_dim, self.num_input_channels)
                self.backbone_models[camera_name] = backbone_model
            self.backbone_output_dim = backbone_output_dim
        else:
            self.backbone_model, self.backbone_output_dim = init_feature_extractor_model(backbone_model_name, model_init_weights, device, freeze_backbone_until, image_dim, self.num_input_channels)
        
        num_camera_patches = SequenceDataset.get_num_patches(camera_names, add_center_crop_view_flag, llava_anyres_flag, no_llava_anyres_global_image_flag)
        if self.merge_global_and_center_embs_flag and self.add_center_crop_view_flag:
            num_camera_patches -= 1 # As merging global and center crop views

        # If the images features are not pooled, each image will be encoded as (patches, embedding) instead of (embedding) -> extra dimension
        # We feed this extra dimension in the time steps dimension to the transformer. -> positional embedding needs to be extended.
        if not self.global_pool_image_features_flag:
            dummy_input = torch.zeros((1, 3, *self.image_dim), dtype=torch.float32)
            if self.use_separate_backbones_flag:
                backbone_model = self.backbone_models[list(self.backbone_models.keys())[0]]
            else:
                backbone_model = self.backbone_model
            dummy_output = extract_features(backbone_model, self.backbone_model_name, self.model_init_weights, self.image_dim, dummy_input, self.global_pool_image_features_flag)
            assert dummy_output.ndim == 3
            num_embedded_patches = dummy_output.shape[1]
            patch_embedding_size = dummy_output.shape[2]
            assert patch_embedding_size == self.backbone_output_dim
            embedding_patches_per_image = num_embedded_patches
        else:
            embedding_patches_per_image = 1
            patch_embedding_size = self.backbone_output_dim
            
        image_embedding_size = embedding_patches_per_image * patch_embedding_size

        if self.temporal_mode == "transformer":
            transformer_input_dim = patch_embedding_size
            # Transformer for processing sequences of image embeddings
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=transformer_input_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True
                ),
                num_layers=num_layers,
            )

            # Positional Encoding
            # NOTE: time dimension of positional embedding is 
            # number of encoded image patches (based on feature extractor) * number of cameras * (number of past images + 1 (current image))
            self.positional_encoding = self.create_sinusoidal_embeddings(transformer_input_dim, (history_len + 1) * num_camera_patches * embedding_patches_per_image)
            image_output_dim = transformer_input_dim
        elif self.temporal_mode == "tcn":
            # TODO: probably invalid code now.
            tcn_input_dim = self.backbone_output_dim if not(add_center_crop_view_flag and not merge_global_and_center_embs_flag) else 2*self.backbone_output_dim
            self.tcn = SingleStageTCNModel(
                num_layers=tcn_num_layers,
                num_f_maps=tcn_hidden_dim,
                input_dim=tcn_input_dim,
                causal_conv=True)
            image_output_dim = self.tcn_hidden_dim
        elif self.temporal_mode == "lstm":
            # TODO: probably invalid code now
            lstm_input_dim = self.backbone_output_dim if not(add_center_crop_view_flag and not merge_global_and_center_embs_flag) else 2*self.backbone_output_dim
            self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=self.lstm_hidden_dim, batch_first=True)
            image_output_dim = self.lstm_hidden_dim
        else: # When no temporal mode is used
            if merge_global_and_center_embs_flag and add_center_crop_view_flag: 
                image_output_dim = image_embedding_size * self.num_cameras * (history_len + 1) # As merging global and center crop views
            else:
                image_output_dim = image_embedding_size * num_camera_patches * (history_len + 1) # As image features are concatenated then
        
        # Get the image feature dimension for the moving direction and correction prediction
        #if "is_correction" in selected_multitasks or "dominant_moving_direction" in selected_multitasks:
        #    num_dir_and_corr_images = self.moving_direction_and_corr_history_len + 1
        #    if merge_global_and_center_embs_flag and add_center_crop_view_flag: 
        #        moving_direction_and_corr_image_dim = image_embedding_size * self.num_cameras * num_dir_and_corr_images # As merging global and center crop views
        #    else:
        #        moving_direction_and_corr_image_dim = image_embedding_size * num_camera_patches * num_dir_and_corr_images # As image features are concatenated then
        
        if use_jaw_values_flag:
            # MLP for processing jaw values
            num_jaw_values = 2 * (history_len + 1) # 2 values per timestep
            self.jaw_values_mlp = nn.Linear(num_jaw_values, jaw_values_output_dim)
            image_jaw_mlp_input_dim = image_output_dim + jaw_values_output_dim
            self.image_jaw_mlp = nn.Sequential(
                nn.Linear(image_jaw_mlp_input_dim, hidden_dim),
                nn.ReLU()
            )
            ## ---- For direction and correction head ---- # 
            #if "is_correction" in selected_multitasks or "dominant_moving_direction" in selected_multitasks:
            #    num_jaw_values_dir_corr = 2 * num_dir_and_corr_images # 2 values per timestep
            #    self.jaw_values_mlp_dir_corr = nn.Linear(num_jaw_values_dir_corr, jaw_values_output_dim)
            #    image_jaw_mlp_input_dim_dir_corr = moving_direction_and_corr_image_dim + jaw_values_output_dim
            #    self.image_jaw_mlp_dir_corr = nn.Sequential(
            #        nn.Linear(image_jaw_mlp_input_dim_dir_corr, hidden_dim),
            #        nn.ReLU()
            #    )
            
            
        # Apply multitask training on only images (+jaw values) to better improve image/jaw embeddings
        if selected_multitasks:
            multitasks_input_dim = hidden_dim if use_jaw_values_flag else image_output_dim
            # Use nn.ModuleDict to store MLP heads
            selected_multitasks_output_dim_dict = SequenceDataset.get_multitask_labels_output_dim_dict(selected_multitasks)
            self.multitask_mlp_heads = nn.ModuleDict()
            for multitask, num_outputs in selected_multitasks_output_dim_dict.items():
                #if multitask in ["dominant_moving_direction", "is_correction"]:
                #    if not self.use_phase_history_for_moving_direction_and_corr_pred_flag: # If not using phase information, else add MLP head further down after phase info
                #        multitask_moving_direction_and_corr_input_dim = hidden_dim if use_jaw_values_flag else moving_direction_and_corr_image_dim
                #        self.multitask_mlp_heads[multitask] = nn.Linear(multitask_moving_direction_and_corr_input_dim, num_outputs)
                #else:
                self.multitask_mlp_heads[multitask] = nn.Linear(multitasks_input_dim, num_outputs)
            
        if use_phase_history_flag or use_phase_history_for_moving_direction_and_corr_pred_flag: 
            # Embedding for phase history
            self.phase_embedding = nn.Embedding(len(candidate_texts)+1, phase_emb_dim)
            history_mlp_input_dim = phase_emb_dim * phase_history_len
            self.history_mlp = nn.Sequential(
                nn.Linear(history_mlp_input_dim, phase_history_output_dim),
                nn.ReLU()
            )
            
            # Add mapping from commands to indices (with padding)
            self.history_phase_to_index = command_to_index
            self.history_phase_to_index = {k: v + 1 for k, v in self.history_phase_to_index.items()}
            self.history_phase_to_index["padding"] = 0            

        # MLP for processing the final output
        if use_jaw_values_flag and use_phase_history_flag:
            mlp_input_dim = hidden_dim + phase_history_output_dim
        elif use_jaw_values_flag:
            mlp_input_dim = hidden_dim
        elif use_phase_history_flag:
            mlp_input_dim = image_output_dim + phase_history_output_dim
        else:
            mlp_input_dim = image_output_dim

        if one_hot_flag:
            output_dim = len(candidate_texts)

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Add phase information optionally to the direction instruction (to capture better when it is at a wrong tube)
        #if self.use_phase_history_for_moving_direction_and_corr_pred_flag:
        #    multitask_moving_direction_and_corr_input_dim = hidden_dim if use_jaw_values_flag else image_output_dim
        #    multitask_moving_direction_and_corr_input_dim += phase_history_output_dim
            #if "dominant_moving_direction" in selected_multitasks:
            #    num_outputs = selected_multitasks_output_dim_dict["dominant_moving_direction"]
            #    self.multitask_mlp_heads["dominant_moving_direction"] = nn.Linear(multitask_moving_direction_and_corr_input_dim, num_outputs)
            #if "is_correction" in selected_multitasks and self.use_phase_history_for_moving_direction_and_corr_pred_flag:
            #    num_outputs = selected_multitasks_output_dim_dict["is_correction"]
            #    self.multitask_mlp_heads["is_correction"] = nn.Linear(multitask_moving_direction_and_corr_input_dim, num_outputs)

        # TODO: Check if architecture is correct
        # Future frame prediction
        self.future_frame_prediction_flag = future_frame_prediction_flag 
        if future_frame_prediction_flag:
            self.next_frame_decoder = FrameDecoder(input_dim=hidden_dim, output_channels=3, img_size=image_dim)

        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1))


        total, trainable = count_parameters(self)
        print(f"Total parameters: {total / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable / 1e6:.2f}M")

    def forward(self, images, psm2_psm1_jaw_values=None, phase_history=None):
        
        if self.use_phase_history_flag:
            assert len(phase_history[0]) == self.phase_history_len, f"Phase history should have length {self.phase_history_len}"
        
        # Given images of shape (b, t, k, c, h, w)
        batch_size, timesteps, num_camera_patches, num_channels, h, w = images.shape

        # Apply camera dropout
        if self.training and self.camera_dropout_prob > 0:
            camera_dropout_mask = torch.bernoulli(torch.ones(batch_size, num_camera_patches, device=self.device) * (1 - self.camera_dropout_prob))
            camera_dropout_mask = camera_dropout_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            images = images * camera_dropout_mask

        # Check if padding is required
        if timesteps < self.history_len + 1:
            padding_needed = self.history_len + 1 - timesteps
            padding = torch.zeros(
                (batch_size, padding_needed, num_camera_patches, num_channels, h, w), device=images.device
            )
            images = torch.cat([padding, images], dim=1)
            timesteps = self.history_len + 1  # Update timesteps to reflect the new length

        if self.use_separate_backbones_flag: 
            image_features = []
            for camera_patch_idx, camera_patch_name in enumerate(self.camera_patch_names):
                camera_name = "left_img_dir" if "left_img_dir" in camera_patch_name else camera_patch_name 
                camera_image_reshaped = images[:, :, camera_patch_idx, :, :, :].reshape(batch_size * timesteps, num_channels, h, w)
                camera_backbone_model = self.backbone_models[camera_name]
                dataset_mean, dataset_std = self.dataset_mean_std_camera_dict[camera_name]
                # Apply transformations for backbone model --> backbone model expects images to be normalized and resized e.g. to 224*224
                camera_images_transformed = preprocess_inputs(camera_image_reshaped, self.backbone_model_name, dataset_mean, dataset_std)
                camera_image_features = extract_features(camera_backbone_model, self.backbone_model_name, self.model_init_weights, self.image_dim, camera_images_transformed, self.global_pool_image_features_flag)
                image_features.append(camera_image_features)
            image_features = torch.cat(image_features, dim=0)             
        else:
            images_transformed_list = []
            # Apply transformations for backbone model --> backbone model expects images to be normalized and resized e.g. to 224*224
            for camera_patch_idx, camera_patch_name in enumerate(self.camera_patch_names):
                camera_name = "left_img_dir" if "left_img_dir" in camera_patch_name else camera_patch_name 
                camera_image_reshaped = images[:, :, camera_patch_idx, :, :, :].reshape(batch_size * timesteps, num_channels, h, w)
                dataset_mean, dataset_std = self.dataset_mean_std_camera_dict[camera_name]
                # Apply transformations for backbone model --> backbone model expects images to be normalized and resized e.g. to 224*224
                camera_images_transformed = preprocess_inputs(camera_image_reshaped, self.backbone_model_name, dataset_mean, dataset_std)
                images_transformed_list.append(camera_images_transformed)
            images_transformed = torch.stack(images_transformed_list, dim=1)
            images_transformed = images_transformed.reshape(batch_size * timesteps * num_camera_patches, num_channels, h, w) # Reshape to (b*t*k, c, h, w)

            # Apply dropout to segmentation masks
            if self.use_seg_masks_input_flag:
                if self.training and self.seg_masks_dropout_prob > 0:
                    # Create a dropout mask for the segmentation mask channels (channels > 3)
                    seg_masks_dropout_mask = torch.ones((batch_size * timesteps * num_camera_patches, num_channels, h, w), device=self.device)
                    seg_masks_dropout_mask[:, 3:, :, :] = torch.bernoulli(torch.ones((batch_size * timesteps * num_camera_patches, num_channels - 3, h, w), device=self.device) * (1 - self.seg_masks_dropout_prob))
                    images_transformed = images_transformed * seg_masks_dropout_mask # Apply the dropout mask to the images

            image_features = extract_features(self.backbone_model, self.backbone_model_name, self.model_init_weights, self.image_dim, images_transformed, self.global_pool_image_features_flag)
            # Shape: (b*t*k, d) or (b*t*k, non-pooled patches, d)

        if image_features.ndim == 3:
            num_embedded_patches = image_features.shape[1]
            patch_embedding_size = image_features.shape[2]
            image_features = image_features.reshape(-1, num_embedded_patches * patch_embedding_size)
        else:
            num_embedded_patches = None

        if self.add_center_crop_view_flag and self.merge_global_and_center_embs_flag:
            image_features = image_features.reshape(batch_size, timesteps, num_camera_patches, -1)
            # Identify the indices corresponding to 'left_img_dir' - so the global and central view on the DaVinci endoscope image
            left_img_dir_indices = [i for i, camera_patch_name in enumerate(self.camera_patch_names) if "left_img_dir" in camera_patch_name]

            # Add the embeddings for the global and center crop view for 'left_img_dir'
            if len(left_img_dir_indices) > 1:
                left_img_features = image_features[:, :, left_img_dir_indices].sum(dim=2).unsqueeze(2)
                
                # Remove the individual channels for 'left_img_dir' and replace with the combined feature
                other_indices = [i for i in range(image_features.size(2)) if i not in left_img_dir_indices]
                if other_indices:
                    image_features = torch.cat((image_features[:, :, other_indices], left_img_features), dim=2)
                else:
                    image_features = left_img_features
                    
            num_camera_patches -= 1 # As merging global and center crop views

        # Reshape the image features to [batch_size, timesteps*camera patches, feature_dim]
        if self.temporal_mode in ["tcn", "lstm"]: # As the input needs to be from the same kind
            # TODO (Discussion Pascal & Paul): Bring patch dimension back if not None?
            # TODO: Check if the current code is correct.
            image_features_reshaped = image_features.reshape(
                batch_size, timesteps, self.num_cameras, -1
            ).to(torch.float32)
        else:
            image_features_reshaped = image_features.reshape(
                batch_size, timesteps * num_camera_patches, -1
            ).to(torch.float32) 

        # Use the transformer to process the image features or concatenate them
        if self.temporal_mode == "transformer":          
            # Bring patch dimension back if not None
            if num_embedded_patches is not None:
                image_features_reshaped = image_features_reshaped.reshape(batch_size, timesteps * num_embedded_patches * num_camera_patches, patch_embedding_size)
            assert image_features_reshaped.shape[1:] == self.positional_encoding.shape, "Whoops, sorry. See code for NOTE"
            # NOTE: I removed [: timesteps * num_camera_patches, :] as indexing from self.positional encoding.

            # Add positional encoding
            image_features_reshaped += self.positional_encoding.to(image_features_reshaped.device)

            # Pass the concatenated features through the Transformer
            transformer_out = self.transformer(
                image_features_reshaped
            )

            # Extract the final output of the Transformer for each sequence in the batch
            final_image_output = transformer_out[:, -1, :]
        elif self.temporal_mode == "tcn":
            image_features_tcn = image_features_reshaped.permute(0, 2, 1)  # (batch_size, channels, timesteps*camera patches)
            final_image_output = self.tcn(image_features_tcn)
        elif self.temporal_mode == "lstm":
            final_image_output, _ = self.lstm(image_features_reshaped)
            final_image_output = final_image_output[:, -1, :]
        else:
            # Concatenate the image features
            final_image_output = image_features_reshaped.reshape(batch_size, -1)            

        # if "dominant_moving_direction" in self.selected_multitasks or "is_correction" in self.selected_multitasks:
            # Take only the last n images for predicting the moving direction and correction and flatten the features
        #    num_dir_and_corr_images = self.moving_direction_and_corr_history_len + 1
        #    if not self.temporal_mode is None:
        #        #TODO
        #        moving_dir_and_corr_input = final_image_output
        #    else:
        #        moving_dir_and_corr_input = image_features.reshape(batch_size, timesteps, num_camera_patches, -1)
        #        moving_dir_and_corr_input = moving_dir_and_corr_input[:, -num_dir_and_corr_images:, :, :].reshape(batch_size, -1)

        # Process the jaw values (if available)
        if self.use_jaw_values_flag:
            # Apply jaw values dropout
            if self.training and self.jaw_values_dropout_prob > 0:
                jaw_input_dropout_mask = torch.bernoulli(torch.ones((batch_size, 1, 2), device=self.device) * (1 - self.jaw_values_dropout_prob))
                psm2_psm1_jaw_values = psm2_psm1_jaw_values * jaw_input_dropout_mask
            psm2_psm1_jaw_values_flattened = psm2_psm1_jaw_values.reshape(batch_size, -1) # Flatten the jaw values: (batch_size, timesteps, 2) -> (batch_size, timesteps*2)
            
            psm2_psm1_jaw_values_features = self.jaw_values_mlp(psm2_psm1_jaw_values_flattened)
            psm2_psm1_jaw_values_image_features = torch.cat((final_image_output, psm2_psm1_jaw_values_features), dim=1)
            final_emb = self.image_jaw_mlp(psm2_psm1_jaw_values_image_features)     
            
            # ---- For direction and correction head ---- #
            #if "dominant_moving_direction" in self.selected_multitasks or "is_correction" in self.selected_multitasks:
            #    # Apply jaw values dropout
            #    if self.training and self.jaw_values_dropout_prob > 0:
            #        jaw_input_dropout_mask_dir_corr = torch.bernoulli(torch.ones((batch_size, 1, 2), device=self.device) * (1 - self.jaw_values_dropout_prob))
            #        psm2_psm1_jaw_values_dir_corr = psm2_psm1_jaw_values[:, -num_dir_and_corr_images:] * jaw_input_dropout_mask_dir_corr
            #    else:
            #        psm2_psm1_jaw_values_dir_corr = psm2_psm1_jaw_values[:, -num_dir_and_corr_images:]
            #    psm2_psm1_jaw_values_flattened_dir_corr = psm2_psm1_jaw_values_dir_corr.reshape(batch_size, -1)
                
            #    psm2_psm1_jaw_values_features_dir_corr = self.jaw_values_mlp_dir_corr(psm2_psm1_jaw_values_flattened_dir_corr)
            #    psm2_psm1_jaw_values_image_features_dir_corr = torch.cat((moving_dir_and_corr_input, psm2_psm1_jaw_values_features_dir_corr), dim=1)
            #    moving_dir_and_corr_input = self.image_jaw_mlp_dir_corr(psm2_psm1_jaw_values_image_features_dir_corr)
                 
        else:
            final_emb = final_image_output

        # Forward pass for each multitask output (without phase information)
        multitask_logits_dict = {} 
        if self.selected_multitasks:
            for multitask, mlp_head in self.multitask_mlp_heads.items():
                # TODO weg?
                #if multitask in ["dominant_moving_direction", "is_correction"]:
                #    if not self.use_phase_history_for_moving_direction_and_corr_pred_flag:
                        # TODO MLP size in init.
                #        multitask_logits_dict[multitask] = mlp_head(moving_dir_and_corr_input)
                #else:
                multitask_logits_dict[multitask] = mlp_head(final_emb)

        # Process the phase history (if available)
        if self.use_phase_history_flag or self.use_phase_history_for_moving_direction_and_corr_pred_flag:
            # Apply phase history dropout
            if self.training and self.phase_history_dropout_prob > 0:
                phase_history_dropout_mask = torch.bernoulli(torch.ones((batch_size, 1), device=self.device) * (1 - self.phase_history_dropout_prob))
                phase_history = (phase_history * phase_history_dropout_mask).to(torch.int64)
            
            phase_history_embeddings = self.phase_embedding(phase_history)
            phase_history_embeddings_reshaped = phase_history_embeddings.reshape(batch_size, -1)
            phase_history_output = self.history_mlp(phase_history_embeddings_reshaped)
            
            # Concatenate the final embedding with the phase history output
            if self.use_phase_history_flag:
                final_emb = torch.cat((final_emb, phase_history_output), dim=1)
            #if ("dominant_moving_direction" in self.selected_multitasks or "is_correction" in self.selected_multitasks) and self.use_phase_history_for_moving_direction_and_corr_pred_flag:
            #    moving_dir_and_corr_input = torch.cat((moving_dir_and_corr_input, phase_history_output), dim=1)

        # # Forward pass for the dominant moving direction and correction (if using phase information)
        # if self.use_phase_history_for_moving_direction_and_corr_pred_flag: 
        #     if "dominant_moving_direction" in self.selected_multitasks:
        #         multitask_logits_dict["dominant_moving_direction"] = self.multitask_mlp_heads["dominant_moving_direction"](final_emb)
        #     if "is_correction" in self.selected_multitasks:
        #         multitask_logits_dict["is_correction"] = self.multitask_mlp_heads["is_correction"](final_emb)

        if self.one_hot_flag:
            # Directly predict the logits for each command
            logits = self.mlp(final_emb)
            command_emb_pred = final_emb # From transformer/MLP
        else:
            # Predict the command embedding
            command_emb_pred = self.mlp(final_emb)
            # Compute the similarity scores as logits
            logits = self.compute_similarities(command_emb_pred) / self.temperature.clamp(min=1e-8)
        
        # TODO: Check if this works
        if self.future_frame_prediction_flag:
            # Predict the next frame
            predicted_frame_logits = self.next_frame_decoder(final_emb)
            multitask_logits_dict["future_frame_prediction"] = predicted_frame_logits
        
        return logits, command_emb_pred, multitask_logits_dict, self.temperature

    def get_backbone_and_other_params(self):
        # Returns the parameter groups for the optimizer, e.g., using different learning rates for the backbone and the other parameters 
        
        # Collect the backbone parameters
        if self.use_separate_backbones_flag:
            # If using separate backbones for each camera
            backbone_params = []
            for camera_name, backbone in self.backbone_models.items():
                backbone_params += list(backbone.parameters())
        else:
            # If using a single shared backbone
            backbone_params = list(self.backbone_model.parameters())

        # Collect other parameters (excluding the backbone)
        other_params = [
            param for name, param in self.named_parameters() 
            if "backbone" not in name  # Exclude all backbone parameters
        ]
        
        return backbone_params, other_params

    def compute_similarities(self, embeddings):
        # Compute the cosine similarities
        cosine_similarities = (embeddings @ self.candidate_embeddings.T) / (
            embeddings.norm(dim=-1, keepdim=True)
            * self.candidate_embeddings.norm(dim=-1, keepdim=True).T
        )

        return cosine_similarities

    @staticmethod
    def create_sinusoidal_embeddings(d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def decode_logits(self, logits, temperature):
        # Returns the command with the highest logit 
                
        # Compute the probabilities
        probs = (
            logits
            if self.one_hot_flag
            else torch.nn.functional.softmax(logits / temperature, dim=-1)
        )

        # Find the indices of the max logit for each example in the batch
        _, max_indices = torch.max(probs, dim=-1)

        return [self.candidate_texts[index] for index in max_indices.cpu().numpy()]

    def get_nearest_embedding(self, embeddings):
        # Compute cosine similarities
        similarities = self.compute_similarities(embeddings)

        # Get the index of the maximum similarity for each prediction
        indices = similarities.argmax(dim=-1)

        # Print the top 5 candidates
        probs = torch.nn.functional.softmax(similarities, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], 5)
        normalized_top_probs = top_probs / top_probs.sum()
        for i, (index, prob) in enumerate(zip(top_indices, normalized_top_probs)):
            print(
                f"Candidate {i}: {self.candidate_texts[index]}, Normalized Prob: {prob:.4f}"
            )

        # Map the indices back to the embeddings
        return [self.candidate_embeddings[i] for i in indices.cpu().numpy()]

    def get_random_from_top_k(self, embeddings, k=3):
        similarities = (embeddings @ self.candidate_embeddings.T) / (
            embeddings.norm(dim=-1, keepdim=True)
            * self.candidate_embeddings.norm(dim=-1, keepdim=True).T
        )
        top_k_indices = similarities.topk(k, dim=-1)[1]

        # Randomly select one from the top-k for each row
        selected_indices = [
            random.choice(indices_row) for indices_row in top_k_indices.cpu().numpy()
        ]

        return [self.candidate_texts[i] for i in selected_indices]

    def sample_with_temperature(self, embeddings, temperature=1.0):
        similarities = (embeddings @ self.candidate_embeddings.T) / (
            embeddings.norm(dim=-1, keepdim=True)
            * self.candidate_embeddings.norm(dim=-1, keepdim=True).T
        )
        probs = torch.nn.functional.softmax(similarities / temperature, dim=-1)
        sampled_indices = torch.multinomial(
            probs, 1
        ).squeeze()  # Squeezing to potentially remove singleton dimensions
        # Check if sampled_indices is a scalar (0-dim) or an array
        if sampled_indices.ndim == 0:
            # If it's a scalar, we make it a one-element array
            sampled_indices = [sampled_indices.item()]
        else:
            # Otherwise, we convert it to a list
            sampled_indices = sampled_indices.tolist()

        return [self.candidate_texts[i] for i in sampled_indices]


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# Example usage:
if __name__ == "__main__":      
    import os
    import matplotlib.pyplot as plt
    from torchvision.transforms import v2
    import albumentations as A
    import cv2
    
    from instructor.dataset_daVinci import load_merged_data, SequenceDataset
    from instructor.utils import set_seed   

    # Parameters for the test
    gpu = 0
    tissue_samples_ids = [1]
    camera_names = ["left_img_dir"] # ["endo_psm2", "left_img_dir", "endo_psm1"] # "right_img_dir"
    camera_file_suffixes = ["_left.jpg"] # ["_psm2.jpg", "_left.jpg", "_psm1.jpg"] # "_right.jpg"
    dataset_names = ["base_chole_clipping_cutting"] # , "phantom_chole", "base_chole_clipping_cutting_amos"]
    datasets_dir = [os.path.join(os.getenv("PATH_TO_DATASET"), dataset_name) for dataset_name in dataset_names]
    num_episodes_list = [200]*len(datasets_dir)
    batch_size_train = batch_size_val=  2
    history_len = 3
    prediction_offset = 12 # Get command for the current timestep
    history_step_size = 12
    num_episodes = 200 # Number of randlomy generated stitched episodes
    use_phase_history_flag = True
    phase_history_len = 2
    use_jaw_values_flag = True
    use_img_transformer_flag = False 
    reduced_base_instruction_set_flag = False
    one_hot_flag = True
    backbone_model_name = "resnet"
    model_init_weights = "dino"
    image_dim = (224, 224)
    prediction_step_size = 30
    recovery_probability = 0.8
    phase_history_only_phase_switches_flag = True
    camera_dropout_prob = jaw_values_dropout_prob = phase_history_dropout_prob=0.4
    llava_anyres_flag = True
    no_llava_anyres_global_image_flag = False
    wrist_images_rel_width = 3/4
    llava_anyres_rel_width = 2/3
    uniform_sampling_flag = False
    selected_multitasks = ["psm1_instrument", "curr_tube", "total_number_clips", "clip_loaded", "gallbladder_grabbed", "clips_left_tube", "clips_right_tube", "psm2_instrument_closed", "psm1_instrument_closed"]
    use_seg_masks_input_flag = False
    merge_seg_masks_flag = False
    seg_mask_objs = ["clips", "left_tube", "right_tube"] # "clips", "left_tube", "right_tube", "flap"
    seg_masks_dropout_prob = 1
    extra_corrections_sampling_flag = False
    extra_corrections_sampling_probability = 0
    extra_repeated_phase_last_frame_sampling_flag = True
    extra_repeated_phase_last_frame_sampling_probability= 0.2
    add_center_crop_view_flag = True
    merge_global_and_center_embs_flag=False
    distance_from_border_y=0.1
    distance_from_border_x=0.25
    y_offset=-0.1
    temporal_mode="transformer"

    cmap = plt.get_cmap("tab10") # Color map to use

    # Define transforms/augmentations (resize transformation already applied in __getitem__ method)
    torch_input_transforms = []
    
    # NOTE: Automatic augmentations
    # torch_input_transforms.append(transforms.RandAugment())
    # torch_input_transforms.append(transforms.TrivialAugmentWide())
    # torch_input_transforms.append(transforms.AugMix())
    
    # NOTE: Manual augmentations
    # Torch augmentations
    torch_input_transforms.append(transforms.RandomResizedCrop(image_dim, scale=(0.8, 1.0), antialias=True))
    torch_input_transforms.append(transforms.RandomRotation(degrees=[-5.0, 5.0]))
    torch_input_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)))
    # torch_color_input_transforms.append(v2.RandomPerspective(p=0.5))
    
    # Torch Color augmentations
    torch_color_input_transforms = []
    torch_color_input_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.5, hue=0.08))
    # torch_color_input_transforms.append(v2.RandomPosterize(bits=7, p=0.25))
    # torch_color_input_transforms.append(v2.RandomAdjustSharpness(2, p=0.25))
    # torch_color_input_transforms.append(transforms.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.75))
    # torch_color_input_transforms.append(v2.RandomPhotometricDistort(p=0.8))
    # torch_color_input_transforms.append(transforms.RandomGrayscale(p=0.2))
    
    # Albumentations augmentations
    albumentation_input_transforms = []
    min_height, min_width = max(1, image_dim[0] // 40), max(1, image_dim[1] // 40)
    max_height, max_width = min(image_dim[0] // 30, image_dim[0]), min(image_dim[1] // 30, image_dim[1]) 
    albumentation_input_transforms.append(A.CoarseDropout(max_holes=128, max_height=max_height, max_width=max_width, min_holes=1, min_height=min_height, 
                    min_width=min_width, fill_value=0, p=0.5))
    
    # Store the transforms in a dictionary
    torch_input_transforms = transforms.Compose(torch_input_transforms)
    torch_color_input_transforms = transforms.Compose(torch_color_input_transforms)
    num_patches = SequenceDataset.get_num_patches(camera_names, add_center_crop_view_flag, llava_anyres_flag, no_llava_anyres_global_image_flag)
    num_input_images = num_patches * (history_len + 1)
    albumentations_additional_targets = dict(zip([f"image{i}" for i in range(num_input_images)], ["image"] * num_input_images))    
    albumentation_input_transforms = A.Compose(albumentation_input_transforms, additional_targets=albumentations_additional_targets)
    input_transforms = {"torch": torch_input_transforms, "albumentations": albumentation_input_transforms, "torch_color": torch_color_input_transforms}

    # Load the dataloader
    train_dataloader, val_dataloader, ds_metadata_dict = load_merged_data(
        dataset_dirs=datasets_dir,
        num_episodes_list=num_episodes_list,
        camera_names=camera_names,
        camera_file_suffixes=camera_file_suffixes,
        history_len=history_len,
        prediction_offset=prediction_offset,
        history_step_size=history_step_size,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
        input_transforms=input_transforms,
        use_phase_history_flag=use_phase_history_flag,
        use_jaw_values_flag=use_jaw_values_flag,
        reduced_base_instruction_set_flag=reduced_base_instruction_set_flag,
        phase_history_len=phase_history_len,
        prediction_step_size=prediction_step_size,
        recovery_probability=recovery_probability,
        phase_history_only_phase_switches_flag=phase_history_only_phase_switches_flag,
        image_dim=image_dim,
        llava_anyres_flag=llava_anyres_flag,
        no_llava_anyres_global_image_flag=no_llava_anyres_global_image_flag,
        wrist_images_rel_width=wrist_images_rel_width,
        llava_anyres_rel_width=llava_anyres_rel_width,
        uniform_sampling_flag=uniform_sampling_flag,
        selected_multitasks=selected_multitasks,
        use_seg_masks_input_flag=use_seg_masks_input_flag,
        seg_mask_objs=seg_mask_objs,
        merge_seg_masks_flag=merge_seg_masks_flag,
        extra_corrections_sampling_flag=extra_corrections_sampling_flag,
        extra_corrections_sampling_probability=extra_corrections_sampling_probability,
        extra_repeated_phase_last_frame_sampling_flag=extra_repeated_phase_last_frame_sampling_flag,
        extra_repeated_phase_last_frame_sampling_probability=extra_repeated_phase_last_frame_sampling_probability,
        add_center_crop_view_flag=add_center_crop_view_flag,
        distance_from_border_y=distance_from_border_y,
        distance_from_border_x=distance_from_border_x,
        y_offset=y_offset
    )    
    candidate_embeddings = ds_metadata_dict["candidate_embeddings"]
    candidate_texts = ds_metadata_dict["candidate_texts"]
    
    # Load the model
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    candidate_embeddings = candidate_embeddings.to(device)
    command_to_index = {command: i for i, command in enumerate(candidate_texts)}
    model = Instructor(
        device=device,
        history_len=history_len,
        history_step_size=history_step_size,
        prediction_offset=prediction_offset,
        candidate_embeddings=candidate_embeddings,
        candidate_texts=candidate_texts,
        command_to_index=command_to_index,
        camera_names=camera_names,
        use_jaw_values_flag=use_jaw_values_flag,
        use_phase_history_flag=use_phase_history_flag,
        temporal_mode=temporal_mode,
        one_hot_flag=one_hot_flag,
        backbone_model_name=backbone_model_name,
        model_init_weights=model_init_weights,
        phase_history_len=phase_history_len,
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
        distance_from_border_y=distance_from_border_y,
        distance_from_border_x=distance_from_border_x,
        y_offset=y_offset,
        merge_global_and_center_embs_flag=merge_global_and_center_embs_flag
    )
    model.to(device)

    idx_in_batch = 0
    for split_name, dataloader in [("train", train_dataloader), ("val", val_dataloader)]:
        # Fetch a batch of data and pass it through the model
        for batch in dataloader:
            if use_jaw_values_flag:
                image_sequence, command_embedding, gt_command, jaw_values, phase_history, multitask_label_indices_dict = batch
            else:
                image_sequence, command_embedding, gt_command, phase_history, multitask_label_indices_dict = batch
            image_sequence = image_sequence.to(device)
            if use_jaw_values_flag:
                jaw_values = jaw_values.to(device)
            if use_phase_history_flag:
                phase_history_indexed = [[model.history_phase_to_index[phase_command_list[batch_idx]] for batch_idx in range(len(phase_command_list))] for phase_command_list in phase_history]
                phase_history_indexed = torch.tensor(phase_history_indexed).to(device)
            next_phase_predictions_logits, command_emb_pred, multitask_logits_dict, temperature = model(image_sequence, jaw_values, phase_history_indexed)            
            pred_command = model.decode_logits(next_phase_predictions_logits, temperature)
            
            print(f"\nSplit: {split_name}")
            print(f"Image sequence shape: {image_sequence.shape}")
            print(f"Language data shape: {command_embedding.shape}")
            print(f"Predictions shape: {next_phase_predictions_logits.shape}")
            print(f"Ground truth command ({prediction_offset=}): {gt_command[idx_in_batch]}")
            print(f"Predicted command [untrained] ({prediction_offset=}): {pred_command[idx_in_batch]}\n")
            print(f"Phase history (idx=0): {[phase[0] for phase in phase_history]}")
            if use_jaw_values_flag:
                print(f"Jaw values (idx=0): {jaw_values[0]}")
            if multitask_label_indices_dict:
                multitask_label_dict = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(multitask_label_indices_dict, batch_wise=True)
                multitask_pred_label_dict = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(multitask_logits_dict, batch_wise=True, logits_flag=True)
                print("Multitask GT-Pred labels:")
                for k in multitask_label_dict:
                    print(f"{k}: {multitask_label_dict[k][idx_in_batch]} - {multitask_pred_label_dict[k][idx_in_batch]}")
            if not one_hot_flag:
                print(f"Temperature: {temperature.item()}")
            break

        # Create a figure with subplots: one row per timestamp, one column per camera
        fig_height = 4 * (history_len + 1)
        camera_patch_names = SequenceDataset.get_camera_patch_names(camera_names, add_center_crop_view_flag, llava_anyres_flag, no_llava_anyres_global_image_flag)
        fig_width = len(camera_patch_names) * 3
        fig, axes = plt.subplots(history_len + 1, len(camera_patch_names), figsize=(fig_width, fig_height))
        if history_len == 0:
            axes = axes[np.newaxis, :]
        if len(camera_patch_names) == 1:
            axes = axes[:, np.newaxis]

        # Loop over each timestamp and camera to plot the images
        for t in range(history_len + 1):
            for cam_idx, cam_patch_name in enumerate(camera_patch_names):
                ax = axes[t, cam_idx]  # Get the specific subplot axis
                img = image_sequence[0, t, cam_idx].permute(1, 2, 0).detach().cpu().numpy()
                
                # If the image is stacked with segmentation masks, extract the contours of the masks and put them on the image
                if img.shape[2] > 3:
                    img_bgr = cv2.cvtColor(img[:, :, :3]*255, cv2.COLOR_RGB2BGR).astype(np.uint8)  # Convert to BGR for OpenCV
                    seg_masks = image_sequence[0, t, cam_idx, 3:]  # The remaining channels are segmentation masks
                    if merge_seg_masks_flag:
                        class_ids = torch.unique(seg_masks)
                        seg_masks_per_class = [(seg_masks == class_id).numpy().astype(np.uint8).squeeze() for class_id in class_ids if class_id != 0]  # Skip the background class (obj_id=0)
                    else:
                        seg_masks_per_class = [seg_mask.squeeze().cpu().numpy().astype(np.uint8) for seg_mask in seg_masks]
                    for class_id, seg_mask in enumerate(seg_masks_per_class, start=1):
                        if seg_mask.sum() == 0:
                            print(f"Segmentation mask for class {class_id} is empty.")
                            continue
                        contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        class_color = np.array(cmap(class_id-1)[:3])  # Extract only RGB values
                        class_color_bgr = tuple((class_color * 255).astype(int)[::-1].tolist())  # Convert to BGR and ensure it's a tuple
                        cv2.drawContours(img_bgr, contours, -1, class_color_bgr, 2)  # Draw the contours on the BGR image
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert back to RGB for Matplotlib
                
                ax.imshow(img)
                ax.set_title(f"{cam_patch_name} at timestep {t}")
                ax.axis('off')  # Optionally turn off the axis

        # Set title to command
        fig.suptitle(f"Gt Command: {gt_command[idx_in_batch]}\nPrediction [untrained]: {pred_command[idx_in_batch]}")
        plt.tight_layout()
        example_dataset_plots_folder_path = os.path.join(path_to_yay_robot, "examples_plots", "untrained_model_pred")
        if not os.path.exists(example_dataset_plots_folder_path):
            os.makedirs(example_dataset_plots_folder_path)
        file_name = os.path.join(example_dataset_plots_folder_path, f"untrained_model_pred_img_{split_name=}{history_len=}_{history_step_size=}_{image_dim=}_{use_seg_masks_input_flag=}.png")
        file_path = os.path.join(example_dataset_plots_folder_path, file_name)
        plt.savefig(file_path)
        print(f"Saved {file_name}\n\n---------")
        plt.close(fig)  # Close the figure to free memory