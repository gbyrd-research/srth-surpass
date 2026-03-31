import os

### Dataset parameters
DATA_DIR = os.getenv("PATH_TO_DATASET")

INSTRUMENT_CLOSED_THRESHOLD = 0

# NOTE: Choose the dataset name as the dataset dir folder name
DATASET_CONFIGS = {
    "base_chole_clipping_cutting": {
        "dataset_dir": os.path.join(DATA_DIR, "base_chole_clipping_cutting"),
        "num_episodes": 2500, # And for validation doing num_episodes//2
        "tissue_samples_to_exclude": ["tissue_1"], # Should not be used for HL policy training - good for LL and ML policy training 
        "tissue_samples_wrist_cameras_to_exclude": ["tissue_4"],
        "camera_names": ["endo_psm2", "left_img_dir", "right_img_dir", "endo_psm1"], 
        "camera_file_suffixes": ["_psm2.jpg", "_left.jpg", "_right.jpg", "_psm1.jpg"],
        "after_phase_offset": 0,
        "before_phase_offset": 0, 
        "tissue_samples_old_grab_pull_separation": [f"tissue_{idx}" for idx in [1,4,5,6,8,12,13]],
        "correct_psm1_rotation_tissues": [f"tissue_{idx}" for idx in [5,6,8,12,13,14,18]],
        "val_tissues": (["tissue_18", "tissue_35"], ["tissue_40", "tissue_54"], ["tissue_19", "tissue_14"]),
        "test_tissues": [],
        "incomplete_demos_flag": False
    },
    "base_chole_clipping_cutting_amos": {
        "dataset_dir": os.path.join(DATA_DIR, "base_chole_clipping_cutting_amos"),
        "num_episodes": 300, # And for validation doing num_episodes//2
        "tissue_samples_to_exclude": [], # Should not be used for HL policy training - good for LL and ML policy training 
        "tissue_samples_wrist_cameras_to_exclude": [],
        "camera_names": ["endo_psm2", "left_img_dir", "right_img_dir", "endo_psm1"], 
        "camera_file_suffixes": ["_psm2.jpg", "_left.jpg", "_right.jpg", "_psm1.jpg"],
        "after_phase_offset": 0,
        "before_phase_offset": 0, 
        "tissue_samples_old_grab_pull_separation": [f"tissue_{idx}" for idx in [1,4,5,6,8,12,13]],
        "correct_psm1_rotation_tissues": [],
        "val_tissue_split": (["tissue_55"], ["tissue_62"], ["tissue_68"]),
        "test_tissues": [],
        "incomplete_demos_flag": False
    },
    "experiments": {
        "dataset_dir": os.path.join(DATA_DIR, "experiments"),
        "num_episodes": 300, # And for validation doing num_episodes//2
        "tissue_samples_to_exclude": [], # Should not be used for HL policy training - good for LL and ML policy training 
        "tissue_samples_wrist_cameras_to_exclude": [],
        "camera_names": ["endo_psm2", "left_img_dir", "endo_psm1"], 
        "camera_file_suffixes": ["_psm2.jpg", "_left.jpg", "_psm1.jpg"],
        "after_phase_offset": 0,
        "before_phase_offset": 0, 
        "correct_psm1_rotation_tissues": [],
        "val_tissues": (["tissue_100"], ["tissue_101"], ["tissue_102"]),
        "test_tissues": [],
        "incomplete_demos_flag": True
    },
    "phantom_chole": {
        "dataset_dir": os.path.join(DATA_DIR, "phantom_chole"),
        "num_episodes": 500, 
        "tissue_samples_to_exclude": ["phantom_0"], # Should not be used for HL policy training - good for LL and ML policy training 
        "tissue_samples_wrist_cameras_to_exclude": [],
        "camera_names": ["endo_psm2", "left_img_dir", "right_img_dir", "endo_psm1"], 
        "camera_file_suffixes": ["_psm2.jpg", "_left.jpg", "_right.jpg", "_psm1.jpg"],
        "after_phase_offset": 0,
        "before_phase_offset": 0,
        "tissue_samples_old_grab_pull_separation": [f"tissue_{idx}" for idx in [0,1,2,3]],
        "correct_psm1_rotation_tissues": [f"tissue_{idx}" for idx in [0,1,2,3]],
        "incomplete_demos_flag": False
    },
}