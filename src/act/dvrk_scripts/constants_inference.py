import pathlib
import numpy as np
import os

"""
Directions: set task name, action mode (e.g. hybrid, ego, etc), and the correct normalization params (usually just mean/std).
Make sure action mode is exactly what you want!
"""


### Task parameters
# DATA_DIR = "/home/imerse/chole_ws/data"
DATA_DIR = os.getenv("PATH_TO_DATASET")
TASK_CONFIGS = {
    



    ## tissue 4,5,6 (except needle throw from tissue 1 to 2)

    'suturing_final':{
        'dataset_dir': DATA_DIR,
        'phantom': False,
        'use_auto_label': False,
        'goal_condition_style': 'map',
        'no_qpos': False,     ## True when SRT, False when ACT
        'num_episodes': 671,  # 112 + 110 + 332 (after filtering to 3 core tasks)
        'num_episodes_val': 12,
        'tissue_samples_ids': [4, 5, 6],
        'tissue_samples_ids_val': [10],
        'selected_phases': ['3_knot_tying_recovery', '1_needle_pickup', '1_needle_pickup_recovery', '3_knot_tying', '2_needle_throw', '2_needle_throw_recovery'],  # Filter out recovery tasks
        'camera_file_suffixes':  ["_left.jpg", "_psm2.jpg", "_psm1.jpg"],   # need to match camera_names below
        'episode_len': 500, # not to be confused with number of demos
        'cutting_action_pad_size': 0,
        "recovery_ratio": 1.0,
        'action_mode': ['hybrid',
        {'max_': np.array([0.04279040962838755, 0.054697995116330854, 0.05486217567194245, 0.9999999999906766, 0.9296439157046034, 0.7821769384033312, 0.9471267964639998, 0.999999999925747, 0.9894531097697882, 1.3273867625611704, 0.03833946699978986, 0.03973159750372202, 0.02462215367946551, 0.9999999998642655, 0.8134547359216306, 0.7834710376721901, 0.8089995111420474, 0.99999999997923, 0.8334898656034556, 1.3803311487867052]), 

        'min_': np.array([-0.036464446235394354, -0.06388680898925794, -0.02038767948685958, 0.27494876273153257, -0.95078160471653, -0.7807861850870974, -0.9378354946185374, -0.09716974193735776, -0.9316882359192845, -0.3490660000000003, -0.041246261683692914, -0.04026145768661816, -0.013500450387770119, 0.3767397198435095, -0.8080026794163349, -0.8442369232507425, -0.748795943457156, 0.48007394917559987, -0.6662898274986029, -0.3490660000000003]),

        'mean': np.array([0.001410389793981085, -0.0007131324515301552, 0.0009840458086499476, 0.9752912632977296, 0.001981641757904754, 0.011873039337537363, -0.003069231798042605, 0.9532981255042895, 0.023196625174659785, -0.23019795411250024, -0.0012582801840106527, 0.0007635927359010288, 0.0004619426873278165, 0.9794179996513332, -0.01101860258228741, -0.0018037690554784205, 0.01315729790361867, 0.9831141293118357, 0.02670289302773284, 0.05052287212104786]),

        'std': np.array([0.01, 0.010416102610019736, 0.01, 0.048063671033240475, 0.16644192420999918, 0.13656143364830978, 0.1733778223038079, 0.09153206278080891, 0.2285545723071133, 0.3110773060496116, 0.01, 0.01, 0.01, 0.04556944222815957, 0.144689189481379, 0.13268075653851708, 0.15087615579563593, 0.0390566217077149, 0.09116696592770279, 0.3257051074911022]) }],
        
        'norm_scheme': 'std',
        'save_frequency': 50,
        'camera_names': ['left', 'left_wrist', 'right_wrist', 'mask'],
        'merging_subtasks': False,
        },

    'suturing_final_gail':{
        'dataset_dir': DATA_DIR,
        'phantom': False,
        'use_auto_label': False,
        'goal_condition_style': 'map',
        'no_qpos': False,     ## True when SRT_GAIL, False when ACT_GAIL
        'num_episodes': 671,  # All 6 tasks from tissues 4, 5, 6
        'num_episodes_val': 12,
        'tissue_samples_ids': [4, 5, 6],
        'tissue_samples_ids_val': [10],
        'selected_phases': ['3_knot_tying_recovery', '1_needle_pickup', '1_needle_pickup_recovery', '3_knot_tying', '2_needle_throw', '2_needle_throw_recovery'],
        'camera_file_suffixes':  ["_left.jpg", "_psm2.jpg", "_psm1.jpg"],
        'episode_len': 500,
        'cutting_action_pad_size': 0,
        "recovery_ratio": 1.0,
        
        # GAIL-specific parameters
        'use_gail': True,
        'gram_weight': 0.1,  # Weight for Gram anchoring loss
        'gram_loss_type': 'frobenius',  # Options: 'frobenius', 'cosine', 'kl'
        
        # Multi-level options (choose one):
        'use_multilevel_gram': True,  # PRIMARY+SECONDARY+TERTIARY: Policy + Visual + Action
        'multilevel_weights': [1.0, 0.5, 0.0],  # [policy, visual, action]
        
        'use_hierarchical_gram': False,  # Multi-scale on policy only
        'hierarchical_scales': [1, 4, 16],  # Multi-scale if hierarchical
        'hierarchical_weights': [1.0, 0.5, 0.0],
        
        'expert_features_path': 'expert_features_tissues_4_5_6.pt',  # Pre-computed expert hidden states
        
        # Same normalization stats as baseline
        'action_mode': ['hybrid',
        {'max_': np.array([0.04279040962838755, 0.054697995116330854, 0.05486217567194245, 0.9999999999906766, 0.9296439157046034, 0.7821769384033312, 0.9471267964639998, 0.999999999925747, 0.9894531097697882, 1.3273867625611704, 0.03833946699978986, 0.03973159750372202, 0.02462215367946551, 0.9999999998642655, 0.8134547359216306, 0.7834710376721901, 0.8089995111420474, 0.99999999997923, 0.8334898656034556, 1.3803311487867052]), 

        'min_': np.array([-0.036464446235394354, -0.06388680898925794, -0.02038767948685958, 0.27494876273153257, -0.95078160471653, -0.7807861850870974, -0.9378354946185374, -0.09716974193735776, -0.9316882359192845, -0.3490660000000003, -0.041246261683692914, -0.04026145768661816, -0.013500450387770119, 0.3767397198435095, -0.8080026794163349, -0.8442369232507425, -0.748795943457156, 0.48007394917559987, -0.6662898274986029, -0.3490660000000003]),

        'mean': np.array([0.001410389793981085, -0.0007131324515301552, 0.0009840458086499476, 0.9752912632977296, 0.001981641757904754, 0.011873039337537363, -0.003069231798042605, 0.9532981255042895, 0.023196625174659785, -0.23019795411250024, -0.0012582801840106527, 0.0007635927359010288, 0.0004619426873278165, 0.9794179996513332, -0.01101860258228741, -0.0018037690554784205, 0.01315729790361867, 0.9831141293118357, 0.02670289302773284, 0.05052287212104786]),

        'std': np.array([0.01, 0.010416102610019736, 0.01, 0.048063671033240475, 0.16644192420999918, 0.13656143364830978, 0.1733778223038079, 0.09153206278080891, 0.2285545723071133, 0.3110773060496116, 0.01, 0.01, 0.01, 0.04556944222815957, 0.144689189481379, 0.13268075653851708, 0.15087615579563593, 0.0390566217077149, 0.09116696592770279, 0.3257051074911022]) }],
        
        'norm_scheme': 'std',
        'save_frequency': 50,
        'camera_names': ['left', 'left_wrist', 'right_wrist', 'mask'],
        'merging_subtasks': False,
        },

    'suturing_final_multilevel_gail':{
        'dataset_dir': DATA_DIR + "/Jesse/",
        'phantom': False,
        'use_auto_label': False,
        'goal_condition_style': 'map',
        'no_qpos': False,     ## True when SRT_GAIL, False when ACT_GAIL
        'num_episodes': 671,  # All 6 tasks from tissues 4, 5, 6
        'num_episodes_val': 12,
        'tissue_samples_ids': [4, 5, 6],
        'tissue_samples_ids_val': [10],
        'selected_phases': ['3_knot_tying_recovery', '1_needle_pickup', '1_needle_pickup_recovery', '3_knot_tying', '2_needle_throw', '2_needle_throw_recovery'],
        'camera_file_suffixes':  ["_left.jpg", "_psm2.jpg", "_psm1.jpg"],
        'episode_len': 500,
        'cutting_action_pad_size': 0,
        "recovery_ratio": 1.0,
        
        # MULTI-LEVEL GAIL parameters
        'use_gail': True,
        'gram_weight': 1.0,  # Overall weight for Gram anchoring loss
        'gram_loss_type': 'frobenius',  # Options: 'frobenius', 'cosine', 'kl'
        
        # Multi-level: Policy + Visual + Action
        'use_multilevel_gram': True,
        'multilevel_weights': [1.0, 0.5, 0.0],  # [policy, visual, action]
        
        # Alternative: Hierarchical (multi-scale on policy only)
        'use_hierarchical_gram': False,
        'hierarchical_scales': [1, 4, 16],
        'hierarchical_weights': [1.0, 0.5, 0.0],
        
        'expert_features_path': 'expert_features_multilevel_tissues_4_5_6.pt',  # Pre-computed multi-level features
        
        # Same normalization stats as baseline
        'action_mode': ['hybrid',
        {'max_': np.array([0.04279040962838755, 0.054697995116330854, 0.05486217567194245, 0.9999999999906766, 0.9296439157046034, 0.7821769384033312, 0.9471267964639998, 0.999999999925747, 0.9894531097697882, 1.3273867625611704, 0.03833946699978986, 0.03973159750372202, 0.02462215367946551, 0.9999999998642655, 0.8134547359216306, 0.7834710376721901, 0.8089995111420474, 0.99999999997923, 0.8334898656034556, 1.3803311487867052]), 

        'min_': np.array([-0.036464446235394354, -0.06388680898925794, -0.02038767948685958, 0.27494876273153257, -0.95078160471653, -0.7807861850870974, -0.9378354946185374, -0.09716974193735776, -0.9316882359192845, -0.3490660000000003, -0.041246261683692914, -0.04026145768661816, -0.013500450387770119, 0.3767397198435095, -0.8080026794163349, -0.8442369232507425, -0.748795943457156, 0.48007394917559987, -0.6662898274986029, -0.3490660000000003]),

        'mean': np.array([0.001410389793981085, -0.0007131324515301552, 0.0009840458086499476, 0.9752912632977296, 0.001981641757904754, 0.011873039337537363, -0.003069231798042605, 0.9532981255042895, 0.023196625174659785, -0.23019795411250024, -0.0012582801840106527, 0.0007635927359010288, 0.0004619426873278165, 0.9794179996513332, -0.01101860258228741, -0.0018037690554784205, 0.01315729790361867, 0.9831141293118357, 0.02670289302773284, 0.05052287212104786]),

        'std': np.array([0.01, 0.010416102610019736, 0.01, 0.048063671033240475, 0.16644192420999918, 0.13656143364830978, 0.1733778223038079, 0.09153206278080891, 0.2285545723071133, 0.3110773060496116, 0.01, 0.01, 0.01, 0.04556944222815957, 0.144689189481379, 0.13268075653851708, 0.15087615579563593, 0.0390566217077149, 0.09116696592770279, 0.3257051074911022]) }],
        
        'norm_scheme': 'std',
        'save_frequency': 50,
        'camera_names': ['left', 'left_wrist', 'right_wrist', 'mask'],
        'merging_subtasks': False,
        },

    'suturing_dp_dot':{
        'dataset_dir': DATA_DIR,
        'phantom': False,
        'use_auto_label': False,
        'goal_condition_style': 'dot',
        'no_qpos': False,     ## True when SRT, False when ACT
        'num_episodes': 671,
        'num_episodes_val': 12,
        'tissue_samples_ids': [4, 5, 6],
        'tissue_samples_ids_val': [10],
        'camera_file_suffixes':  ["_left.jpg", "_psm2.jpg", "_psm1.jpg"],   # need to match camera_names below
        'episode_len': 500, # not to be confused with number of demos
        'cutting_action_pad_size': 0,
        "recovery_ratio": 1.0,
        'action_mode': ['hybrid',
        {'max_': np.array([0.04279040962838755, 0.054697995116330854, 0.05486217567194245, 0.9999999999906766, 0.9296439157046034, 0.7821769384033312, 0.9471267964639998, 0.999999999925747, 0.9894531097697882, 1.3273867625611704, 0.03833946699978986, 0.03973159750372202, 0.02462215367946551, 0.9999999998642655, 0.8134547359216306, 0.7834710376721901, 0.8089995111420474, 0.99999999997923, 0.8334898656034556, 1.3803311487867052]), 

        'min_': np.array([-0.036464446235394354, -0.06388680898925794, -0.02038767948685958, 0.27494876273153257, -0.95078160471653, -0.7807861850870974, -0.9378354946185374, -0.09716974193735776, -0.9316882359192845, -0.3490660000000003, -0.041246261683692914, -0.04026145768661816, -0.013500450387770119, 0.3767397198435095, -0.8080026794163349, -0.8442369232507425, -0.748795943457156, 0.48007394917559987, -0.6662898274986029, -0.3490660000000003]),

        'mean': np.array([0.001410389793981085, -0.0007131324515301552, 0.0009840458086499476, 0.9752912632977296, 0.001981641757904754, 0.011873039337537363, -0.003069231798042605, 0.9532981255042895, 0.023196625174659785, -0.23019795411250024, -0.0012582801840106527, 0.0007635927359010288, 0.0004619426873278165, 0.9794179996513332, -0.01101860258228741, -0.0018037690554784205, 0.01315729790361867, 0.9831141293118357, 0.02670289302773284, 0.05052287212104786]),

        'std': np.array([0.01, 0.010416102610019736, 0.01, 0.048063671033240475, 0.16644192420999918, 0.13656143364830978, 0.1733778223038079, 0.09153206278080891, 0.2285545723071133, 0.3110773060496116, 0.01, 0.01, 0.01, 0.04556944222815957, 0.144689189481379, 0.13268075653851708, 0.15087615579563593, 0.0390566217077149, 0.09116696592770279, 0.3257051074911022]) }],
        
        'norm_scheme': 'min_max',
        'save_frequency': 50,
        'camera_names': ['left', 'left_wrist', 'right_wrist'],
        'merging_subtasks': False,
        },


}

