import pathlib
import numpy as np

"""
Direction: make sure that in the task name, you specify what action mode you're using (e.g. knot_tying_HYBRID).
Also, the convention is that the action mode params contains action mode info, and the max-min parameters
of THAT particular action representation.

Carefully check hyperparameters in the trainer train_generic.py
"""

### Task parameters
DATA_DIR = 'save_dir'

TASK_CONFIGS = {
    'knot_tying_hybrid':{
        'dataset_dir': '../brian_knot_tying_12_26_and_27_2023',
        'num_diffusion_iters': 100,
        'num_episodes': 512,
        'action_mode': ['hybrid', 
                        {'max_': np.array([0.02375182, 0.02045474, 0.02108023, 1.        , 0.99563164,
                                        0.91924004, 0.99582772, 1.        , 0.8632355 , 1.42631945,
                                        0.01603716, 0.01864712, 0.02314806, 1.        , 0.99605926,
                                        0.669374  , 0.92886248, 1.        , 0.72124946, 1.25202274]), # jaw angle

                        'min_': np.array([-0.02807084, -0.01803029, -0.01694217, -0.52108483, -0.9445956 ,
                                        -0.58891274, -0.87099595, -0.72066514, -0.66530012, -0.349066  ,
                                        -0.02152792, -0.01365673, -0.01062903, -0.71263727, -0.99287953,
                                        -0.70212741, -0.99898118, -0.70104911, -0.90225085, -0.349066  ]), # jaw angle
                                        }], # diffusion policy always requres min_max scaling so mean_std params are removed to avoid confusion
                        
        'norm_scheme': 'min_max',
        'save_frequency': 250,
        'camera_names': ['left', 'right', 'left_wrist', 'right_wrist'],
    },

    'needle_pickup_handover_hybrid':{
        'dataset_dir': '../brian_needle_pickup_handover',
        'num_episodes': 240,
        'action_mode': ['hybrid', 
                        {'max_': np.array([0.02184909, 0.011936  , 0.01483935, 1.        , 0.99996195,
                        0.41603338, 0.97735615, 1.        , 0.94899971, 1.40146157,
                        0.02553431, 0.02539591, 0.02646763, 1.        , 0.96923362,
                        0.9706872 , 0.99899981, 1.        , 0.86763858, 1.32403904]), # jaw angle

                        'min_': np.array([-0.01979794, -0.01506985, -0.01563597, -0.79181285, -0.99567573,
                            -0.60622445, -0.97014993, -0.51231733, -0.52347339, -0.349066  ,
                            -0.02795868, -0.02831652, -0.02433966, -0.75809375, -0.99999913,
                            -0.85638976, -0.99998054, -0.64346645, -0.99875995, -0.349066  ]), # jaw angle
                        }],
        'num_diffusion_iters': 100,
        'norm_scheme': 'min_max',
        'save_frequency': 400,
        'camera_names': ['left', 'right', 'left_wrist', 'right_wrist'],
    },


    'tissue_lift_hybrid':{
        'dataset_dir': '../tissue_lift_dataset',
        'num_episodes': 225,
        'action_mode': ['hybrid', 
                        {'max_': np.array([9.58660137e-04, 1.75706492e-03, 3.64667092e-03, 1.00000000e+00,
                            3.86978489e-02, 1.18860762e-01, 1.24669370e-02, 1.00000000e+00,
                            4.69644806e-02, 1.17910816e+00, 1.09582952e-02, 2.93239996e-02,
                            1.61841069e-02, 1.00000000e+00, 9.05215216e-01, 6.33072174e-01,
                            8.22567627e-01, 1.00000000e+00, 6.66133118e-01, 1.20431394e+00]), # jaw angle

                        'min_': np.array([-1.24455611e-03, -8.94456938e-04, -8.40753279e-04,  9.92575536e-01,
                            -1.17187559e-02, -5.26328010e-02, -3.87794124e-02,  9.84983897e-01,
                            -1.72568916e-01, -1.16398991e-01, -1.91699568e-02, -1.28107500e-02,
                            -1.13766139e-02, -1.32009680e-01, -8.52134505e-01, -3.83912177e-01,
                            -9.99390286e-01, -2.12526976e-01, -3.34999517e-01, -3.49066000e-01]), # jaw angle
                            }],
        'num_diffusion_iters': 50,
        'norm_scheme': 'min_max',
        'save_frequency': 400,
        'camera_names': ['left', 'right', 'left_wrist', 'right_wrist'],
    },
}