#!/usr/bin/env python3
"""
Quick reference for multi-dataset training commands
"""

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  MULTI-DATASET TRAINING QUICK REFERENCE                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

📦 WHAT YOU HAVE NOW:
  ✅ Multi-dataset co-training implementation
  ✅ Automatic dataset balancing
  ✅ Custom dataset weighting support  
  ✅ Handles all dataset format combinations
  ✅ Ready-to-use training script for cnh_exvivo + exvivo_29

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 TRAINING ON CNH_EXVIVO + EXVIVO_29 (YOUR REQUEST):

Option 1 - Shell Script (Easiest):
  
  Auto-balanced:
    $ ./run_multi_cnh_exvivo29.sh
  
  Use GPU 1:
    $ ./run_multi_cnh_exvivo29.sh 1
  
  Custom weights (cnh_exvivo 2x more):
    $ ./run_multi_cnh_exvivo29.sh 0 2.0 1.0

Option 2 - Python Script:
  
  $ python train_multi_dataset_cnh_exvivo29.py --gpu 0
  
  $ python train_multi_dataset_cnh_exvivo29.py \\
      --gpu 0 \\
      --dataset_weights 2.0 1.0 \\
      --batch_size 12

Option 3 - Using imitate_episodes.py (General):
  
  $ python imitate_episodes.py \\
      --task_name cnh_exvivo exvivo_29_auto_label_srt \\
      --dataset_weights 2.0 1.0 \\
      --ckpt_dir ckpt_dir_multi \\
      --policy_class SRT \\
      --batch_size 12 \\
      --gpu 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 DATASET WEIGHTING:

Auto-balanced (default):
  - System automatically balances datasets by size
  - Each dataset contributes equally to training
  - No manual configuration needed

Custom weights:
  --dataset_weights 2.0 1.0    # First dataset 2x more
  --dataset_weights 1.0 2.0    # Second dataset 2x more
  --dataset_weights 3.0 2.0 1.0  # Three datasets with custom weights

When to use custom weights:
  ✓ Prioritize higher-quality datasets
  ✓ Weight more critical tasks higher
  ✓ Manually adjust for specific needs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧪 TESTING BEFORE TRAINING:

Test data loading (recommended first):
  $ python test_multidataset.py

Test all format combinations:
  $ python test_all_format_combinations.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 DOCUMENTATION:

Detailed guide:
  MULTI_DATASET_TRAINING_GUIDE.md
  - Full usage instructions
  - Troubleshooting
  - Advanced configuration
  - Examples

Technical details:
  MULTI_DATASET_FIX_SUMMARY.md
  - Format compatibility fixes
  - Implementation details
  - Dataset format combinations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 SUPPORTED DATASET FORMATS:

All these combinations work automatically:
  ✅ CSV (camera_source + timestamp) + Images (timestamp)
  ✅ CSV (timestamp only) + Images (frame-indexed)  
  ✅ CSV (camera_source + timestamp) + Images (frame-indexed)

Your datasets:
  • cnh_exvivo: ✅ camera_source + timestamp CSV, timestamp images
  • exvivo_29_auto_label_srt: ✅ timestamp CSV, frame images
  • base_chole: ✅ timestamp CSV, frame images
  • Jesse/suturing: ✅ timestamp CSV, frame images

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚙️  CONFIGURATION:

All files configured with your existing training setup:
  • Policy: SRT
  • Image encoder: efficientnet_b3film  
  • Language: distilbert
  • Chunk size: 60
  • Hidden dim: 512
  • Batch size: 12
  • Learning rate: 1e-5
  • KL weight: 10

Modify any parameter via command-line args!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 FILES CREATED:

Training scripts:
  ✅ train_multi_dataset_cnh_exvivo29.py  - Main training script
  ✅ run_multi_cnh_exvivo29.sh           - Shell wrapper
  
Test scripts:
  ✅ test_multidataset.py                - Data loading tests
  ✅ test_all_format_combinations.py      - Format detection tests
  
Documentation:
  ✅ MULTI_DATASET_TRAINING_GUIDE.md     - Complete usage guide
  ✅ MULTI_DATASET_FIX_SUMMARY.md        - Technical details

Core implementation:
  ✅ utils.py: load_data_dvrk_multi_dataset()
  ✅ generic_dataset_invivo.py: Format detection fixes
  ✅ imitate_episodes.py: Auto multi-dataset detection

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎬 READY TO START? RUN THIS:

  $ ./run_multi_cnh_exvivo29.sh

This will start training on both datasets with automatic balancing!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 TIP: Monitor training progress with:
  $ tail -f training_multi_cnh_exvivo29.log

╔══════════════════════════════════════════════════════════════════════════════╗
║                        ENJOY YOUR TRAINING! 🚀                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
