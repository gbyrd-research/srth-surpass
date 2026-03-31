#!/bin/bash
# Example script showing how to add wrist rotation configs to your dataset

# EXAMPLE 1: Add task-level rotation for all episodes in a task
# Use this when all episodes in a task need the same rotation

DATASET_DIR="/path/to/your/dataset"
TASK_NAME="needle_pickup_task"

echo "Creating task-level wrist rotation config..."
cat > "${DATASET_DIR}/${TASK_NAME}/wrist_rotation.json" << EOF
{
  "left_wrist": -52.0,
  "right_wrist": -52.0
}
EOF

echo "✓ Task-level config created at: ${DATASET_DIR}/${TASK_NAME}/wrist_rotation.json"
echo ""

# EXAMPLE 2: Add episode-specific rotation (overrides task-level)
# Use this when specific episodes need different rotations

TISSUE_SAMPLE="tissue_5"
PHASE="3_grasp"
EPISODE="episode_042"

echo "Creating episode-level wrist rotation config..."
EPISODE_DIR="${DATASET_DIR}/${TISSUE_SAMPLE}/${PHASE}/${EPISODE}"
mkdir -p "${EPISODE_DIR}"  # Create directory if it doesn't exist

cat > "${EPISODE_DIR}/wrist_rotation.json" << EOF
{
  "left_wrist": 90.0,
  "right_wrist": 0
}
EOF

echo "✓ Episode-level config created at: ${EPISODE_DIR}/wrist_rotation.json"
echo ""

# EXAMPLE 3: Batch create configs for multiple episodes
echo "Creating configs for multiple episodes..."

for episode in episode_001 episode_002 episode_003; do
    EPISODE_DIR="${DATASET_DIR}/tissue_7/5_suturing/${episode}"
    mkdir -p "${EPISODE_DIR}"
    
    cat > "${EPISODE_DIR}/wrist_rotation.json" << EOF
{
  "left_wrist": 45.0,
  "right_wrist": -45.0
}
EOF
    echo "✓ Created config for ${episode}"
done

echo ""
echo "Done! Your dataset now has rotation calibration configs."
echo ""
echo "Priority: Episode-level > Task-level > None"
echo "Training will automatically use these rotations."
