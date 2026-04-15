Create a conda environment with ros1 noetic installed:

```bash
conda create -n srth-old -c conda-forge -c robostack-noetic \
    python=3.11 \
    ros-noetic-desktop
conda activate ros_env
conda config --env --add channels robostack-noetic
mamba install -c conda-forge ros-dev-tools \
    ros-noetic-actionlib \
    ros-noetic-camera-calibration \
    ros-noetic-camera-calibration-parsers \
    ros-noetic-catkin \
    ros-noetic-controller-manager \
    ros-noetic-controller-manager-msgs \
    ros-noetic-cv-bridge \
    ros-noetic-dynamic-reconfigure \
    ros-noetic-genmsg \
    ros-noetic-genpy \
    ros-noetic-gencpp \
    ros-noetic-geneus \
    ros-noetic-genlisp \
    ros-noetic-gennodejs \
    ros-noetic-diagnostic-updater \
    ros-noetic-diagnostic-analysis \
    ros-noetic-diagnostic-common-diagnostics \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-plugins \
    ros-noetic-image-geometry \
    ros-noetic-laser-geometry \
    ros-noetic-message-filters \
    ros-noetic-interactive-markers \
    ros-noetic-joint-state-publisher \
    ros-noetic-joint-state-publisher-gui \
    breezy
```

*NOTE:* You first must downgrade setuptools (necessary for installing some older dependencies):

```bash
python -m pip install "setuptools<82"
```

Install software dependencies:

```bash
pip install --no-build-isolation -r requirements_ll.txt
```