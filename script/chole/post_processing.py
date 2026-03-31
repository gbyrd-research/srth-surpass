import streamlit as st
import pandas as pd
import os
import time
import bisect
import cv2
import tempfile
import json
from pathlib import Path
import subprocess
import hashlib
import numpy as np

# MUST be first Streamlit command
st.set_page_config(layout="wide", page_title="Surgical Robot Data Explorer")

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# Try to import streamlit-keyup for keyboard shortcuts
try:
    from streamlit_keyup import st_keyup
    _HAS_KEYUP = True
except ImportError:
    _HAS_KEYUP = False
    # Warning moved to sidebar section below

# --- Optimized Data Loading ---
def get_image_data(folder_path):
    if not os.path.exists(folder_path):
        return [], []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])
    ts = []
    for f in files:
        name_part = f.split('_')[0].replace('frame', '')
        try:
            ts.append(int(name_part))
        except ValueError:
            ts.append(0)
    return files, ts

def find_closest_index(target_ts, ts_list):
    if not ts_list: return None
    idx = bisect.bisect_left(ts_list, target_ts)
    if idx == 0: return 0
    if idx == len(ts_list): return idx - 1
    before, after = ts_list[idx - 1], ts_list[idx]
    return idx if (after - target_ts) < (target_ts - before) else idx - 1

def apply_image_corrections(img, settings):
    """
    settings is a dict that may contain:
      - rotation (degrees)
      - brightness (-100..100)
      - contrast (0.5..2.0)
      - saturation (0..2.0)
      - gamma (0.1..3.0)
      - r_mul, g_mul, b_mul (channel multipliers, 0.5..2.0)
    """
    if img is None:
        return img

    out = img.astype(np.float32)

    # Channel multipliers
    r_mul = float(settings.get('r_mul', 1.0))
    g_mul = float(settings.get('g_mul', 1.0))
    b_mul = float(settings.get('b_mul', 1.0))
    # OpenCV uses BGR
    out[:, :, 0] *= b_mul
    out[:, :, 1] *= g_mul
    out[:, :, 2] *= r_mul

    # Contrast and brightness: out = out * contrast + brightness
    contrast = float(settings.get('contrast', 1.0))
    brightness = float(settings.get('brightness', 0.0))
    out = out * contrast + brightness

    # Saturation: convert to HSV and scale S channel
    sat = float(settings.get('saturation', 1.0))
    if sat != 1.0:
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= sat
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # Gamma correction
    gamma = float(settings.get('gamma', 1.0))
    if gamma != 1.0 and gamma > 0:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype('uint8')
        out = cv2.LUT(np.clip(out, 0, 255).astype(np.uint8), table).astype(np.float32)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def rotate_image(img, angle):
    """Rotate image around center by angle degrees."""
    if img is None:
        return img
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


@st.cache_data
def create_synced_videos(episode_path, left_files, left_ts, psm1_files, psm1_ts, psm2_files, psm2_ts, fps=30, force_rebuild=False):
    temp_dir = tempfile.gettempdir()
    video_paths = {}
    
    # Use 'mp4v' for the initial write (usually works everywhere)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    first_img = cv2.imread(os.path.join(episode_path, "left_img_dir", left_files[0]))
    h, w = first_img.shape[:2]
    
    # Cache directory inside episode so we can reuse generated videos across runs
    cache_dir = os.path.join(episode_path, "cached_videos")
    os.makedirs(cache_dir, exist_ok=True)

    # Temporary raw file paths
    raw_left = os.path.join(temp_dir, "raw_left.mp4")
    raw_psm1 = os.path.join(temp_dir, "raw_psm1.mp4")
    raw_psm2 = os.path.join(temp_dir, "raw_psm2.mp4")

    def write_and_convert(raw_path, files, ts_list, folder_name, role_name):
        if not files: return None
        # Determine role settings (rotation + color corrections) from the task-level wrist_rotation.json (if present)
        try:
            ep_rot = os.path.join(episode_path, 'wrist_rotation.json')
            task_rot = os.path.join(os.path.dirname(episode_path), 'wrist_rotation.json')
            if os.path.exists(ep_rot):
                with open(ep_rot, 'r') as _f:
                    _rots = json.load(_f)
            elif os.path.exists(task_rot):
                with open(task_rot, 'r') as _f:
                    _rots = json.load(_f)
            else:
                _rots = {}
        except Exception:
            _rots = {}

        raw_role_val = _rots.get(role_name)
        if isinstance(raw_role_val, dict):
            role_settings = raw_role_val
        elif raw_role_val is not None:
            # old numeric format -> rotation only
            role_settings = {'rotation': float(raw_role_val)}
        else:
            role_settings = {}

        angle_for_role = int(round(float(role_settings.get('rotation', 0.0))))

        # Include a short hash of the role settings so cache changes when settings change
        cfg_hash = hashlib.md5(json.dumps(role_settings, sort_keys=True).encode()).hexdigest()[:8]
        cached_out = os.path.join(cache_dir, f"{role_name}_f{fps}_cfg{cfg_hash}.mp4")
        if os.path.exists(cached_out) and not force_rebuild:
            return cached_out
        
        # 1. Write the raw video
        img_sample = cv2.imread(os.path.join(episode_path, folder_name, files[0]))
        writer = cv2.VideoWriter(raw_path, fourcc, fps, (img_sample.shape[1], img_sample.shape[0]))

        for master_ts in left_ts:
            idx = find_closest_index(master_ts, ts_list)
            img = cv2.imread(os.path.join(episode_path, folder_name, files[idx]))
            if img is None:
                # skip missing frames
                continue
            # If this is a wrist camera and we have a rotation, apply it before writing
            if role_name in ('psm1', 'psm2') and angle_for_role != 0:
                img = rotate_image(img, float(angle_for_role))

            # Apply color/lighting corrections if any
            if role_name in ('psm1', 'psm2') and role_settings:
                try:
                    img = apply_image_corrections(img, role_settings)
                except Exception:
                    pass

            writer.write(img)
        writer.release()

        # 2. Convert to H.264 using FFmpeg (Streamlit compatible)
        out_path = cached_out
        # -y overwrites, -c:v libx264 is the web standard
        cmd = f"ffmpeg -y -i {raw_path} -c:v libx264 -pix_fmt yuv420p -crf 23 {out_path}"
        subprocess.run(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        return out_path

    with st.spinner("Processing videos with FFmpeg (or reusing cached files)..."):
        video_paths['left'] = write_and_convert(raw_left, left_files, left_ts, "left_img_dir", 'left')
        video_paths['psm1'] = write_and_convert(raw_psm1, psm1_files, psm1_ts, "endo_psm1", 'psm1')
        video_paths['psm2'] = write_and_convert(raw_psm2, psm2_files, psm2_ts, "endo_psm2", 'psm2')

    return video_paths

# --- Sidebar: Navigation ---
st.sidebar.title("📁 Dataset Browser")
base_path = st.sidebar.text_input("Dataset Base Path", value="/home/iulian/chole_ws/data/cnh_exvivo_chole")

if not os.path.exists(base_path):
    st.error("Invalid Path")
    st.stop()

def get_subdirs(p):
    return sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])

subsets = get_subdirs(base_path)
if not subsets: st.stop()
selected_subset = st.sidebar.selectbox("Subset", subsets)

subset_path = os.path.join(base_path, selected_subset)
tasks = get_subdirs(subset_path)
if not tasks: st.stop()
selected_task = st.sidebar.selectbox("Task", tasks)

task_path = os.path.join(subset_path, selected_task)
episodes = get_subdirs(task_path)
if not episodes: st.stop()

# Analyze episodes for outliers (significantly different frame counts)
episode_frame_counts = {}
for ep in episodes:
    ep_path = os.path.join(task_path, ep)
    left_ep, _ = get_image_data(os.path.join(ep_path, "left_img_dir"))
    episode_frame_counts[ep] = len(left_ep)

# Detect outliers using IQR method (frames significantly different from median)
frame_counts = list(episode_frame_counts.values())
if len(frame_counts) > 3:
    q1 = np.percentile(frame_counts, 25)
    q3 = np.percentile(frame_counts, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_episodes = {ep for ep, count in episode_frame_counts.items() if count < lower_bound or count > upper_bound}
else:
    outlier_episodes = set()

# Load recovery episodes list
def get_recovery_json_path_for_task(tpath):
    return os.path.join(tpath, 'recovery_episodes.json')

def load_recovery_episodes_for_task(tpath):
    recovery_json = get_recovery_json_path_for_task(tpath)
    if os.path.exists(recovery_json):
        try:
            with open(recovery_json, 'r') as f:
                data = json.load(f)
                return data.get('recovery_episodes', [])
        except Exception:
            return []
    return []

recovery_episodes_list = load_recovery_episodes_for_task(task_path)

# Display warning for outliers in sidebar
if outlier_episodes:
    st.sidebar.warning(f"⚠️ {len(outlier_episodes)} episode(s) with unusual frame counts detected (may need splitting)")

# Initialize session state for episode navigation
if 'current_episode_index' not in st.session_state:
    st.session_state.current_episode_index = 0
if 'keyboard_trigger' not in st.session_state:
    st.session_state.keyboard_trigger = 0

# Show warning about missing library if needed
if not _HAS_KEYUP:
    st.sidebar.warning("Install streamlit-keyup for keyboard shortcuts: pip install streamlit-keyup")

# Keyboard shortcut handler (invisible text input that captures key presses)
if _HAS_KEYUP:
    with st.sidebar:
        key_pressed = st_keyup("", key="keyboard_nav", placeholder="Press N/P/R for shortcuts")
        
        if key_pressed:
            key_lower = key_pressed.lower()
            if key_lower in ['n', 'arrowright']:
                if st.session_state.current_episode_index < len(episodes) - 1:
                    st.session_state.current_episode_index += 1
                    st.session_state.keyboard_trigger += 1
                    st.rerun()
            elif key_lower in ['p', 'arrowleft']:
                if st.session_state.current_episode_index > 0:
                    st.session_state.current_episode_index -= 1
                    st.session_state.keyboard_trigger += 1
                    st.rerun()
            elif key_lower == 'r':
                st.session_state.keyboard_trigger += 1
                # Recovery toggle will be handled below
else:
    # Fallback: Use streamlit components for simple keyboard handling
    st.sidebar.markdown("""
    <div id="keyboard-handler" tabindex="0" style="position: fixed; opacity: 0; pointer-events: none;"></div>
    <script>
    const handler = window.parent.document.getElementById('keyboard-handler');
    if (handler) {
        handler.focus();
        handler.addEventListener('keydown', function(e) {
            if (e.key === 'n' || e.key === 'ArrowRight' || e.key === 'p' || e.key === 'ArrowLeft' || e.key === 'r') {
                const buttons = window.parent.document.querySelectorAll('button');
                if (e.key === 'n' || e.key === 'ArrowRight') {
                    buttons.forEach(btn => {
                        if (btn.innerText.includes('Next Episode')) btn.click();
                    });
                } else if (e.key === 'p' || e.key === 'ArrowLeft') {
                    buttons.forEach(btn => {
                        if (btn.innerText.includes('Previous Episode')) btn.click();
                    });
                } else if (e.key === 'r') {
                    buttons.forEach(btn => {
                        if (btn.innerText.includes('Mark as Recovery') || btn.innerText.includes('Remove Recovery Mark')) {
                            btn.click();
                            return;
                        }
                    });
                }
                e.preventDefault();
            }
        });
    }
    </script>
    """, unsafe_allow_html=True)

# --- Episode navigation buttons
st.sidebar.markdown("---")
if _HAS_KEYUP:
    st.sidebar.markdown("**Navigation** (N/P or ←/→ keys)")
else:
    st.sidebar.markdown("**Navigation** (use buttons)")

# Display episode counter
total_episodes = len(episodes)
current_position = st.session_state.current_episode_index + 1
st.sidebar.markdown(f"**Episode {current_position}/{total_episodes}**")

nav_col1, nav_col2 = st.sidebar.columns(2)

with nav_col1:
    if st.button("⬅️ Previous Episode", key="prev_episode", disabled=(st.session_state.current_episode_index == 0), type="secondary"):
        st.session_state.current_episode_index = max(0, st.session_state.current_episode_index - 1)
        st.rerun()

with nav_col2:
    if st.button("Next Episode ➡️", key="next_episode", disabled=(st.session_state.current_episode_index >= len(episodes) - 1), type="secondary"):
        st.session_state.current_episode_index = min(len(episodes) - 1, st.session_state.current_episode_index + 1)
        st.rerun()

st.sidebar.caption("Hotkeys: N/P or ←/→ = navigate | R = toggle recovery" if _HAS_KEYUP else "Install streamlit-keyup for keyboard shortcuts")

# Custom format function for episode dropdown with highlighting
def format_episode_option(ep, idx):
    frame_count = episode_frame_counts[ep]
    is_current = (idx == st.session_state.current_episode_index)
    is_recovery = ep in recovery_episodes_list
    is_outlier = ep in outlier_episodes
    
    prefix = ""
    if is_current:
        prefix = "➤ "  # Current episode marker
    if is_outlier:
        prefix += "🔴 "
    if is_recovery:
        prefix += "♻️ "
    
    return f"{prefix}{ep} ({frame_count} frames)"

# Create episode options with indices for tracking
episode_options = list(enumerate(episodes))
    
selected_episode_tuple = st.sidebar.selectbox(
    "Episode", 
    episode_options,
    index=st.session_state.current_episode_index,
    format_func=lambda x: format_episode_option(x[1], x[0]),
    key="episode_selector"
)

# Update session state if user manually changed episode
st.session_state.current_episode_index = selected_episode_tuple[0]
selected_episode = selected_episode_tuple[1]

episode_path = os.path.join(task_path, selected_episode)

# --- Wrist rotation correction helper functions (defined early for use in previews) ---
def rotation_task_path():
    return os.path.join(task_path, 'wrist_rotation.json')

def rotation_episode_path():
    return os.path.join(episode_path, 'wrist_rotation.json')

def load_rotation_angles():
    """Load episode-specific settings if present, otherwise fall back to task defaults."""
    ep = rotation_episode_path()
    taskp = rotation_task_path()
    if os.path.exists(ep):
        try:
            with open(ep, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    if os.path.exists(taskp):
        try:
            with open(taskp, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_rotation_angles(angles_dict, target='episode'):
    """Save angles_dict to episode or task level. target in {'episode','task'}"""
    if target == 'task':
        p = rotation_task_path()
    else:
        p = rotation_episode_path()
    try:
        with open(p, 'w') as f:
            json.dump(angles_dict, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save rotation angles: {e}")
        return False

# --- Load All Data Streams ---
left_files, left_ts = get_image_data(os.path.join(episode_path, "left_img_dir"))
right_files, right_ts = get_image_data(os.path.join(episode_path, "right_img_dir"))
psm1_files, psm1_ts = get_image_data(os.path.join(episode_path, "endo_psm1"))
psm2_files, psm2_ts = get_image_data(os.path.join(episode_path, "endo_psm2"))

if not left_files:
    st.error("No images found in left_img_dir!")
    st.stop()

@st.cache_data
def load_kinematics(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    return df

try:
    df_kin = load_kinematics(os.path.join(episode_path, "ee_csv.csv"))
    kin_ts = df_kin['timestamp'].tolist()
except:
    df_kin, kin_ts = None, []

# --- Video Generation Control ---
st.sidebar.divider()
st.sidebar.markdown("**Video Generation**")
generate_videos = st.sidebar.checkbox("Generate videos", value=False, help="Check to generate synced videos. Uncheck for faster loading with preview images only.")

if generate_videos:
    # --- Create Videos ---
    fps = st.sidebar.number_input("Video FPS", min_value=1, max_value=60, value=30)
    # Option to force regeneration of cached videos
    force_regen = st.sidebar.checkbox("Regenerate cached videos", value=False, help="If checked, videos will be re-created even if cached copies exist in the episode folder")
    video_paths = create_synced_videos(episode_path, left_files, left_ts, psm1_files, psm1_ts, psm2_files, psm2_ts, fps, force_rebuild=force_regen)
else:
    video_paths = None

# --- Display Videos or Preview Images ---
st.divider()
v1, v2, v3 = st.columns(3)

with v1:
    st.subheader("PSM1 Wrist")
    if generate_videos and video_paths:
        if video_paths['psm1'] and os.path.exists(video_paths['psm1']):
            st.video(video_paths['psm1'])
        elif video_paths['psm1']:
            st.warning("Video file not found. Check 'Regenerate cached videos' to rebuild.")
        else:
            st.info("No PSM1 images available")
    else:
        # Show first frame preview
        if psm1_files:
            preview_path = os.path.join(episode_path, 'endo_psm1', psm1_files[0])
            preview_img = cv2.imread(preview_path)
            if preview_img is not None:
                # Apply rotation if saved
                rotation_angles_preview = load_rotation_angles()
                raw = rotation_angles_preview.get('psm1')
                if isinstance(raw, dict):
                    angle = float(raw.get('rotation', 0.0))
                    if angle != 0:
                        preview_img = rotate_image(preview_img, angle)
                    preview_img = apply_image_corrections(preview_img, raw)
                elif raw is not None:
                    angle = float(raw)
                    if angle != 0:
                        preview_img = rotate_image(preview_img, angle)
                st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), caption='PSM1 First Frame Preview', use_container_width=True)
            else:
                st.warning("Could not load preview image")
        else:
            st.info("No PSM1 images available")

with v2:
    st.subheader("Endo Left")
    if generate_videos and video_paths:
        if video_paths['left'] and os.path.exists(video_paths['left']):
            st.video(video_paths['left'])
        elif video_paths['left']:
            st.warning("Video file not found. Check 'Regenerate cached videos' to rebuild.")
        else:
            st.info("No left images available")
    else:
        # Show first frame preview
        if left_files:
            preview_path = os.path.join(episode_path, 'left_img_dir', left_files[0])
            preview_img = cv2.imread(preview_path)
            if preview_img is not None:
                st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), caption='Left Camera First Frame Preview', use_container_width=True)
            else:
                st.warning("Could not load preview image")
        else:
            st.info("No left images available")

with v3:
    st.subheader("PSM2 Wrist")
    if generate_videos and video_paths:
        if video_paths['psm2'] and os.path.exists(video_paths['psm2']):
            st.video(video_paths['psm2'])
        elif video_paths['psm2']:
            st.warning("Video file not found. Check 'Regenerate cached videos' to rebuild.")
        else:
            st.info("No PSM2 images available")
    else:
        # Show first frame preview
        if psm2_files:
            preview_path = os.path.join(episode_path, 'endo_psm2', psm2_files[0])
            preview_img = cv2.imread(preview_path)
            if preview_img is not None:
                # Apply rotation if saved
                rotation_angles_preview = load_rotation_angles()
                raw = rotation_angles_preview.get('psm2')
                if isinstance(raw, dict):
                    angle = float(raw.get('rotation', 0.0))
                    if angle != 0:
                        preview_img = rotate_image(preview_img, angle)
                    preview_img = apply_image_corrections(preview_img, raw)
                elif raw is not None:
                    angle = float(raw)
                    if angle != 0:
                        preview_img = rotate_image(preview_img, angle)
                st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), caption='PSM2 First Frame Preview', use_container_width=True)
            else:
                st.warning("Could not load preview image")
        else:
            st.info("No PSM2 images available")

if not generate_videos:
    st.info("💡 Check 'Generate videos' in the sidebar to create synced videos for playback.")

# --- Recovery Episode Marking ---
st.divider()
st.subheader("Mark as Recovery Episode")
st.markdown("Mark this episode as a recovery demonstration (e.g., error correction, retry after failure). **Press 'R' as hotkey.**")

def get_recovery_json_path():
    """Path to recovery episodes JSON file in the task folder"""
    return os.path.join(task_path, 'recovery_episodes.json')

def load_recovery_episodes():
    """Load list of recovery episodes from task folder"""
    recovery_json = get_recovery_json_path()
    if os.path.exists(recovery_json):
        try:
            with open(recovery_json, 'r') as f:
                data = json.load(f)
                return data.get('recovery_episodes', [])
        except Exception:
            return []
    return []

def save_recovery_episodes(episodes_list):
    """Save list of recovery episodes to task folder"""
    recovery_json = get_recovery_json_path()
    try:
        data = {
            'task': selected_task,
            'recovery_episodes': sorted(list(set(episodes_list)))  # remove duplicates and sort
        }
        with open(recovery_json, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save recovery episodes: {e}")
        return False

# Load current recovery episodes
recovery_episodes = load_recovery_episodes()
is_recovery = selected_episode in recovery_episodes

# Display current status
if is_recovery:
    st.info(f"♻️ Episode '{selected_episode}' is marked as a recovery episode")
else:
    st.info(f"Episode '{selected_episode}' is NOT marked as a recovery episode")

# Buttons to mark/unmark
col_rec1, col_rec2 = st.columns(2)
with col_rec1:
    if st.button("🔴 Mark as Recovery Episode (R)", disabled=is_recovery, key=f"mark_recovery_{selected_task}_{selected_episode}") or (_HAS_KEYUP and key_pressed and key_pressed.lower() == 'r' and not is_recovery):
        if selected_episode not in recovery_episodes:
            recovery_episodes.append(selected_episode)
            if save_recovery_episodes(recovery_episodes):
                st.success(f"Marked '{selected_episode}' as recovery episode")
                st.rerun()

with col_rec2:
    if st.button("✓ Remove Recovery Mark (R)", disabled=not is_recovery, key=f"unmark_recovery_{selected_task}_{selected_episode}") or (_HAS_KEYUP and key_pressed and key_pressed.lower() == 'r' and is_recovery):
        if selected_episode in recovery_episodes:
            recovery_episodes.remove(selected_episode)
            if save_recovery_episodes(recovery_episodes):
                st.success(f"Removed recovery mark from '{selected_episode}'")
                st.rerun()

# Show all recovery episodes in current task
if recovery_episodes:
    with st.expander(f"View all recovery episodes in '{selected_task}' ({len(recovery_episodes)} total)"):
        for ep in sorted(recovery_episodes):
            st.text(f"• {ep}")
else:
    st.caption("No recovery episodes marked yet in this task.")

# --- Wrist rotation correction UI ---
# (Functions already defined above, just keep the UI section)

# Load any previously saved rotation angles
rotation_angles = load_rotation_angles()

st.divider()
st.subheader("Wrist Image Rotation Correction")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**PSM1 Wrist Rotation**")
    if psm1_files:
        sample_path = os.path.join(episode_path, 'endo_psm1', psm1_files[0])
        img = cv2.imread(sample_path)
        if img is None:
            st.warning("Unable to load sample image for PSM1")
        else:
            raw = rotation_angles.get('psm1')
            if isinstance(raw, dict):
                default_angle = float(raw.get('rotation', 0.0))
                default_brightness = float(raw.get('brightness', 0.0))
                default_contrast = float(raw.get('contrast', 1.0))
                default_saturation = float(raw.get('saturation', 1.0))
                default_gamma = float(raw.get('gamma', 1.0))
                default_r = float(raw.get('r_mul', 1.0))
                default_g = float(raw.get('g_mul', 1.0))
                default_b = float(raw.get('b_mul', 1.0))
            else:
                default_angle = float(raw) if raw is not None else 0.0
                default_brightness = 0.0
                default_contrast = 1.0
                default_saturation = 1.0
                default_gamma = 1.0
                default_r = default_g = default_b = 1.0

            angle = st.slider('Rotate PSM1 (degrees)', min_value=-180.0, max_value=180.0, value=default_angle, step=1.0, key=f"psm1_angle_{selected_task}_{selected_episode}")

            brightness = st.slider('Brightness', min_value=-100, max_value=100, value=int(default_brightness), step=1, key=f"psm1_brightness_{selected_task}_{selected_episode}")
            contrast = st.slider('Contrast', min_value=0.5, max_value=2.0, value=float(default_contrast), step=0.01, key=f"psm1_contrast_{selected_task}_{selected_episode}")
            saturation = st.slider('Saturation', min_value=0.0, max_value=2.0, value=float(default_saturation), step=0.01, key=f"psm1_saturation_{selected_task}_{selected_episode}")
            gamma = st.slider('Gamma', min_value=0.1, max_value=3.0, value=float(default_gamma), step=0.01, key=f"psm1_gamma_{selected_task}_{selected_episode}")
            r_mul = st.slider('R multiplier', min_value=0.5, max_value=2.0, value=float(default_r), step=0.01, key=f"psm1_rmul_{selected_task}_{selected_episode}")
            g_mul = st.slider('G multiplier', min_value=0.5, max_value=2.0, value=float(default_g), step=0.01, key=f"psm1_gmul_{selected_task}_{selected_episode}")
            b_mul = st.slider('B multiplier', min_value=0.5, max_value=2.0, value=float(default_b), step=0.01, key=f"psm1_bmul_{selected_task}_{selected_episode}")

            preview = rotate_image(img, angle)
            settings = {
                'rotation': float(angle),
                'brightness': float(brightness),
                'contrast': float(contrast),
                'saturation': float(saturation),
                'gamma': float(gamma),
                'r_mul': float(r_mul),
                'g_mul': float(g_mul),
                'b_mul': float(b_mul),
            }
            preview = apply_image_corrections(preview, settings)
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption=f'PSM1 preview (rot {angle}°, brightness {brightness})', use_container_width=True)
            save_target_psm1 = st.radio('Save PSM1 settings to', options=['episode', 'task'], index=0, horizontal=True, key=f"psm1_save_target_{selected_task}_{selected_episode}")
            
            col1a, col1b = st.columns(2)
            with col1a:
                if st.button('Confirm PSM1 settings', key=f"confirm_psm1_{selected_task}_{selected_episode}"):
                    rotation_angles['psm1'] = settings
                    if save_rotation_angles(rotation_angles, target=save_target_psm1):
                        st.success(f"Saved PSM1 settings to {save_target_psm1}")
            
            with col1b:
                if st.button('Apply PSM1 to all tasks in tissue', key=f"apply_psm1_all_tasks_{selected_task}_{selected_episode}"):
                    # Apply to all tasks in the current subset (tissue)
                    tasks_in_tissue = get_subdirs(subset_path)
                    success_count = 0
                    for task_name in tasks_in_tissue:
                        task_rot_path = os.path.join(subset_path, task_name, 'wrist_rotation.json')
                        try:
                            # Load existing settings for this task or start fresh
                            if os.path.exists(task_rot_path):
                                with open(task_rot_path, 'r') as f:
                                    task_rots = json.load(f)
                            else:
                                task_rots = {}
                            
                            # Update PSM1 settings
                            task_rots['psm1'] = settings
                            
                            # Save back
                            with open(task_rot_path, 'w') as f:
                                json.dump(task_rots, f, indent=2)
                            success_count += 1
                        except Exception:
                            pass
                    st.success(f"Applied PSM1 settings to {success_count}/{len(tasks_in_tissue)} tasks in '{selected_subset}'")

with col2:
    st.markdown("**PSM2 Wrist Rotation**")
    if psm2_files:
        sample_path = os.path.join(episode_path, 'endo_psm2', psm2_files[0])
        img2 = cv2.imread(sample_path)
        if img2 is None:
            st.warning("Unable to load sample image for PSM2")
        else:
            raw2 = rotation_angles.get('psm2')
            if isinstance(raw2, dict):
                default_angle2 = float(raw2.get('rotation', 0.0))
                default_brightness2 = float(raw2.get('brightness', 0.0))
                default_contrast2 = float(raw2.get('contrast', 1.0))
                default_saturation2 = float(raw2.get('saturation', 1.0))
                default_gamma2 = float(raw2.get('gamma', 1.0))
                default_r2 = float(raw2.get('r_mul', 1.0))
                default_g2 = float(raw2.get('g_mul', 1.0))
                default_b2 = float(raw2.get('b_mul', 1.0))
            else:
                default_angle2 = float(raw2) if raw2 is not None else 0.0
                default_brightness2 = 0.0
                default_contrast2 = 1.0
                default_saturation2 = 1.0
                default_gamma2 = 1.0
                default_r2 = default_g2 = default_b2 = 1.0

            angle2 = st.slider('Rotate PSM2 (degrees)', min_value=-180.0, max_value=180.0, value=default_angle2, step=1.0, key=f"psm2_angle_{selected_task}_{selected_episode}")
            brightness2 = st.slider('Brightness', min_value=-100, max_value=100, value=int(default_brightness2), step=1, key=f"psm2_brightness_{selected_task}_{selected_episode}")
            contrast2 = st.slider('Contrast', min_value=0.5, max_value=2.0, value=float(default_contrast2), step=0.01, key=f"psm2_contrast_{selected_task}_{selected_episode}")
            saturation2 = st.slider('Saturation', min_value=0.0, max_value=2.0, value=float(default_saturation2), step=0.01, key=f"psm2_saturation_{selected_task}_{selected_episode}")
            gamma2 = st.slider('Gamma', min_value=0.1, max_value=3.0, value=float(default_gamma2), step=0.01, key=f"psm2_gamma_{selected_task}_{selected_episode}")
            r_mul2 = st.slider('R multiplier', min_value=0.5, max_value=2.0, value=float(default_r2), step=0.01, key=f"psm2_rmul_{selected_task}_{selected_episode}")
            g_mul2 = st.slider('G multiplier', min_value=0.5, max_value=2.0, value=float(default_g2), step=0.01, key=f"psm2_gmul_{selected_task}_{selected_episode}")
            b_mul2 = st.slider('B multiplier', min_value=0.5, max_value=2.0, value=float(default_b2), step=0.01, key=f"psm2_bmul_{selected_task}_{selected_episode}")

            preview2 = rotate_image(img2, angle2)
            settings2 = {
                'rotation': float(angle2),
                'brightness': float(brightness2),
                'contrast': float(contrast2),
                'saturation': float(saturation2),
                'gamma': float(gamma2),
                'r_mul': float(r_mul2),
                'g_mul': float(g_mul2),
                'b_mul': float(b_mul2),
            }
            preview2 = apply_image_corrections(preview2, settings2)
            st.image(cv2.cvtColor(preview2, cv2.COLOR_BGR2RGB), caption=f'PSM2 preview (rot {angle2}°, brightness {brightness2})', use_container_width=True)
            save_target_psm2 = st.radio('Save PSM2 settings to', options=['episode', 'task'], index=0, horizontal=True, key=f"psm2_save_target_{selected_task}_{selected_episode}")
            
            col2a, col2b = st.columns(2)
            with col2a:
                if st.button('Confirm PSM2 settings', key=f"confirm_psm2_{selected_task}_{selected_episode}"):
                    rotation_angles['psm2'] = settings2
                    if save_rotation_angles(rotation_angles, target=save_target_psm2):
                        st.success(f"Saved PSM2 settings to {save_target_psm2}")
            
            with col2b:
                if st.button('Apply PSM2 to all tasks in tissue', key=f"apply_psm2_all_tasks_{selected_task}_{selected_episode}"):
                    # Apply to all tasks in the current subset (tissue)
                    tasks_in_tissue = get_subdirs(subset_path)
                    success_count = 0
                    for task_name in tasks_in_tissue:
                        task_rot_path = os.path.join(subset_path, task_name, 'wrist_rotation.json')
                        try:
                            # Load existing settings for this task or start fresh
                            if os.path.exists(task_rot_path):
                                with open(task_rot_path, 'r') as f:
                                    task_rots = json.load(f)
                            else:
                                task_rots = {}
                            
                            # Update PSM2 settings
                            task_rots['psm2'] = settings2
                            
                            # Save back
                            with open(task_rot_path, 'w') as f:
                                json.dump(task_rots, f, indent=2)
                            success_count += 1
                        except Exception:
                            pass
                    st.success(f"Applied PSM2 settings to {success_count}/{len(tasks_in_tissue)} tasks in '{selected_subset}'")

if rotation_angles:
    st.info(f"Current saved rotations: {rotation_angles}")

# --- Episode Splitting ---
st.divider()
st.subheader("Split Episode")
st.markdown("Create a new episode from a time range of the current episode (useful when one recording contains multiple demonstrations).")

if left_files and df_kin is not None:
    # Use same timestamp scaling as 3D visualization for consistency
    ts_series = pd.to_numeric(df_kin['timestamp'], errors='coerce').dropna().astype(float)
    if not ts_series.empty:
        JS_MAX = (1 << 53) - 1
        max_ts = float(ts_series.max())
        for s in [1.0, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18]:
            if max_ts / s <= JS_MAX:
                scale = s
                break
        else:
            scale = 1.0

        scaled_ts = ts_series / scale
        if scale >= 1e9:
            unit = 'seconds (ts/1e9)'
        elif scale >= 1e6:
            unit = 'milliseconds (ts/1e6)'
        elif scale >= 1e3:
            unit = 'microseconds (ts/1e3)'
        else:
            unit = 'raw'

        min_val = float(scaled_ts.min())
        max_val = float(scaled_ts.max())
        split_range = st.slider(f"Select time range for new episode [{unit}]", min_value=min_val, max_value=max_val, value=(min_val, max_val), format="%.3f", key=f"split_range_{selected_task}_{selected_episode}")

        low = float(split_range[0]) * scale
        high = float(split_range[1]) * scale
        ts_numeric = pd.to_numeric(df_kin['timestamp'], errors='coerce').astype(float)
        mask = (ts_numeric >= low) & (ts_numeric <= high)
        df_split = df_kin[mask.fillna(False)]

        st.write(f"Selected range contains {len(df_split)} kinematics rows")

        new_episode_name = st.text_input("New episode name", value=f"{selected_episode}_split", key=f"new_ep_name_{selected_task}_{selected_episode}")

        if st.button("Create split episode", key=f"create_split_{selected_task}_{selected_episode}"):
            import shutil
            new_episode_path = os.path.join(task_path, new_episode_name)
            if os.path.exists(new_episode_path):
                st.error(f"Episode '{new_episode_name}' already exists!")
            else:
                try:
                    os.makedirs(new_episode_path, exist_ok=True)

                    # Copy images within timestamp range for each camera folder
                    for folder_name, files_list, ts_list in [
                        ('left_img_dir', left_files, left_ts),
                        ('right_img_dir', right_files, right_ts),
                        ('endo_psm1', psm1_files, psm1_ts),
                        ('endo_psm2', psm2_files, psm2_ts)
                    ]:
                        src_folder = os.path.join(episode_path, folder_name)
                        dst_folder = os.path.join(new_episode_path, folder_name)
                        os.makedirs(dst_folder, exist_ok=True)

                        for ts_val, fname in zip(ts_list, files_list):
                            if low <= ts_val <= high:
                                shutil.copy2(os.path.join(src_folder, fname), os.path.join(dst_folder, fname))

                    # Save filtered kinematics
                    df_split.to_csv(os.path.join(new_episode_path, 'ee_csv.csv'), index=False)

                    # Copy wrist_rotation.json if exists at episode level
                    ep_rot = rotation_episode_path()
                    if os.path.exists(ep_rot):
                        shutil.copy2(ep_rot, os.path.join(new_episode_path, 'wrist_rotation.json'))

                    st.success(f"Created new episode '{new_episode_name}' with {len(df_split)} kinematics rows")
                except Exception as e:
                    st.error(f"Failed to create split episode: {e}")
    else:
        st.warning("No valid timestamps found in kinematics.")
else:
    st.info("Episode splitting requires images and kinematics data.")

# --- Kinematics Display ---
st.divider()
st.subheader("Kinematics Data")
if df_kin is not None:
    with st.expander("Show Full Kinematics"):
        st.dataframe(df_kin)

# --- 3D Kinematics Visualization ---
st.divider()
st.subheader("3D Kinematics Visualization")
def find_axis_col(df, arm, axis):
    # common patterns we try to match
    patterns = [
        f"{arm}_pose.position.{axis}",
        f"{arm}.pose.position.{axis}",
        f"{arm}_{axis}",
        f"{arm}.{axis}",
        f"{arm}_position_{axis}",
        f"{arm}_pose_position_{axis}",
    ]
    for p in patterns:
        if p in df.columns:
            return p
    # fuzzy match: contains arm, position and axis
    for col in df.columns:
        low = col.lower()
        if arm in low and 'position' in low and axis in low:
            return col
    # fallback: endswith axis and contains arm
    for col in df.columns:
        if col.lower().endswith(axis) and arm in col.lower():
            return col
    return None

def get_xyz_for_arm(df, arm):
    x = find_axis_col(df, arm, 'x')
    y = find_axis_col(df, arm, 'y')
    z = find_axis_col(df, arm, 'z')
    if x and y and z:
        return x, y, z
    return None, None, None

if df_kin is None:
    st.info("No kinematics (ee_csv.csv) found for this episode.")
else:
    # detect columns for psm1 and psm2
    p1x, p1y, p1z = get_xyz_for_arm(df_kin, 'psm1')
    p2x, p2y, p2z = get_xyz_for_arm(df_kin, 'psm2')

    if not any([p1x, p2x]):
        st.warning("Could not detect psm1/psm2 position columns automatically. Looking for columns containing 'psm1'/'psm2' and 'position.x/y/z'.")

    show_3d = st.checkbox("Show 3D kinematics plot", value=True)
    if show_3d:
        arms_to_plot = st.multiselect("Arms to plot", options=['psm1', 'psm2'], default=['psm1', 'psm2'])

        # time range selector (scale timestamps to fit JS numeric limits)
        if 'timestamp' in df_kin.columns:
            ts_series = pd.to_numeric(df_kin['timestamp'], errors='coerce').dropna().astype(float)
            if ts_series.empty:
                df_plot = df_kin
            else:
                JS_MAX = (1 << 53) - 1
                max_ts = float(ts_series.max())
                # choose a scale (divisor) so that max_ts/scale <= JS_MAX
                for s in [1.0, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18]:
                    if max_ts / s <= JS_MAX:
                        scale = s
                        break
                else:
                    scale = 1.0

                scaled_ts = ts_series / scale
                # friendly unit string for common scales
                if scale >= 1e9:
                    unit = 'seconds (ts/1e9)'
                elif scale >= 1e6:
                    unit = 'milliseconds (ts/1e6)'
                elif scale >= 1e3:
                    unit = 'microseconds (ts/1e3)'
                else:
                    unit = 'raw'

                min_val = float(scaled_ts.min())
                max_val = float(scaled_ts.max())
                trange_scaled = st.slider(f"Timestamp range [{unit}]", min_value=min_val, max_value=max_val, value=(min_val, max_val), format="%.3f")

                low = float(trange_scaled[0]) * scale
                high = float(trange_scaled[1]) * scale
                ts_numeric = pd.to_numeric(df_kin['timestamp'], errors='coerce').astype(float)
                mask = (ts_numeric >= low) & (ts_numeric <= high)
                df_plot = df_kin[mask.fillna(False)]
        else:
            df_plot = df_kin

        if df_plot.empty:
            st.info("No kinematics rows in selected time range.")
        else:
            # Build plot using Plotly if available, otherwise Matplotlib
            if _HAS_PLOTLY:
                fig = go.Figure()
                if 'psm1' in arms_to_plot and p1x and p1y and p1z:
                    fig.add_trace(go.Scatter3d(x=df_plot[p1x], y=df_plot[p1y], z=df_plot[p1z], mode='lines+markers', name='psm1', marker=dict(size=2)))
                if 'psm2' in arms_to_plot and p2x and p2y and p2z:
                    fig.add_trace(go.Scatter3d(x=df_plot[p2x], y=df_plot[p2y], z=df_plot[p2z], mode='lines+markers', name='psm2', marker=dict(size=2)))
                fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), height=700)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # fallback to matplotlib
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                if 'psm1' in arms_to_plot and p1x and p1y and p1z:
                    ax.plot(df_plot[p1x].values, df_plot[p1y].values, df_plot[p1z].values, label='psm1')
                    ax.scatter(df_plot[p1x].values[0], df_plot[p1y].values[0], df_plot[p1z].values[0], color='green', s=20, label='psm1 start')
                if 'psm2' in arms_to_plot and p2x and p2y and p2z:
                    ax.plot(df_plot[p2x].values, df_plot[p2y].values, df_plot[p2z].values, label='psm2')
                    ax.scatter(df_plot[p2x].values[0], df_plot[p2y].values[0], df_plot[p2z].values[0], color='orange', s=20, label='psm2 start')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.legend()
                st.pyplot(fig)