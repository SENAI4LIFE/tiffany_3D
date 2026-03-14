# Tiffany — Hexapod Simulation (Gazebo / ROS 2)

Simulation of **Tiffany**, a 3-DOF-per-leg hexapod robot.
Hardware repo: https://github.com/Penguin-Lab/tiffany

---

## Prerequisites

- Ubuntu 24.04 (or WSL2 with Ubuntu 24.04)
- ROS 2 Jazzy — install from https://docs.ros.org/en/jazzy/Installation.html
- Gazebo Sim 8 (ships with `ros-jazzy-ros-gz`)

---

## Install Dependencies

```bash
sudo apt install -y \
  ros-jazzy-ros-gz \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-xacro \
  ros-jazzy-rviz2 \
  ros-jazzy-rqt-image-view \
  python3-colcon-common-extensions \
  python3-pyqt5 \
  python3-catkin-pkg python3-lark python3-empy \
  python3-jinja2 python3-yaml python3-typeguard
```

---

## Environment Setup

Add to `~/.bashrc` and reload:

```bash
source /opt/ros/jazzy/setup.bash
source ~/tiffany_gazebo/install/setup.bash
export GZ_SIM_RESOURCE_PATH=/opt/ros/jazzy/share:~/tiffany_gazebo/install/hexapod_ws/share
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0
```

```bash
source ~/.bashrc
```

---

## Clone

```bash
git clone https://github.com/SENAI4LIFE/tiffany_gazebo.git
cd ~/tiffany_gazebo
```

---

## Build

```bash
colcon build --symlink-install
```

---

## Run

### Terminal 1 — Simulation

```bash
ros2 launch hexapod_ws main.launch.py
```

Controllers load automatically via timed actions (~15 s). Wait for the `hexapod_controller` spawner to confirm before sending commands.

### Terminal 2 — Teleop

```bash
ros2 run hexapod_ws teleop_hexapod.py
```

| Key | Action |
|-----|--------|
| `E` | Boot robot |
| `R` | Shutdown robot |
| `W` / `S` | Walk forward / backward |
| `A` / `D` | Rotate left / right |
| `Q` / `Z` | Strafe left / right |
| `B` | Rebolar |
| `N` | Balance mode |
| `G` / `P` | Patinha |
| Arrow keys | Body pose (roll / pitch) |
| `C` | Turn navigation mode |
| `X` | Omni navigation mode |
| `SPACE` | Stop |

---

## Sensor Visualization

### Camera Feed

```bash
ros2 run rqt_image_view rqt_image_view
```

Select `/camera/image_raw` from the dropdown.

### Lidar in RViz

```bash
ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true
```

In RViz:
1. Set **Fixed Frame** to `lidar_frame`
2. Click **Add → By topic → `/scan` → LaserScan**
3. Click **Add → TF** to visualize joint frames
