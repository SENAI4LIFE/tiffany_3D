# Tiffany — Hexapod Simulation (Gazebo / ROS 2)
Simulation of **Tiffany**, a 3-DOF-per-leg hexapod robot.
Hardware repo: https://github.com/Penguin-Lab/tiffany
---
## Prerequisites
- Ubuntu 24.04 (or WSL2 with Ubuntu 24.04)
- ROS 2 Jazzy — install from https://docs.ros.org/en/jazzy/Installation.html
- Gazebo Sim 8 (ships with `ros-jazzy-ros-gz`)
---
## Setup
**1. Clone**
```bash
git clone https://github.com/SENAI4LIFE/tiffany_gazebo.git
```
**2. Navigate to workspace**
```bash
cd ~/tiffany_gazebo
```
**3. Install Dependencies**
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
**4. Source ROS**
```bash
source /opt/ros/jazzy/setup.bash
```
**5. Make scripts executable**
```bash
chmod +x src/hexapod_ws/scripts/hexapod_runner.py
chmod +x src/hexapod_ws/scripts/teleop_hexapod.py
```
**6. Build**
```bash
colcon build --symlink-install
```
**7. Environment Setup**
```bash
source setup.bash
```
---
## Run
### Terminal 1 — Simulation
```bash
cd ~/tiffany_gazebo
source setup.bash
ros2 launch hexapod_ws main.launch.py
```
Controllers load automatically via timed actions (~15 s). Wait for the `hexapod_controller` spawner to confirm before sending commands.
### Terminal 2 — Teleop
```bash
cd ~/tiffany_gazebo
source setup.bash
ros2 run hexapod_ws teleop_hexapod.py
```
### Terminal 3 — Camera Feed
```bash
source /opt/ros/jazzy/setup.bash
ros2 run rqt_image_view rqt_image_view
```
Select `/camera/image_raw` from the dropdown.
### Terminal 4 — Lidar in RViz
```bash
source /opt/ros/jazzy/setup.bash
ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true
```
In RViz:
1. Set **Fixed Frame** to `lidar`
2. Click **Add → By topic → `/scan` → LaserScan**
3. Click **Add → TF** to visualize joint frames
---
## Controls
| Key | Action |
|-----|--------|
| `E` | Boot robot |
| `Q` | Shutdown robot |
| `W` / `S` | Walk forward / backward |
| `A` / `D` | Rotate left / right |
| `↑ ↓ ← →` | Walk / rotate (same as WASD, when not in pose mode) |
| `Z` | Toggle pose mode (arrows then tilt body roll / pitch) |
| `R` | Rebolar |
| `B` | Balance mode |
| `P` | Patinha (toggle) |
| `C` | Turn navigation mode |
| `X` | Omni navigation mode |
| `SPACE` | Stop |
