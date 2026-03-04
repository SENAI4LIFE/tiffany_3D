# Project Description
This project is a simulation environment for a 3 degree of freedom (DOF) per leg hexapod named Tiffany. <br>
Robot is https://github.com/Penguin-Lab/tiffany.  <br>

# Modeling and Tools
It was firstly prototyped using Autodesk Fusion, then the .stl was uploaded to onshape, therefore parts inserted into a assembly, mate connectors joined parts and set degrees of freedom.<br>
![Fusion Model](./fusion.png)
# Figure 1 - Fusion Model

![Onshape Model](./onshape.png)
# Figure 2 - Onshape Model

# Exporting
onshape-to-robot was used to convert the onshape model assembly into .urdf, which is the description of the robot.
https://github.com/Rhoban/onshape-to-robot <br>
https://onshape-to-robot.readthedocs.io/

# (In Development)
Refining the kinematics inside pybullet before going through ros and gazebo implementation.

## Setting things up
### Cloning the Repository
```bash
git clone https://github.com/SENAI4LIFE/tiffany_sim.git
```

### Create virtual environment
```bash
source ~/tiffany/bin/activate
```

### Activate environment
```bash
python3 -m venv tiffany
```
Shortcut
```bash
nano ~/.bashrc
```

Paste at the end of bash:
```bash
alias tiffany='source ~/tiffany/bin/activate'
alias tiffanypy='cd tiffany_sim/workspace/pybullet'
alias tiffanygz='cd tiffany_sim/workspace/gazebosim'
```

# (PyBullet) 
## Navigate to project directory
```bash
cd ~/tiffany_sim/workspace/pybullet
```

```bash
pip install --upgrade pip
pip install pybullet
pip install numpy
```

### Start Simulation
```bash
python3 main.py
```

