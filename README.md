# Project Description
This project is a simulation/workaround for a 3 degree of freedom (DOF) per leg hexapod robot named Tiffany. <br>
Robot is https://github.com/Penguin-Lab/tiffany.  <br>

# Modeling and Tools
It was firstly prototyped using Autodesk Fusion, then .step exported.<br>
![Fusion Model](./fusion.png)
## Figure 1 - Fusion Model
.step files were inserted, duplicated and mirrowed into a assembly, mate connectors joined parts and set degrees of freedom.
![Onshape Model](./onshape.png)
## Figure 2 - Onshape Model

# Exporting
onshape-to-robot was used to convert the onshape model assembly into .urdf, which is the description of the robot. <br>
https://github.com/Rhoban/onshape-to-robot <br>
https://onshape-to-robot.readthedocs.io/

If you want to use your own hexapod model, you should modify the keys and a config.json to generate your own assets. <br>
To get the API keys, follow the onshape-to-robot documentation <br>
keys file: <br>
```
export ONSHAPE_API=https://cad.onshape.com
export ONSHAPE_ACCESS_KEY=https://onshape-to-robot.readthedocs.io/en/latest/getting_started.html#using-bashrc
export ONSHAPE_SECRET_KEY=https://onshape-to-robot.readthedocs.io/en/latest/getting_started.html#using-bashrc
```
config.json file:
```
{        
    "documentId": "https://onshape-to-robot.readthedocs.io/en/latest/config.html",
    "outputFormat": "urdf",
    "assemblyName": "robot",
    "ignore": {
        "CORPO_CENTER": "collision"
    }
}

```

After setting up the files, get the tool:
```
pip install onshape-to-robot
```
then, source your keys path using:
```
source keys
```
```
onshape-to-robot pybullet (path of the folder with the config.json)
```
# (In Development)
Refining the kinematics inside pybullet before going through ros and gazebo implementation.

## Setting things up
### Cloning the Repository
```bash
git clone https://github.com/SENAI4LIFE/tiffany_sim.git
```

### Create virtual environment
```bash
python3 -m venv venv
```

### Activate environment
```bash
source ~/venv/bin/activate
```
Shortcut
```bash
nano ~/.bashrc
```

Paste at the end of bash:
```bash
alias venv='source ~/venv/bin/activate'
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

