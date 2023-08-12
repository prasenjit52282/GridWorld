# GridWorld

<p>
<img src="./assets/gridworld.png" style="float:left;width:40%"/>Gridworld is a tool for easily producing custom grid environments to test model-based and model-free (only table-based) Reinforcement Learning algorithms. The package provides an uniform way of defining a grid-world and place agent and goal state. Further, it builds the transition probability matrix (P_sas) and the reward matrix (R_sa) from the defined environment to test planning algorithms. Moreover, for model-free algorithms, the package provides a openai-gym like interface to interact with the environment and explore.</p>

# Installation
To install the package in your python(>=3.9) environment you need to run the below commands:
```
git clone https://github.com/prasenjit52282/GridWorld.git
cd GridWorld
python setup.py install
```

# File Structure
```
- gridworld
    └── modules
        └── images
            └── agent.png
            └── goal.png
            └── wall.png
            └── {direction}.png
        └── __init__.py
        └── agent.py
        └── goal.py
        └── state.py
        └── wall.py
    └── __init__.py
    └── gridworld.py
- requirements.txt
- .gitignore
- LICENSE
- MANIFEST.in
- setup.py
- test.py
```