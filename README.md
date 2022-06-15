# Codebase for Hierarchical _k_-Step Latent (HKSL)

### Prerequisites
This codebase requires the MuJoCo physics engine [1].

### Training an HKSL Agent
An HKSL agent can be trained by running the ```player.py``` script.

For example, if we wish to train  an agent in Cartpole, Swingup with no distractors, run:
```(bash)
python3 player.py --env cartpole_swingup
```

To run Cartpole, Swingup with easy color distractors, run:
```(bash)
python3 player.py --env gdc-cartpole_swingup__easy__dynamic --distractors color
```

To run Cartpole, Swingup with medium camera distractors, run:
```(bash)
python3 player.py --env gdc-cartpole_swingup__medium__dynamic --distractors camera
```

### Helpful Sources
We make use of the distracting control suite [2] code, which can be found in the ```/distracting_control``` folder.

Our training loop and agent class structure is based off of SAC-AE's [3] codebase.

### References
[1] E. Todorov, T. Erez, and Y. Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5026–5033, 2012.

[2] . Stone, O. Ramirez, K. Konolige, and R. Jonschkowski. The distracting control suite – a challenging benchmark for reinforcement learning from pixels. arXiv preprint arXiv:2101.02722, 2021.

[3] D. Yarats, A. Zhang, I. Kostrikov, B. Amos, J. Pineau, and R. Fergus. Improving sample efficiency in model-free reinforcement learning from images. arXiv preprint arXiv:1910.01741, 2020.