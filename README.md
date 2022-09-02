# opt-rkhs-bounds

This repo contains the supplementary material for the paper

```
@article{scharnhorst2021robust,
  title={Robust Uncertainty Bounds in Reproducing Kernel Hilbert Spaces: A Convex Optimization Approach},
  author={Scharnhorst, Paul and Maddalena, Emilio T and Jiang, Yuning and Jones, Colin N},
  journal={arXiv preprint arXiv:2104.09582},
  year={2021}
}
```

## Description

In the above work, we propose a novel uncertainty quantification technique that bounds the out-of-sample values of an unknown real-valued function. The method is developed in the a kernel setting, and can account for finite measurement noise. 

We showcase how the theory can be used through a number of examples, including: a function bouding task, a data-driven optimization problem with unknown constraints, and the safety certification of a sequence of control actions for a dynamical system.

<img src="https://github.com/PREDICT-EPFL/opt-rkhs-bounds/blob/main/pics/repo_pic_A.png" width="650" height="auto">
<img src="https://github.com/PREDICT-EPFL/opt-rkhs-bounds/blob/main/pics/repo_pic_B.png" width="650" height="auto">

## MATLAB Dependencies 

- YALMIP (https://yalmip.github.io/)
- casADi (https://web.casadi.org/)
- Multi-parametric Toolbox 3.0 (https://www.mpt3.org/)
