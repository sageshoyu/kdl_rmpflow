# kdl_rmpflow
implementation of [RMPFlow](https://arxiv.org/abs/1811.07049) using Orocos KDL for forward kinematics (based on Georgia tech here https://github.com/gtrll/multi-robot-rmpflow)

# Standalone Installation (no ROS)
1. install mujoco and mujoco_py using these [instructions](https://github.com/openai/mujoco-py). 
2. install SIP 4 - [instructions](https://docs.huihoo.com/pyqt/sip4/installation.html) (DO NOT use pip) No need to set any flags on <code>configure.py</code>.
3. build orocos kdl from source - [repo](https://github.com/orocos/orocos_kinematics_dynamics/commits/master)

    Use this commit: 2e3fc6a20c9634861c5708aa9c016f080a7a3f7c
    Note: remove lines 35-38, 40 in [kinfam.sip](https://github.com/orocos/orocos_kinematics_dynamics/blob/master/python_orocos_kdl/PyKDL/sip/kinfam.sip) and then build as normal to change
    Python3 bindings API back to Python2 - kdl_parser_py will break otherwise
    
    Build orocos_kdl first ([instructions](https://www.orocos.org/kdl/installation-manual)), then python_orocos_kdl after.
    Note: Use Eigen 3, NOT Eigen 2 for orocos_kdl. When installing python_orocos_kdl, ensure that you set the python binding flags to SIP and that the other
    Python flags accurately reflect your current system configuration. 
    
    After building orocos_kdl, build PyKDL using tehse [instructions](https://github.com/orocos/orocos_kinematics_dynamics/issues/115).l
4. build [kdl_parser_py](https://github.com/ros/kdl_parser) (and all of its dependencies: [catkin](http://wiki.ros.org/catkin?distro=noetic), [urdfdom_py](http://wiki.ros.org/urdfdom_py))

Note: yes, there is a setup.py, but it currently is not updated because dependencies are in flux.

## Other Dependencies:
1) Numpy
2) Scipy
3) [QuatDMP](https://github.com/sageshoyu/QuatDMP)
4) Open3D


# usage
see rmp_example_sparse.py for a minimal example using a Jaco arm
