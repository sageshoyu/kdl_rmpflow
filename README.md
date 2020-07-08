# kdl rmp
implementation of Riemmanian Motion Policies flow using Orocos KDL for forward kinematics (based on Georgia tech here https://github.com/gtrll/multi-robot-rmpflow)

# standalone installation (no ROS)
1. install mujoco and mujoco_py using these [instructions](https://github.com/openai/mujoco-py). Note that we have a lab license and I will email you the access key.
2. install SIP 4 - [instructions](https://docs.huihoo.com/pyqt/sip4/installation.html) (DO NOT use pip)
3. build orocos kdl - [repo](https://github.com/orocos/orocos_kinematics_dynamics/commits/master)
    
    Note: remove lines 35-38, 40 in [kinfam.sip](https://github.com/orocos/orocos_kinematics_dynamics/blob/master/python_orocos_kdl/PyKDL/sip/kinfam.sip) and then build as normal to change
    Python3 bindings API back to Python2 - kdl_parser_py will break otherwise
4. build [kdl_parser_py](https://github.com/ros/kdl_parser) (install all dependencies as well)

5. python setup.py install 



# usage
see rmp_example_sparse.py for a minimal example using a Jaco arm
