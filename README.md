# kdl rmp
implementation of Riemmanian Motion Policies flow using Orocos KDL for forward kinematics (based on Georgia tech here https://github.com/gtrll/multi-robot-rmpflow)

# standalone installation (no ROS)
1. install mujoco and mujoco_py using these [instructions](https://github.com/openai/mujoco-py). 
2. install SIP 4 - [instructions](https://docs.huihoo.com/pyqt/sip4/installation.html) (DO NOT use pip)
3. build orocos kdl from source - [repo](https://github.com/orocos/orocos_kinematics_dynamics/commits/master)

    Note: remove lines 35-38, 40 in [kinfam.sip](https://github.com/orocos/orocos_kinematics_dynamics/blob/master/python_orocos_kdl/PyKDL/sip/kinfam.sip) and then build as normal to change
    Python3 bindings API back to Python2 - kdl_parser_py will break otherwise
    
    Build orocos_kdl first, then python_orocos_kdl after.
    Note: ensure that you set the python binding flags to SIP, and then the other
    Python flags accurately reflect your current system configuration.
4. build [kdl_parser_py](https://github.com/ros/kdl_parser) (and all of its dependencies: [catkin](http://wiki.ros.org/catkin?distro=noetic), [urdfdom_py](http://wiki.ros.org/urdfdom_py))

5. python setup.py install 



# usage
see rmp_example_sparse.py for a minimal example using a Jaco arm
