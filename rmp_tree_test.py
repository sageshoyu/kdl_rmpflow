from urdf_parser_py.urdf import URDF as u_parser
from motor_skills.rmp.kdl_rmp import tree_from_robot
import numpy as np

robot = u_parser.from_xml_file('assets/kinova_j2s6s300/ros-j2s6s300.xml')
root, leaves = tree_from_robot(robot)

test = np.arange(1, 13).reshape(-1, 1)
for proj in root.children:
    print(proj.name + " psi res: " + str(proj.psi(test).T))

for proj in root.children:
    print(proj.name + " child: " + str(proj.children[0].name))
    print("same in dict: " + str(proj.children[0] == leaves[proj.children[0].name]))
