from motor_skills.rmp.jaco_rmp import JacoFlatRMP

j_rmp = JacoFlatRMP()
print(j_rmp.eval([1, 1.5, 2.3, 0.2, 1.1, 1.1], [0]*6))

