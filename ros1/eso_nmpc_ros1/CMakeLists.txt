cmake_minimum_required(VERSION 3.0.2)
project(eso_nmpc_ros1)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mavros_msgs
  nav_msgs
  rospy
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/eso_nmpc_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY src/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/${PROJECT_NAME}
)
