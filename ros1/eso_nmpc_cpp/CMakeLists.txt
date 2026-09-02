cmake_minimum_required(VERSION 3.0.2)
project(eso_nmpc_cpp)

add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs mavros_msgs message_generation nav_msgs roscpp std_msgs trajectory_msgs)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

add_message_files(FILES NmpcTrajectorySetpoint.msg)
generate_messages(DEPENDENCIES std_msgs)

catkin_package(
  CATKIN_DEPENDS geometry_msgs mavros_msgs message_runtime nav_msgs roscpp std_msgs trajectory_msgs)

set(ESO_NMPC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "eso_nmpc repository root")
set(ACADOS_SOURCE_DIR "$ENV{ACADOS_SOURCE_DIR}" CACHE PATH "acados source tree")
if(ACADOS_SOURCE_DIR STREQUAL "")
  set(ACADOS_SOURCE_DIR "${ESO_NMPC_ROOT}/../acados" CACHE PATH "acados source tree" FORCE)
endif()
set(ACADOS_GENERATED_DIR "${ESO_NMPC_ROOT}/generated/quadrotor_nmpc_ros1_100hz")
set(ACADOS_HASH "e03501ad")

if(NOT EXISTS "${ACADOS_SOURCE_DIR}/include/acados_c/ocp_nlp_interface.h")
  message(FATAL_ERROR "acados headers not found; set ACADOS_SOURCE_DIR")
endif()
set(ACADOS_SOLVER_HEADER "${ACADOS_GENERATED_DIR}/acados_solver_ocp_quadrotor_nmpc_ros1_100hz_${ACADOS_HASH}.h")
set(ACADOS_SOLVER_LIB "${ACADOS_GENERATED_DIR}/libacados_ocp_solver_ocp_quadrotor_nmpc_ros1_100hz_${ACADOS_HASH}.so")
if(NOT EXISTS "${ACADOS_SOLVER_HEADER}" OR NOT EXISTS "${ACADOS_SOLVER_LIB}")
  message(FATAL_ERROR "Generated acados solver missing; run solver/generate_solver.py first")
endif()
find_library(ACADOS_LIBRARY acados PATHS "${ACADOS_SOURCE_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(HPIPM_LIBRARY hpipm PATHS "${ACADOS_SOURCE_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(BLASFEO_LIBRARY blasfeo PATHS "${ACADOS_SOURCE_DIR}/lib" NO_DEFAULT_PATH REQUIRED)

include_directories(include ${catkin_INCLUDE_DIRS} /usr/include/eigen3
  "${ACADOS_GENERATED_DIR}" "${ACADOS_SOURCE_DIR}/include"
  "${ACADOS_SOURCE_DIR}/include/blasfeo/include" "${ACADOS_SOURCE_DIR}/include/hpipm/include")
add_executable(${PROJECT_NAME} src/eso_nmpc_cpp_node.cpp)
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${ACADOS_SOLVER_LIB}
  ${ACADOS_LIBRARY} ${HPIPM_LIBRARY} ${BLASFEO_LIBRARY} Threads::Threads dl m)
# Keep an old-style RPATH in the executable.  The generated solver and
# libacados have transitive hpipm/blasfeo dependencies; DT_RUNPATH is not
# searched transitively by the dynamic loader, while DT_RPATH is.  This lets
# rosrun/roslaunch work without requiring users to export LD_LIBRARY_PATH.
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_RPATH "${ACADOS_GENERATED_DIR};${ACADOS_SOURCE_DIR}/lib"
  INSTALL_RPATH "${ACADOS_GENERATED_DIR};${ACADOS_SOURCE_DIR}/lib"
  LINK_FLAGS "-Wl,--disable-new-dtags")

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
