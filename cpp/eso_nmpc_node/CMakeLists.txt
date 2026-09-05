cmake_minimum_required(VERSION 3.16)
project(eso_nmpc_node LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

set(ESO_NMPC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "eso_nmpc repository root")
set(_acados_from_environment "$ENV{ACADOS_SOURCE_DIR}")
if(NOT DEFINED ACADOS_SOURCE_DIR OR ACADOS_SOURCE_DIR STREQUAL "")
  if(NOT _acados_from_environment STREQUAL "")
    set(ACADOS_SOURCE_DIR "${_acados_from_environment}" CACHE PATH "acados source tree")
  else()
    # Keep the common source checkout layout convenient without assuming a
    # particular user's home directory.  Pass -DACADOS_SOURCE_DIR explicitly
    # when acados is installed elsewhere (as it normally is on the Orin).
    set(ACADOS_SOURCE_DIR "${ESO_NMPC_ROOT}/../acados" CACHE PATH "acados source tree")
  endif()
endif()
if(NOT EXISTS "${ACADOS_SOURCE_DIR}/include/acados_c/ocp_nlp_interface.h")
  message(FATAL_ERROR
    "acados headers not found under '${ACADOS_SOURCE_DIR}'. "
    "Set ACADOS_SOURCE_DIR or pass -DACADOS_SOURCE_DIR=/path/to/acados.")
endif()
set(ACADOS_GENERATED_DIR "${ESO_NMPC_ROOT}/generated/quadrotor_nmpc" CACHE PATH "generated acados directory")
set(ACADOS_HASH "1c2d851e" CACHE STRING "generated acados formulation hash")
if(NOT ACADOS_HASH MATCHES "^[0-9a-fA-F]+$")
  message(FATAL_ERROR
    "ACADOS_HASH must contain only hexadecimal characters: '${ACADOS_HASH}'.")
endif()
string(TOUPPER "${ACADOS_HASH}" ACADOS_HASH_UPPER)
set(ACADOS_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${ACADOS_CONFIG_DIR}")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/eso_nmpc_node/acados_solver_config.hpp.in"
  "${ACADOS_CONFIG_DIR}/acados_solver_config.hpp" @ONLY)
set(ACADOS_SOLVER_LIB "${ACADOS_GENERATED_DIR}/libacados_ocp_solver_ocp_quadrotor_nmpc_${ACADOS_HASH}.so")
set(ACADOS_SOLVER_HEADER "${ACADOS_GENERATED_DIR}/acados_solver_ocp_quadrotor_nmpc_${ACADOS_HASH}.h")
if(NOT EXISTS "${ACADOS_SOLVER_LIB}" OR NOT EXISTS "${ACADOS_SOLVER_HEADER}")
  message(FATAL_ERROR
    "Missing generated acados artifact: ${ACADOS_SOLVER_LIB}. "
    "Run 'python3 solver/generate_solver.py' from the repository root on this target.")
endif()

find_library(ACADOS_LIBRARY NAMES acados PATHS "${ACADOS_SOURCE_DIR}/lib" NO_DEFAULT_PATH)
find_library(HPIPM_LIBRARY NAMES hpipm PATHS "${ACADOS_SOURCE_DIR}/lib" NO_DEFAULT_PATH)
find_library(BLASFEO_LIBRARY NAMES blasfeo PATHS "${ACADOS_SOURCE_DIR}/lib" NO_DEFAULT_PATH)
if(NOT ACADOS_LIBRARY OR NOT HPIPM_LIBRARY OR NOT BLASFEO_LIBRARY)
  message(FATAL_ERROR
    "acados shared libraries (acados, hpipm, blasfeo) were not found under "
    "${ACADOS_SOURCE_DIR}/lib")
endif()

add_executable(${PROJECT_NAME}
  src/eso_nmpc_node.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE
  include
  "${ACADOS_CONFIG_DIR}"
  ${EIGEN3_INCLUDE_DIRS}
  "${ACADOS_GENERATED_DIR}"
  "${ACADOS_SOURCE_DIR}/include"
  "${ACADOS_SOURCE_DIR}/include/blasfeo/include"
  "${ACADOS_SOURCE_DIR}/include/hpipm/include"
)
target_link_libraries(${PROJECT_NAME}
  "${ACADOS_SOLVER_LIB}"
  "${ACADOS_LIBRARY}"
  "${HPIPM_LIBRARY}"
  "${BLASFEO_LIBRARY}"
  Threads::Threads m dl
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_RPATH "${ACADOS_GENERATED_DIR};${ACADOS_SOURCE_DIR}/lib"
  INSTALL_RPATH "${ACADOS_GENERATED_DIR};${ACADOS_SOURCE_DIR}/lib"
)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
target_link_options(${PROJECT_NAME} PRIVATE -Wl,--disable-new-dtags)
ament_target_dependencies(${PROJECT_NAME} rclcpp px4_msgs std_msgs Eigen3)

install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS check_environment.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(FILES README.md DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
ament_package()
