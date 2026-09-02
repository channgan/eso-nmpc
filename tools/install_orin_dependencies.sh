#!/usr/bin/env bash
# Install/build the dependencies for the C++ NMPC companion process.
# This script is intended for an Ubuntu 22.04/ROS 2 Humble Orin.  It is
# deliberately explicit about source/build locations so it never installs an
# x86 generated solver on an ARM target.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
USER_HOME_DIR="${HOME}"
ACADOS_DIR="${ACADOS_SOURCE_DIR:-${USER_HOME_DIR}/acados}"
AGENT_DIR="${MICRO_XRCE_AGENT_DIR:-${USER_HOME_DIR}/Micro-XRCE-DDS-Agent}"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
PX4_WS="${PX4_ROS2_WS:-${USER_HOME_DIR}/px4_ros2_ws}"
SKIP_ACADOS=0
SKIP_AGENT=0
SKIP_BUILD=0

usage() {
  cat <<EOF
Usage: $0 [options]

Installs the ROS/C++/Python dependencies, builds ARM64 acados and
Micro-XRCE-DDS-Agent when needed, generates the native solver, and builds the
eso_nmpc_node package.

Options:
  --acados DIR       acados checkout/build prefix (default: ${ACADOS_DIR})
  --agent DIR        Micro-XRCE-DDS-Agent checkout (default: ${AGENT_DIR})
  --px4-ws DIR       PX4 ROS 2 workspace (default: ${PX4_WS})
  --skip-acados      do not clone/build acados
  --skip-agent       do not clone/build Micro-XRCE-DDS-Agent
  --skip-build       install/check only; do not generate or compile
  -h, --help         show this help
EOF
}

while (($#)); do
  case "$1" in
    --acados|--agent|--px4-ws)
      (($# >= 2)) || { echo "missing value for $1" >&2; exit 2; }
      case "$1" in
        --acados) ACADOS_DIR="$2" ;;
        --agent) AGENT_DIR="$2" ;;
        --px4-ws) PX4_WS="$2" ;;
      esac
      shift 2
      ;;
    --skip-acados) SKIP_ACADOS=1; shift ;;
    --skip-agent) SKIP_AGENT=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

info() { printf '\n[install] %s\n' "$*"; }
fail() { echo "[install] ERROR: $*" >&2; exit 1; }

[[ "${EUID}" -ne 0 ]] || fail "run as a normal user; the script uses sudo only for apt and system installs"
command -v sudo >/dev/null || fail "sudo is required"
command -v apt-get >/dev/null || fail "this installer requires Ubuntu/Debian apt"

if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == "ubuntu" ]] || fail "Ubuntu is required; detected ${ID:-unknown}"
  if [[ "${VERSION_ID:-}" != "22.04" ]]; then
    echo "[install] WARNING: tested on Ubuntu 22.04; detected ${VERSION_ID:-unknown}" >&2
  fi
fi

info "installing system and ROS packages"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential cmake ninja-build git pkg-config \
  python3-dev python3-pip python3-venv python3-colcon-common-extensions \
  libeigen3-dev liblapack-dev \
  "ros-${ROS_DISTRO_NAME}-ros-base" \
  "ros-${ROS_DISTRO_NAME}-rclcpp" \
  "ros-${ROS_DISTRO_NAME}-std-msgs" \
  "ros-${ROS_DISTRO_NAME}-rmw-cyclonedds-cpp"

[[ -f "${ROS_SETUP}" ]] || fail "ROS setup file missing: ${ROS_SETUP}; install/configure the ROS repository first"
# shellcheck disable=SC1090
source "${ROS_SETUP}"

if ((SKIP_AGENT == 0)); then
  if command -v MicroXRCEAgent >/dev/null; then
    info "Micro-XRCE-DDS-Agent already installed: $(command -v MicroXRCEAgent)"
  else
    info "building Micro-XRCE-DDS-Agent for the target architecture"
    if [[ ! -d "${AGENT_DIR}/.git" ]]; then
      [[ ! -e "${AGENT_DIR}" ]] || fail "agent path exists but is not a git checkout: ${AGENT_DIR}"
      mkdir -p "$(dirname -- "${AGENT_DIR}")"
      git clone --depth 1 --branch v2.4.3 \
        https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "${AGENT_DIR}"
    else
      git -C "${AGENT_DIR}" fetch --depth 1 origin v2.4.3
      git -C "${AGENT_DIR}" checkout --detach FETCH_HEAD
    fi
    cmake -S "${AGENT_DIR}" -B "${AGENT_DIR}/build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${AGENT_DIR}/build" --parallel "$(nproc)"
    sudo cmake --install "${AGENT_DIR}/build"
  fi
fi

if ((SKIP_ACADOS == 0)); then
  info "building acados and its ARM64 shared libraries"
  if [[ ! -d "${ACADOS_DIR}/.git" ]]; then
    if [[ -f "${ACADOS_DIR}/include/acados_c/ocp_nlp_interface.h" ]]; then
      info "using existing acados tree: ${ACADOS_DIR}"
    else
      [[ ! -e "${ACADOS_DIR}" ]] || fail "acados path exists but is not a checkout with headers: ${ACADOS_DIR}"
      mkdir -p "$(dirname -- "${ACADOS_DIR}")"
      git clone https://github.com/acados/acados.git "${ACADOS_DIR}"
    fi
  fi
  if [[ ! -f "${ACADOS_DIR}/include/acados_c/ocp_nlp_interface.h" ]]; then
    fail "acados headers are missing under ${ACADOS_DIR}"
  fi
  if [[ -d "${ACADOS_DIR}/.git" ]]; then
    git -C "${ACADOS_DIR}" submodule update --init --recursive
  fi
  cmake -S "${ACADOS_DIR}" -B "${ACADOS_DIR}/build" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="${ACADOS_DIR}"
  cmake --build "${ACADOS_DIR}/build" --parallel "$(nproc)"
  cmake --install "${ACADOS_DIR}/build"
fi

export ACADOS_SOURCE_DIR="${ACADOS_DIR}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export LD_LIBRARY_PATH="${ACADOS_DIR}/lib:${LD_LIBRARY_PATH:-}"

if ((SKIP_BUILD == 0)); then
  info "creating the solver-generation Python environment"
  if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
    python3 -m venv "${REPO_ROOT}/.venv"
  fi
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -r "${REPO_ROOT}/requirements.txt"
  python -m pip install -e "${ACADOS_DIR}/interfaces/acados_template"

  info "generating the native solver"
  cd "${REPO_ROOT}"
  python solver/generate_solver.py

  info "building the ROS 2 C++ node"
  [[ -d "${PX4_WS}/src" ]] || fail "PX4 ROS 2 workspace missing: ${PX4_WS}/src"
  [[ -f "${PX4_WS}/src/px4_msgs/msg/nmpc_trajectory_setpoint.msg" ]] || \
    fail "px4_msgs with NmpcTrajectorySetpoint is missing under ${PX4_WS}/src/px4_msgs"
  mkdir -p "${PX4_WS}/src"
  if [[ ! -e "${PX4_WS}/src/eso_nmpc_node" ]]; then
    ln -s "${REPO_ROOT}/cpp/eso_nmpc_node" "${PX4_WS}/src/eso_nmpc_node"
  elif [[ "$(realpath -m -- "${PX4_WS}/src/eso_nmpc_node")" != "${REPO_ROOT}/cpp/eso_nmpc_node" ]]; then
    fail "${PX4_WS}/src/eso_nmpc_node already points to a different package; remove it or pass the correct PX4_ROS2_WS"
  fi
  cd "${PX4_WS}"
  source "${ROS_SETUP}"
  if [[ -f "install/setup.bash" ]]; then source install/setup.bash; fi
  colcon build --packages-select px4_msgs eso_nmpc_node --cmake-clean-cache \
    --cmake-args -DESO_NMPC_ROOT="${REPO_ROOT}" -DACADOS_SOURCE_DIR="${ACADOS_DIR}"
fi

info "running the read-only deployment check"
cd "${REPO_ROOT}"
source "${ROS_SETUP}"
if [[ -f "${PX4_WS}/install/setup.bash" ]]; then source "${PX4_WS}/install/setup.bash"; fi
CHECKER_ARGS=(--root "${REPO_ROOT}" --acados "${ACADOS_DIR}")
if ((SKIP_BUILD == 0)); then
  CHECKER_ARGS+=(--require-generator --executable "${PX4_WS}/install/eso_nmpc_node/lib/eso_nmpc_node/eso_nmpc_node")
fi
python3 "${REPO_ROOT}/cpp/eso_nmpc_node/check_environment.py" "${CHECKER_ARGS[@]}"

info "complete; set flight_log_root to a persistent directory before flight"
