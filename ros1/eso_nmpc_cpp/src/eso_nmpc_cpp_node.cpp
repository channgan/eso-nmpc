#include <acados_c/ocp_nlp_interface.h>
#include <acados/utils/types.h>
#include <acados_solver_ocp_quadrotor_nmpc_ros1_iris_100hz_2109265f.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/EstimatorStatus.h>
#include <mavros_msgs/ManualControl.h>
#include <mavros_msgs/MessageInterval.h>
#include <mavros_msgs/RCIn.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <eso_nmpc_cpp/NmpcTrajectorySetpoint.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace eso_nmpc_cpp {
namespace {
constexpr int kNx = 13;
constexpr int kNu = 4;
constexpr int kNp = 7;
constexpr int kN = OCP_QUADROTOR_NMPC_ROS1_IRIS_100HZ_2109265F_N;
constexpr int kPoints = kN + 1;
using State = Eigen::Matrix<double, kNx, 1>;
using States = Eigen::Matrix<double, kNx, kPoints>;
using Controls = Eigen::Matrix<double, kNu, kN>;
using Clock = std::chrono::steady_clock;

Eigen::Vector4d normalize(Eigen::Vector4d q) {
  const double n = q.norm();
  return n > 1e-12 ? q / n : Eigen::Vector4d(1, 0, 0, 0);
}

Eigen::Vector4d multiply(const Eigen::Vector4d &a, const Eigen::Vector4d &b) {
  return Eigen::Vector4d(a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
                        a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
                        a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
                        a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]);
}

Eigen::Vector3d rotate(const Eigen::Vector4d &q, const Eigen::Vector3d &v) {
  const Eigen::Vector4d n = normalize(q);
  const Eigen::Vector4d c(n[0], -n[1], -n[2], -n[3]);
  return multiply(multiply(n, Eigen::Vector4d(0, v[0], v[1], v[2])), c).tail<3>();
}

Eigen::Vector4d matrix_quaternion(const Eigen::Matrix3d &r) {
  Eigen::Quaterniond q(r);
  Eigen::Vector4d out;
  out << q.w(), q.x(), q.y(), q.z();
  return normalize(out);
}

Eigen::Vector4d enu_flu_to_ned_frd(const Eigen::Vector4d &q_enu_flu) {
  const Eigen::Vector4d q_ne(0, std::sqrt(0.5), std::sqrt(0.5), 0);
  const Eigen::Vector4d q_flu_frd(0, 1, 0, 0);
  return normalize(multiply(multiply(q_ne, normalize(q_enu_flu)), q_flu_frd));
}

double stamp_seconds(const Clock::time_point &t) {
  return std::chrono::duration<double>(t.time_since_epoch()).count();
}

double apply_deadzone(double value, double deadzone) {
  if (!std::isfinite(value)) return 0.0;
  value = std::clamp(value, -1.0, 1.0);
  if (std::abs(value) <= deadzone) return 0.0;
  return std::copysign((std::abs(value) - deadzone) / (1.0 - deadzone), value);
}

double wrap_angle(double value) { return std::atan2(std::sin(value), std::cos(value)); }

class VelocityLeso {
 public:
  VelocityLeso(double bandwidth, double clamp, double innovation_limit)
      : beta1_(2.0 * bandwidth), beta2_(bandwidth * bandwidth), clamp_(clamp),
        innovation_limit_(innovation_limit) {
    if (!(bandwidth > 0.0) || !(clamp > 0.0) || !(innovation_limit > 0.0))
      throw std::invalid_argument("ESO parameters must be positive");
  }
  void reset(const Eigen::Vector3d &velocity, const Eigen::Vector3d &disturbance) {
    velocity_hat_ = velocity;
    disturbance_hat_ = disturbance.cwiseMax(-clamp_).cwiseMin(clamp_);
    initialized_ = true;
  }
  Eigen::Vector3d update(const Eigen::Vector3d &velocity,
                        const Eigen::Vector3d &model_acceleration, double dt) {
    if (!(dt > 0.0) || !std::isfinite(dt)) return disturbance_hat_;
    if (!initialized_) { reset(velocity, Eigen::Vector3d::Zero()); return disturbance_hat_; }
    const double step = std::min(dt, 0.1);
    Eigen::Vector3d innovation = velocity - velocity_hat_;
    for (int i = 0; i < 3; ++i)
      innovation[i] = std::clamp(innovation[i], -innovation_limit_, innovation_limit_);
    velocity_hat_ += step * (model_acceleration + disturbance_hat_ + beta1_ * innovation);
    disturbance_hat_ += step * (beta2_ * innovation);
    disturbance_hat_ = disturbance_hat_.cwiseMax(-clamp_).cwiseMin(clamp_);
    return disturbance_hat_;
  }
 private:
  double beta1_, beta2_, clamp_, innovation_limit_;
  Eigen::Vector3d velocity_hat_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d disturbance_hat_{Eigen::Vector3d::Zero()};
  bool initialized_{false};
};

std::string session_stamp() {
  const auto now = std::chrono::system_clock::now();
  const auto tt = std::chrono::system_clock::to_time_t(now);
  std::tm local{};
  localtime_r(&tt, &local);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()).count() % 1000;
  std::ostringstream out;
  out << std::put_time(&local, "%Y%m%d_%H%M%S") << '_' << std::setfill('0') << std::setw(3) << ms;
  return out.str();
}

struct Trajectory {
  uint32_t sequence{0};
  double sample_time{0};
  std::array<double, 3*kPoints> position{};
  std::array<double, 3*kPoints> velocity{};
  std::array<double, 3*kPoints> acceleration{};
  std::array<double, kPoints> yaw{};
};

class AcadosController {
 public:
  AcadosController(double mass, double gravity, double sample_time, double tmin,
                   double tmax, const Eigen::Vector3d &rmax, bool warm)
      : mass_(mass), gravity_(gravity), sample_time_(sample_time), tmin_(tmin),
        tmax_(tmax), rmax_(rmax), warm_(warm) {
    capsule_ = ocp_quadrotor_nmpc_ros1_iris_100hz_2109265f_acados_create_capsule();
    if (!capsule_ || ocp_quadrotor_nmpc_ros1_iris_100hz_2109265f_acados_create_with_discretization(
          capsule_, kN, nullptr) != ACADOS_SUCCESS) throw std::runtime_error("acados create failed");
    config_ = ocp_quadrotor_nmpc_ros1_iris_100hz_2109265f_acados_get_nlp_config(capsule_);
    dims_ = ocp_quadrotor_nmpc_ros1_iris_100hz_2109265f_acados_get_nlp_dims(capsule_);
    in_ = ocp_quadrotor_nmpc_ros1_iris_100hz_2109265f_acados_get_nlp_in(capsule_);
    out_ = ocp_quadrotor_nmpc_ros1_iris_100hz_2109265f_acados_get_nlp_out(capsule_);
  }
  ~AcadosController() {
    if (capsule_) {
      ocp_quadrotor_nmpc_ros1_iris_100hz_2109265f_acados_free(capsule_);
      ocp_quadrotor_nmpc_ros1_iris_100hz_2109265f_acados_free_capsule(capsule_);
    }
  }
  AcadosController(const AcadosController &) = delete;

  bool solve(const State &state, const States &refs, const Controls &ff,
             const Eigen::Vector3d &disturbance, Eigen::Vector4d &command) {
    std::array<double, kNx*kPoints> x{};
    std::array<double, kNu*kN> u{};
    for (int i=0; i<kPoints; ++i) {
      const int src = warm_ && have_last_ ? std::min(i+1, kN) : i;
      for (int j=0; j<kNx; ++j) x[i*kNx+j] = warm_ && have_last_ ? last_x_[src*kNx+j] : refs(j,i);
    }
    for (int i=0; i<kN; ++i) {
      const int src = warm_ && have_last_ ? std::min(i+1, kN-1) : i;
      for (int j=0; j<kNu; ++j) u[i*kNu+j] = warm_ && have_last_ ? last_u_[src*kNu+j] : ff(j,i);
    }
    for (int j=0; j<kNx; ++j) x[j] = state[j];
    for (int i=0; i<kN; ++i) {
      ocp_nlp_out_set(config_, dims_, out_, in_, i, "x", &x[i*kNx]);
      ocp_nlp_out_set(config_, dims_, out_, in_, i, "u", &u[i*kNu]);
      std::array<double,kNp> p{};
      for (int j=0;j<3;++j) p[j]=disturbance[j];
      for (int j=0;j<4;++j) p[3+j]=refs(6+j,i);
      ocp_quadrotor_nmpc_ros1_iris_100hz_2109265f_acados_update_params(capsule_, i, p.data(), kNp);
      std::array<double,13> y{};
      for (int j=0;j<6;++j) y[j]=refs(j,i);
      for (int j=0;j<4;++j) y[9+j]=ff(j,i);
      ocp_nlp_cost_model_set(config_, dims_, in_, i, "yref", y.data());
    }
    ocp_nlp_out_set(config_, dims_, out_, in_, kN, "x", &x[kN*kNx]);
    std::array<double,kNx> x0{}; std::copy(state.data(), state.data()+kNx, x0.data());
    ocp_nlp_constraints_model_set(config_,dims_,in_,out_,0,"lbx",x0.data());
    ocp_nlp_constraints_model_set(config_,dims_,in_,out_,0,"ubx",x0.data());
    std::array<double,kNp> p{}; for (int j=0;j<3;++j) p[j]=disturbance[j];
    for (int j=0;j<4;++j) p[3+j]=refs(6+j,kN);
    ocp_quadrotor_nmpc_ros1_iris_100hz_2109265f_acados_update_params(capsule_, kN, p.data(), kNp);
    std::array<double,9> yt{}; for (int j=0;j<6;++j) yt[j]=refs(j,kN);
    ocp_nlp_cost_model_set(config_,dims_,in_,kN,"yref",yt.data());
    const int status = ocp_quadrotor_nmpc_ros1_iris_100hz_2109265f_acados_solve(capsule_);
    if (status != ACADOS_SUCCESS) { have_last_=false; return false; }
    for (int i=0;i<kPoints;++i) ocp_nlp_out_get(config_,dims_,out_,i,"x",&last_x_[i*kNx]);
    for (int i=0;i<kN;++i) ocp_nlp_out_get(config_,dims_,out_,i,"u",&last_u_[i*kNu]);
    have_last_=true;
    command[0]=std::clamp(last_u_[0],tmin_,tmax_);
    for (int j=1;j<kNu;++j) command[j]=std::clamp(last_u_[j],-rmax_[j-1],rmax_[j-1]);
    return true;
  }
  void reset() { have_last_=false; }
  bool warm_start_available() const { return have_last_; }
 private:
  using Capsule = ocp_quadrotor_nmpc_ros1_iris_100hz_2109265f_solver_capsule;
  Capsule *capsule_{nullptr}; ocp_nlp_config *config_{nullptr}; ocp_nlp_dims *dims_{nullptr};
  ocp_nlp_in *in_{nullptr}; ocp_nlp_out *out_{nullptr};
  double mass_,gravity_,sample_time_,tmin_,tmax_; Eigen::Vector3d rmax_; bool warm_,have_last_{false};
  std::array<double,kNx*kPoints> last_x_{}; std::array<double,kNu*kN> last_u_{};
};
}  // namespace

class Node {
 public:
  Node() : nh_(), pnh_("~"), controller_(param("mass",2.0),param("gravity",9.80665),
      param("sample_time",0.01),param("thrust_min",3.3278),param("thrust_max",27.7314),
      Eigen::Vector3d(param("body_rate_max_x",1.0),param("body_rate_max_y",1.0),param("body_rate_max_z",1.0)),
      param("warm_start",true)) {
    mass_=param("mass",2.0); gravity_=param("gravity",9.80665); dt_=param("sample_time",0.01);
    horizon_=param("horizon_steps",30); timeout_=param("reference_timeout",0.20);
    enabled_=param("control_enabled_at_start",false);
    shadow_mode_=param("shadow_mode",true);
    control_enable_time_=stamp_seconds(Clock::now());
    prestream_s_=param("prestream_s",1.5); service_timeout_s_=param("service_timeout_s",8.0);
    state_timeout_s_=param("state_timeout_s",1.0); offboard_mode_=param("offboard_mode",std::string("OFFBOARD"));
    position_mode_=param("position_mode",std::string("AUTO.LOITER"));
    solver_recovery_time_s_=param("solver_recovery_time",2.0);
    if (!(solver_recovery_time_s_ > 0.0) || !std::isfinite(solver_recovery_time_s_))
      throw std::invalid_argument("solver_recovery_time must be positive");
    handoff_stable_time_s_=param("handoff_stable_time",1.0);
    offboard_entry_delay_s_=param("offboard_entry_delay",0.2);
    handoff_max_horizontal_speed_=param("handoff_max_horizontal_speed",0.15);
    handoff_max_vertical_speed_=param("handoff_max_vertical_speed",0.10);
    handoff_max_position_drift_=param("handoff_max_position_drift",0.25);
    if (!(handoff_stable_time_s_ > 0.0) || !(offboard_entry_delay_s_ >= 0.0) ||
        !(handoff_max_horizontal_speed_ > 0.0) || !(handoff_max_vertical_speed_ > 0.0) ||
        !(handoff_max_position_drift_ > 0.0) || !std::isfinite(handoff_stable_time_s_) ||
        !std::isfinite(offboard_entry_delay_s_) || !std::isfinite(handoff_max_horizontal_speed_) ||
        !std::isfinite(handoff_max_vertical_speed_) || !std::isfinite(handoff_max_position_drift_))
      throw std::invalid_argument("invalid OFFBOARD handoff limits");
    auto_manage_flight_=param("auto_manage_flight",false); auto_land_after_s_=param("auto_land_after_s",0.0);
    timing_flush_period_s_=param("timing_flush_period_ms",250.0)/1000.0;
    if (!(timing_flush_period_s_ > 0.0) || !std::isfinite(timing_flush_period_s_))
      throw std::invalid_argument("timing_flush_period_ms must be positive");
    configure_mavlink_rates_=param("configure_mavlink_rates",true);
    mavlink_rate_hz_=param("mavlink_rate_hz",100.0);
    eso_enabled_=param("eso_enabled",true); eso_activation_delay_s_=param("eso_activation_delay",3.0);
    rc_aux_channel_=param("rc_aux_channel",6); rc_aux_enable_threshold_=param("rc_aux_enable_threshold",0.5);
    rc_timeout_s_=param("rc_timeout_s",0.5); rc_deadzone_=param("rc_deadzone",0.08);
    rc_max_horizontal_speed_=param("rc_max_horizontal_speed",2.0); rc_max_vertical_speed_up_=param("rc_max_vertical_speed_up",3.0);
    rc_max_vertical_speed_down_=param("rc_max_vertical_speed_down",1.5); rc_max_horizontal_acceleration_=param("rc_max_horizontal_acceleration",2.0);
    rc_max_vertical_acceleration_up_=param("rc_max_vertical_acceleration_up",4.0); rc_max_vertical_acceleration_down_=param("rc_max_vertical_acceleration_down",3.0);
    rc_max_yaw_rate_=param("rc_max_yaw_rate",0.5); rc_max_yaw_acceleration_=param("rc_max_yaw_acceleration",1.0);
    rc_hold_max_horizontal_speed_=param("rc_hold_max_horizontal_speed",0.8); rc_hold_max_vertical_speed_=param("rc_hold_max_vertical_speed",0.6);
    rc_max_horizontal_position_lead_=param("rc_max_horizontal_position_lead",0.8); rc_max_vertical_position_lead_=param("rc_max_vertical_position_lead",0.4);
    thrust_min_=param("thrust_min",3.3278); thrust_max_=param("thrust_max",27.7314);
    body_rate_max_=Eigen::Vector3d(param("body_rate_max_x",1.0),param("body_rate_max_y",1.0),param("body_rate_max_z",1.0));
    if (horizon_ != kN) throw std::invalid_argument("horizon_steps must match generated solver horizon 30");
    hover_throttle_=param("hover_throttle",0.3653); throttle_min_=param("throttle_min",0.12); throttle_max_=param("throttle_max",1.0);
    const std::string root=param("flight_log_root",std::string("/home/cy2/nmpc_log"));
    const std::string explicit_path=param("timing_log_path",std::string());
    if (explicit_path.empty()) timing_path_=(std::filesystem::path(root)/session_stamp()/"nmpc_timing.csv").string(); else timing_path_=explicit_path;
    std::filesystem::create_directories(std::filesystem::path(timing_path_).parent_path());
    timing_.open(timing_path_); timing_ << "t_monotonic,sample_age_ms,rx_to_output_ms,solve_ms,position_x_ned,position_y_ned,position_z_ned,reference_x_ned,reference_y_ned,reference_z_ned,tracking_error_m,disturbance_x,disturbance_y,disturbance_z,command_thrust_n,command_roll_rate,command_pitch_rate,command_yaw_rate,normalized_throttle,rc_active,eso_active,estimator_ready,shadow_mode,setpoint_published\n";
    timing_queue_size_=static_cast<std::size_t>(param("timing_queue_size",4096));
    if (timing_queue_size_==0) throw std::invalid_argument("timing_queue_size must be positive");
    max_consecutive_solve_failures_=param("max_consecutive_solve_failures",3);
    if(max_consecutive_solve_failures_<1) throw std::invalid_argument("max_consecutive_solve_failures must be positive");
    fallback_hold_time_s_=param("fallback_hold_time",0.05);
    if(!(fallback_hold_time_s_>=0.0) || !std::isfinite(fallback_hold_time_s_))
      throw std::invalid_argument("fallback_hold_time must be non-negative");
    timing_thread_=std::thread(&Node::timing_worker,this);
    eso_=std::make_unique<VelocityLeso>(param("eso_bandwidth",3.0),param("eso_clamp",1.0),param("eso_innovation_limit",0.5));
    odom_sub_=nh_.subscribe("/mavros/local_position/odom",1,&Node::odom_callback,this);
    trajectory_sub_=nh_.subscribe("/nmpc/in/trajectory_setpoint",1,&Node::trajectory_callback,this);
    enable_sub_=nh_.subscribe("/nmpc/control_enabled",1,&Node::enable_callback,this);
    state_sub_=nh_.subscribe("/mavros/state",10,&Node::state_callback,this);
    estimator_sub_=nh_.subscribe("/mavros/estimator_status",10,&Node::estimator_callback,this);
    rc_sub_=nh_.subscribe("/mavros/manual_control/control",10,&Node::manual_control_callback,this);
    rc_input_sub_=nh_.subscribe("/mavros/rc/in",10,&Node::rc_input_callback,this);
    rates_pub_=nh_.advertise<mavros_msgs::AttitudeTarget>("/mavros/setpoint_raw/attitude",20);
    set_mode_client_=nh_.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
    arm_client_=nh_.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");
    message_interval_client_=nh_.serviceClient<mavros_msgs::MessageInterval>("/mavros/set_message_interval");
    timer_=nh_.createTimer(ros::Duration(dt_),&Node::heartbeat,this);
    phase_started_=stamp_seconds(Clock::now());
    ROS_INFO_STREAM("ROS1 C++ ESO+NMPC ready; shadow_mode=" << (shadow_mode_ ? "true" : "false")
                    << ", timing log: " << timing_path_);
  }
  ~Node() { stop_timing_logger(); }
 private:
  template<class T> T param(const std::string &name, const T &value) { T out; pnh_.param(name,out,value); return out; }
  void enable_callback(const std_msgs::Bool::ConstPtr &m) {
    const bool was_enabled = enabled_; enabled_=m->data;
    if (enabled_ && !was_enabled) {
      control_enable_time_=stamp_seconds(Clock::now());
      eso_active_=false;
      last_disturbance_.setZero();
      last_command_thrust_=mass_*gravity_;
      consecutive_solve_failures_=0;
      last_valid_command_available_=false;
      std::lock_guard<std::mutex> lock(mutex_);
      rc_mode_active_=false;
      rc_neutral_latched_=false;
      controller_.reset();
      handoff_anchor_valid_=false;
      handoff_stable_since_s_=0.0;
      handoff_stable_=false;
      offboard_entry_time_s_=0.0;
      if (!fault_active_ && !manual_reenable_required_) {
        phase_=WAIT;
        phase_started_=stamp_seconds(Clock::now());
      }
      ROS_INFO("NMPC control enabled; ESO activation delay started");
    }
    if (!enabled_ && was_enabled) {
      eso_active_=false;
      std::lock_guard<std::mutex> lock(mutex_);
      rc_mode_active_=false;
      rc_neutral_latched_=false;
      controller_.reset();
      consecutive_solve_failures_=0;
      last_valid_command_available_=false;
      fault_active_=false;
      manual_reenable_required_=false;
      recovery_ready_=false;
      offboard_exit_seen_=false;
      position_mode_requested_=false;
      handoff_anchor_valid_=false;
      handoff_stable_since_s_=0.0;
      handoff_stable_=false;
      offboard_entry_time_s_=0.0;
      phase_=WAIT;
      phase_started_=stamp_seconds(Clock::now());
      ROS_INFO("NMPC control disabled");
    }
  }
  void state_callback(const mavros_msgs::State::ConstPtr &m) {
    const double now = stamp_seconds(Clock::now());
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string previous_mode=fcu_mode_;
    fcu_connected_=m->connected; fcu_armed_=m->armed; fcu_mode_=m->mode; last_state_rx_=now;
    if (fault_active_ || manual_reenable_required_) {
      if (m->mode != offboard_mode_) offboard_exit_seen_=true;
      if (m->mode == offboard_mode_ && offboard_exit_seen_ && recovery_ready_) {
        fault_active_=false;
        manual_reenable_required_=false;
        recovery_ready_=false;
        offboard_exit_seen_=false;
        position_mode_requested_=false;
        phase_=m->armed ? FLIGHT : FINISHED;
        ROS_INFO("NMPC recovery acknowledged by manual OFFBOARD selection; control output resumed");
      }
    } else if (m->mode == offboard_mode_ && previous_mode != offboard_mode_ && enabled_) {
      if (phase_ == READY || phase_ == PRESTREAM || phase_ == OFFBOARD || phase_ == ARMING || phase_ == FLIGHT) {
        phase_=ENGAGING;
        offboard_entry_time_s_=now;
        ROS_INFO("OFFBOARD detected; waiting %.2f s before NMPC trajectory takeover", offboard_entry_delay_s_);
      }
    } else if (m->mode != offboard_mode_ && phase_ == FLIGHT) {
      phase_=READY;
      ROS_WARN("OFFBOARD left; NMPC trajectory output suspended until OFFBOARD is selected again");
    }
  }
  void estimator_callback(const mavros_msgs::EstimatorStatus::ConstPtr &m) {
    const bool horizontal_position=m->pos_horiz_rel_status_flag||m->pos_horiz_abs_status_flag;
    const bool vertical_position=m->pos_vert_abs_status_flag||m->pos_vert_agl_status_flag;
    std::lock_guard<std::mutex> lock(mutex_);
    // PX4 v1.16's legacy MAVLink const-pos bit is also asserted for
    // vehicle_at_rest, including a healthy disarmed vehicle on the ground.  Allow
    // that state for OFFBOARD pre-streaming, but reject const/fake position once
    // armed, when it indicates an unsafe loss of horizontal aiding.
    const bool const_pos_safe_for_prestream=m->const_pos_mode_status_flag&&!fcu_armed_;
    estimator_ready_=m->attitude_status_flag&&m->velocity_horiz_status_flag&&
      m->velocity_vert_status_flag&&horizontal_position&&vertical_position&&
      (!m->const_pos_mode_status_flag||const_pos_safe_for_prestream);
  }
  void manual_control_callback(const mavros_msgs::ManualControl::ConstPtr &m) {
    const double now=stamp_seconds(Clock::now());
    std::lock_guard<std::mutex> lock(mutex_);
    if (!std::isfinite(m->x) || !std::isfinite(m->y) ||
        !std::isfinite(m->z) || !std::isfinite(m->r)) {
      rc_valid_=false;
      return;
    }
    rc_sticks_={m->x,m->y,m->r,m->z}; rc_valid_=true; rc_rx_=now;
  }
  void rc_input_callback(const mavros_msgs::RCIn::ConstPtr &m) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (rc_aux_channel_ <= 0 || static_cast<std::size_t>(rc_aux_channel_) > m->channels.size()) {
      rc_aux_enabled_=false; return;
    }
    const double value=(static_cast<double>(m->channels[rc_aux_channel_-1])-1000.0)/1000.0;
    rc_aux_enabled_=std::isfinite(value) && value>=rc_aux_enable_threshold_;
  }

  bool build_rc_trajectory(const State &state, double dt, Trajectory &trajectory) {
    std::array<float,4> sticks; { std::lock_guard<std::mutex> lock(mutex_); sticks=rc_sticks_; }
    const double roll=apply_deadzone(sticks[0],rc_deadzone_), pitch=apply_deadzone(sticks[1],rc_deadzone_);
    const double yaw_stick=apply_deadzone(sticks[2],rc_deadzone_), throttle=apply_deadzone(sticks[3],rc_deadzone_);
    Eigen::Vector2d horizontal(pitch,roll); const double norm=horizontal.norm(); if(norm>1.0) horizontal/=norm;
    const double measured_yaw=std::atan2(2.0*(state[6]*state[9]+state[7]*state[8]),1.0-2.0*(state[8]*state[8]+state[9]*state[9]));
    const bool neutral=horizontal.norm()<1e-12 && std::abs(throttle)<1e-12 && std::abs(yaw_stick)<1e-12;
    if(!rc_mode_active_){rc_reference_position_=state.head<3>();rc_reference_velocity_.setZero();rc_reference_yaw_=measured_yaw;rc_reference_yaw_rate_=0;rc_neutral_latched_=neutral;rc_mode_active_=true;controller_.reset();}
    dt=std::clamp(dt,1e-3,0.1); Eigen::Vector3d a=Eigen::Vector3d::Zero();
    if(!(neutral&&rc_neutral_latched_)){
      if(!neutral&&rc_neutral_latched_){rc_reference_position_=state.head<3>();rc_reference_velocity_.setZero();rc_reference_yaw_=measured_yaw;rc_reference_yaw_rate_=0;rc_neutral_latched_=false;controller_.reset();}
      const double c=std::cos(rc_reference_yaw_),s=std::sin(rc_reference_yaw_);
      Eigen::Vector2d accel_xy(c*horizontal[0]-s*horizontal[1],s*horizontal[0]+c*horizontal[1]);
      if(neutral){const double speed=rc_reference_velocity_.head<2>().norm();if(speed>1e-12)accel_xy=-rc_max_horizontal_acceleration_*rc_reference_velocity_.head<2>()/speed;}
      else accel_xy*=rc_max_horizontal_acceleration_;
      a.head<2>()=accel_xy;
      if(std::abs(throttle)>1e-12)a[2]=-throttle*(throttle>=0?rc_max_vertical_acceleration_up_:rc_max_vertical_acceleration_down_);
      else if(std::abs(rc_reference_velocity_[2])>1e-12)
        a[2]=rc_reference_velocity_[2]>0.0?-rc_max_vertical_acceleration_down_:rc_max_vertical_acceleration_up_;
      rc_reference_velocity_+=a*dt; const double hs=rc_reference_velocity_.head<2>().norm();
      if(hs>rc_max_horizontal_speed_)rc_reference_velocity_.head<2>()*=rc_max_horizontal_speed_/hs;
      rc_reference_velocity_[2]=std::clamp(rc_reference_velocity_[2],-rc_max_vertical_speed_up_,rc_max_vertical_speed_down_);
      rc_reference_position_+=rc_reference_velocity_*dt;
      Eigen::Vector2d lead=rc_reference_position_.head<2>()-state.head<3>().head<2>(); const double ln=lead.norm();
      if(ln>rc_max_horizontal_position_lead_)rc_reference_position_.head<2>()=state.head<3>().head<2>()+rc_max_horizontal_position_lead_*lead/ln;
      rc_reference_position_[2]=std::clamp(rc_reference_position_[2],state[2]-rc_max_vertical_position_lead_,state[2]+rc_max_vertical_position_lead_);
      const double target_rate=yaw_stick*rc_max_yaw_rate_; rc_reference_yaw_rate_+=std::clamp(target_rate-rc_reference_yaw_rate_,-rc_max_yaw_acceleration_*dt,rc_max_yaw_acceleration_*dt); rc_reference_yaw_=wrap_angle(rc_reference_yaw_+rc_reference_yaw_rate_*dt);
      if(neutral&&rc_reference_velocity_.head<2>().norm()<=rc_hold_max_horizontal_speed_&&state.segment<3>(3).head<2>().norm()<=rc_hold_max_horizontal_speed_&&std::abs(rc_reference_velocity_[2])<=rc_hold_max_vertical_speed_&&std::abs(state[5])<=rc_hold_max_vertical_speed_){rc_reference_position_=state.head<3>();rc_reference_velocity_.setZero();rc_reference_yaw_=measured_yaw;rc_reference_yaw_rate_=0;rc_neutral_latched_=true;a.setZero();controller_.reset();}
    }
    trajectory=Trajectory{}; trajectory.sequence=++rc_sequence_; trajectory.sample_time=dt_; Eigen::Vector3d p=rc_reference_position_,v=rc_reference_velocity_; double yaw=rc_reference_yaw_;
    for(int i=0;i<kPoints;++i){for(int j=0;j<3;++j){trajectory.position[3*i+j]=p[j];trajectory.velocity[3*i+j]=v[j];trajectory.acceleration[3*i+j]=a[j];}trajectory.yaw[i]=yaw;if(i==kN)break;p+=v*dt_+0.5*a*dt_*dt_;v+=a*dt_;const double hs=v.head<2>().norm();if(hs>rc_max_horizontal_speed_)v.head<2>()*=rc_max_horizontal_speed_/hs;v[2]=std::clamp(v[2],-rc_max_vertical_speed_up_,rc_max_vertical_speed_down_);yaw=wrap_angle(yaw+rc_reference_yaw_rate_*dt_);}
    return true;
  }
  void build_hold_trajectory(const State &state, Trajectory &trajectory) const {
    trajectory = Trajectory{};
    trajectory.sample_time = dt_;
    const double yaw = std::atan2(2.0 * (state[6] * state[9] + state[7] * state[8]),
                                 1.0 - 2.0 * (state[8] * state[8] + state[9] * state[9]));
    for (int i = 0; i < kPoints; ++i) {
      for (int j = 0; j < 3; ++j) {
        trajectory.position[3 * i + j] = state[j];
        trajectory.velocity[3 * i + j] = 0.0;
        trajectory.acceleration[3 * i + j] = 0.0;
      }
      trajectory.yaw[i] = yaw;
    }
  }
  void update_handoff_stability(const State &state, double now) {
    if (fault_active_ || manual_reenable_required_ || phase_ == FLIGHT ||
        phase_ == ENGAGING || phase_ == LANDING || phase_ == FINISHED) return;
    const double horizontal_speed=state.segment<3>(3).head<2>().norm();
    const double vertical_speed=std::abs(state[5]);
    if (!handoff_anchor_valid_) {
      handoff_anchor_position_=state.head<3>();
      handoff_anchor_valid_=true;
      handoff_stable_since_s_=now;
      handoff_stable_=false;
      return;
    }
    const double drift=(state.head<3>()-handoff_anchor_position_).norm();
    const bool stable=estimator_ready_ && horizontal_speed<=handoff_max_horizontal_speed_ &&
      vertical_speed<=handoff_max_vertical_speed_ && drift<=handoff_max_position_drift_;
    if (!stable) {
      handoff_anchor_position_=state.head<3>();
      handoff_stable_since_s_=now;
      handoff_stable_=false;
      return;
    }
    if (now-handoff_stable_since_s_>=handoff_stable_time_s_) {
      if (!handoff_stable_) ROS_INFO("Position hold stable for %.2f s; starting OFFBOARD handoff", handoff_stable_time_s_);
      handoff_stable_=true;
    }
  }
  void trajectory_callback(const eso_nmpc_cpp::NmpcTrajectorySetpoint::ConstPtr &m) {
    if (!m->header.frame_id.empty() && m->header.frame_id != "ned") {
      ROS_WARN_THROTTLE(2.0, "Rejecting NMPC trajectory with frame_id='%s'; expected 'ned'",
                        m->header.frame_id.c_str());
      return;
    }
    if (m->points != kPoints || std::abs(m->sample_time-dt_)>1e-4 || m->position.size()!=3*kPoints ||
        m->velocity.size()!=3*kPoints || m->acceleration.size()!=3*kPoints || m->yaw.size()!=kPoints) return;
    for (int i = 0; i < kPoints; ++i) {
      if (!std::isfinite(m->yaw[i])) {
        ROS_WARN_THROTTLE(2.0, "Rejecting NMPC trajectory with non-finite yaw");
        return;
      }
      for (int j = 0; j < 3; ++j) {
        const std::size_t index = static_cast<std::size_t>(3 * i + j);
        if (!std::isfinite(m->position[index]) ||
            !std::isfinite(m->velocity[index]) ||
            !std::isfinite(m->acceleration[index])) {
          ROS_WARN_THROTTLE(2.0, "Rejecting NMPC trajectory with non-finite state sample");
          return;
        }
      }
    }
    Trajectory t; t.sequence=m->sequence; t.sample_time=m->sample_time;
    std::copy(m->position.begin(),m->position.end(),t.position.begin()); std::copy(m->velocity.begin(),m->velocity.end(),t.velocity.begin());
    std::copy(m->acceleration.begin(),m->acceleration.end(),t.acceleration.begin()); std::copy(m->yaw.begin(),m->yaw.end(),t.yaw.begin());
    std::lock_guard<std::mutex> lock(mutex_); trajectory_=t; trajectory_valid_=true; trajectory_rx_=stamp_seconds(Clock::now());
  }
  State state_from_odom(const nav_msgs::Odometry &m) const {
    State s; const auto &p=m.pose.pose.position; s[0]=p.y; s[1]=p.x; s[2]=-p.z;
    Eigen::Vector4d q_enu(m.pose.pose.orientation.w,m.pose.pose.orientation.x,m.pose.pose.orientation.y,m.pose.pose.orientation.z);
    const Eigen::Vector3d v_flu(m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.linear.z);
    const Eigen::Vector3d v_enu=rotate(q_enu,v_flu); s.segment<3>(3)=Eigen::Vector3d(v_enu.y(),v_enu.x(),-v_enu.z());
    const Eigen::Vector4d q=enu_flu_to_ned_frd(q_enu); for(int i=0;i<4;++i) s[6+i]=q[i];
    s[10]=m.twist.twist.angular.x; s[11]=-m.twist.twist.angular.y; s[12]=-m.twist.twist.angular.z; return s;
  }
  bool reference(const Trajectory &t,const Eigen::Vector4d &anchor,const Eigen::Vector3d &disturbance,States &x,Controls &u) const {
    std::array<Eigen::Vector4d,kPoints> q{};
    for(int i=0;i<kPoints;++i){ Eigen::Vector3d a(t.acceleration[3*i],t.acceleration[3*i+1],t.acceleration[3*i+2]);
      const double yaw=t.yaw[i]; Eigen::Vector3d bz(0,0,gravity_); bz+=disturbance-a; if(bz.norm()<1e-12)return false; bz.normalize();
      Eigen::Vector3d h(std::cos(yaw),std::sin(yaw),0), by=bz.cross(h); if(by.norm()<1e-12)by=bz.cross(Eigen::Vector3d(-std::sin(yaw),std::cos(yaw),0)); by.normalize();
      Eigen::Matrix3d r; r.col(2)=bz; r.col(1)=by; r.col(0)=by.cross(bz); q[i]=matrix_quaternion(r); if(i==0&&q[i].dot(anchor)<0)q[i]=-q[i]; if(i&&q[i].dot(q[i-1])<0)q[i]=-q[i];
      for(int j=0;j<3;++j){x(j,i)=t.position[3*i+j];x(3+j,i)=t.velocity[3*i+j];} for(int j=0;j<4;++j)x(6+j,i)=q[i][j];
      if(i<kN)u(0,i)=std::clamp(mass_*(Eigen::Vector3d(0,0,gravity_)+disturbance-a).norm(),thrust_min_,thrust_max_);
    }
    for(int i=0;i<kN;++i){Eigen::Vector4d qc(q[i][0],-q[i][1],-q[i][2],-q[i][3]); Eigen::Vector4d d=normalize(multiply(qc,q[i+1])); if(d[0]<0)d=-d; const double n=d.tail<3>().norm(); Eigen::Vector3d r=Eigen::Vector3d::Zero(); if(n>=1e-12) r=2*std::atan2(n,d[0])*d.tail<3>()/(dt_*n); for(int j=0;j<3;++j)u(j+1,i)=std::clamp(r[j],-body_rate_max_[j],body_rate_max_[j]); }
    for(int i=0;i<kPoints;++i)for(int j=0;j<3;++j)x(10+j,i)=i<kN?u(j+1,i):u(j+1,kN-1);
    for(int i=0;i<kN;++i){u(0,i)=std::clamp(u(0,i),thrust_min_,thrust_max_);}
    return true;
  }
  void odom_callback(const nav_msgs::Odometry::ConstPtr &m) {
    if (!enabled_) return;
    const auto rx=Clock::now();
    State s=state_from_odom(*m);
    if (!s.allFinite()) return;
    update_handoff_stability(s, stamp_seconds(rx));
    Trajectory t;
    bool external_trajectory_valid=false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      external_trajectory_valid=trajectory_valid_ && stamp_seconds(rx)-trajectory_rx_<=timeout_;
      if(external_trajectory_valid) t=trajectory_;
    }
    const double receive=stamp_seconds(rx);
    bool use_rc=false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const bool rc_fresh=rc_valid_ && receive-rc_rx_<=rc_timeout_s_;
      const bool aux=rc_aux_channel_>0 && rc_aux_enabled_;
      use_rc=aux && (rc_fresh || rc_mode_active_);
      if(aux && !use_rc) {
        ROS_WARN_THROTTLE(2.0, "RC-NMPC selected but no valid ManualControl input");
        return;
      }
      if(use_rc && !rc_fresh) rc_sticks_={0,0,0,0};
      if(!use_rc && rc_mode_active_) { rc_mode_active_=false; rc_neutral_latched_=false; controller_.reset(); }
    }
    const bool handoff_phase = phase_ != FLIGHT && phase_ != LANDING && phase_ != FINISHED;
    if(!use_rc && !external_trajectory_valid && !fault_active_ && !manual_reenable_required_ && !handoff_phase) return;
    if(!use_rc && (!external_trajectory_valid || handoff_phase)) build_hold_trajectory(s,t);
    // Match ROS2's sample-time semantics: use the sensor timestamp for ESO
    // and RC-reference integration, not host callback scheduling jitter.
    const double sample_stamp = m->header.stamp.isZero() ? 0.0 : m->header.stamp.toSec();
    double dt=dt_;
    if(sample_stamp>0.0 && last_odom_sample_stamp_>0.0 && sample_stamp>last_odom_sample_stamp_)
      dt=sample_stamp-last_odom_sample_stamp_;
    if(sample_stamp>0.0) last_odom_sample_stamp_=sample_stamp;
    Eigen::Vector3d disturbance=last_disturbance_;
    if(eso_enabled_) {
      const Eigen::Vector4d q(s[6],s[7],s[8],s[9]);
      const Eigen::Matrix3d rot=Eigen::Quaterniond(q[0],q[1],q[2],q[3]).toRotationMatrix();
      const Eigen::Vector3d model=Eigen::Vector3d(0,0,gravity_)-(last_command_thrust_/mass_)*rot.col(2);
      if(receive-control_enable_time_<eso_activation_delay_s_) { eso_->reset(s.segment<3>(3),disturbance); eso_active_=false; }
      else if(!eso_active_) { eso_->reset(s.segment<3>(3),disturbance); eso_active_=true; }
      else { disturbance=eso_->update(s.segment<3>(3),model,dt); last_disturbance_=disturbance; }
    }
    if(use_rc && !build_rc_trajectory(s,dt,t)) return;
    States refs=States::Zero(); Controls ff=Controls::Zero();
    const Eigen::Vector4d anchor=s.segment<4>(6);
    if(!reference(t,anchor,disturbance,refs,ff)) return;
    Eigen::Vector4d cmd; const auto solve0=Clock::now();
    const bool had_warm_start=controller_.warm_start_available();
    bool solve_success=controller_.solve(s,refs,ff,disturbance,cmd);
    if(!solve_success && had_warm_start) {
      ROS_WARN_THROTTLE(1.0, "acados warm-start solve failed; retrying cold start");
      controller_.reset();
      solve_success=controller_.solve(s,refs,ff,disturbance,cmd);
    }
    if(!solve_success) {
      ++consecutive_solve_failures_;
      const double now_s=stamp_seconds(Clock::now());
      const bool can_hold_last=last_valid_command_available_ &&
        now_s-last_valid_command_time_s_<=fallback_hold_time_s_;
      if(can_hold_last && output_allowed()) publish(last_valid_command_);
      ROS_WARN_THROTTLE(1.0, "acados solve failed (%d/%d); bounded fallback=%s",
                        consecutive_solve_failures_,max_consecutive_solve_failures_,can_hold_last?"last command":"none");
      if(consecutive_solve_failures_>=max_consecutive_solve_failures_) {
        if (!fault_active_) {
          fault_active_=true;
          manual_reenable_required_=true;
          recovery_ready_=false;
          offboard_exit_seen_=false;
          position_mode_requested_=false;
          recovery_since_s_=0.0;
          phase_=FAULT_HOLD;
          ROS_ERROR("NMPC fault latched; stopping setpoints and requesting PX4 position mode");
        }
        controller_.reset();
        eso_active_=false;
        last_valid_command_available_=false;
      }
      return;
    }
    consecutive_solve_failures_=0;
    const auto solve1=Clock::now();
    last_command_thrust_=cmd[0];
    last_valid_command_=cmd;
    last_valid_command_time_s_=stamp_seconds(solve1);
    last_valid_command_available_=true;
    if (fault_active_) {
      if (recovery_since_s_ <= 0.0) recovery_since_s_=stamp_seconds(solve1);
      if (!recovery_ready_ && stamp_seconds(solve1)-recovery_since_s_ >= solver_recovery_time_s_) {
        recovery_ready_=true;
        ROS_INFO("NMPC solver healthy for %.2f s; waiting for manual OFFBOARD re-entry", solver_recovery_time_s_);
      }
    }
    if (output_allowed()) {
      publish(cmd);
    }
    const auto pub=Clock::now();
    last_cmd_=cmd;
    const double referr=(s.head<3>()-refs.block<3,1>(0,0)).norm();
    const double sample_age_ms=m->header.stamp.isZero()?std::numeric_limits<double>::quiet_NaN():1e3*(ros::Time::now()-m->header.stamp).toSec();
    const double normalized_throttle=std::clamp(hover_throttle_*cmd[0]/(mass_*gravity_),throttle_min_,throttle_max_);
    std::ostringstream row;
    row<<std::setprecision(12)<<stamp_seconds(pub)<<","<<sample_age_ms<<","<<1e3*std::chrono::duration<double>(pub-rx).count()<<","<<1e3*std::chrono::duration<double>(solve1-solve0).count()<<","<<s[0]<<","<<s[1]<<","<<s[2]<<","<<refs(0,0)<<","<<refs(1,0)<<","<<refs(2,0)<<","<<referr<<","<<disturbance[0]<<","<<disturbance[1]<<","<<disturbance[2]<<","<<cmd[0]<<","<<cmd[1]<<","<<cmd[2]<<","<<cmd[3]<<","<<normalized_throttle<<","<<(use_rc?1:0)<<","<<(eso_active_?1:0)<<","<<(estimator_ready_?1:0)<<","<<(shadow_mode_?1:0)<<","<<(output_allowed()&&!shadow_mode_?1:0)<<"\n";
    enqueue_timing(row.str());
  }
  void enqueue_timing(std::string row) {
    std::lock_guard<std::mutex> lock(timing_mutex_);
    if(timing_queue_.size()>=timing_queue_size_) { timing_queue_.pop_front(); ++timing_dropped_; }
    timing_queue_.push_back(std::move(row));
    timing_cv_.notify_one();
  }
  void stop_timing_logger() {
    if(!timing_thread_.joinable()) return;
    {
      std::lock_guard<std::mutex> lock(timing_mutex_);
      timing_stop_=true;
    }
    timing_cv_.notify_one();
    timing_thread_.join();
  }
  void timing_worker() {
    const auto flush_period=std::chrono::duration<double>(timing_flush_period_s_);
    auto next_flush=Clock::now()+std::chrono::duration_cast<Clock::duration>(flush_period);
    while(true) {
      std::deque<std::string> pending;
      {
        std::unique_lock<std::mutex> lock(timing_mutex_);
        timing_cv_.wait_until(lock,next_flush,[this]{return timing_stop_||!timing_queue_.empty();});
        pending.swap(timing_queue_);
        if(timing_stop_&&pending.empty()) break;
      }
      for(const auto &row:pending) timing_<<row;
      const auto now=Clock::now();
      if(now>=next_flush) { timing_.flush(); next_flush=now+std::chrono::duration_cast<Clock::duration>(flush_period); }
    }
    timing_.flush();
    if(timing_dropped_>0) ROS_WARN("Dropped %zu timing records because the async log queue was full",timing_dropped_);
  }
  bool output_allowed() const {
    if (!enabled_ || fault_active_ || manual_reenable_required_ || shadow_mode_) return false;
    if (fcu_mode_ == offboard_mode_) {
      if (phase_ == FLIGHT) {
        return offboard_entry_time_s_ <= 0.0 ||
          stamp_seconds(Clock::now())-offboard_entry_time_s_ >= offboard_entry_delay_s_;
      }
      return phase_ == ENGAGING || phase_ == ARMING;
    }
    return phase_ == STABILIZE || phase_ == PRESTREAM || phase_ == READY ||
      (auto_manage_flight_ && phase_ == OFFBOARD);
  }
  void request_position_mode() {
    if (shadow_mode_ || !auto_manage_flight_ || !fcu_connected_ || !fcu_armed_ ||
        fcu_mode_ == position_mode_) return;
    const double now=stamp_seconds(Clock::now());
    if (now-last_service_call_<0.5) return;
    mavros_msgs::SetMode srv; srv.request.custom_mode=position_mode_;
    last_service_call_=now;
    if (set_mode_client_.exists() && set_mode_client_.call(srv) && srv.response.mode_sent) {
      position_mode_requested_=true;
      ROS_WARN("NMPC fault: requested PX4 %s mode; waiting for manual OFFBOARD re-entry", position_mode_.c_str());
    } else {
      ROS_ERROR_THROTTLE(1.0, "NMPC fault: failed to request PX4 %s mode", position_mode_.c_str());
    }
  }
  void request_land() {
    if (shadow_mode_ || !auto_manage_flight_ || !fcu_armed_ || phase_ == LANDING || phase_ == FINISHED) return;
    mavros_msgs::SetMode srv; srv.request.custom_mode="AUTO.LAND";
    if (set_mode_client_.exists() && set_mode_client_.call(srv) && srv.response.mode_sent) {
      phase_=LANDING; phase_started_=stamp_seconds(Clock::now());
      ROS_WARN("NMPC requested AUTO.LAND");
    }
  }
  void flight_tick() {
    if(shadow_mode_) return;
    if (fault_active_ || manual_reenable_required_ || phase_ == FAULT_HOLD) {
      if (auto_manage_flight_) request_position_mode();
      return;
    }
    if(!enabled_) return;
    const double now=stamp_seconds(Clock::now());
    bool connected,armed; std::string mode; double state_rx;
    { std::lock_guard<std::mutex> lock(mutex_); connected=fcu_connected_; armed=fcu_armed_; mode=fcu_mode_; state_rx=last_state_rx_; }
    if(!connected || state_rx<=0.0 || now-state_rx>state_timeout_s_) return;
    if(phase_==WAIT) phase_=STABILIZE;
    if(phase_==STABILIZE && handoff_stable_) {
      phase_=PRESTREAM; phase_started_=now;
      ROS_INFO("NMPC safe-hold prestream started");
    }
    if(phase_==PRESTREAM && now-phase_started_>=prestream_s_) {
      phase_=auto_manage_flight_ ? OFFBOARD : READY;
      phase_started_=now;
      if (!auto_manage_flight_) ROS_INFO("NMPC prestream ready; waiting for manual OFFBOARD selection");
    }
    if (phase_==READY && mode==offboard_mode_) {
      phase_=ENGAGING; offboard_entry_time_s_=now;
      ROS_INFO("OFFBOARD detected; waiting %.2f s before NMPC trajectory takeover", offboard_entry_delay_s_);
    }
    if(now-last_service_call_<0.5) return;
    if(phase_==OFFBOARD) {
      mavros_msgs::SetMode srv; srv.request.custom_mode=offboard_mode_;
      set_mode_client_.call(srv); last_service_call_=now;
      if(mode==offboard_mode_) { phase_=ENGAGING; offboard_entry_time_s_=now; phase_started_=now; }
      else if(now-phase_started_>service_timeout_s_) { ROS_ERROR("Timed out entering %s",offboard_mode_.c_str()); request_land(); if(phase_!=LANDING) phase_=FINISHED; }
    } else if(phase_==ENGAGING && mode==offboard_mode_) {
      if (now-offboard_entry_time_s_>=offboard_entry_delay_s_) {
        if (armed) { phase_=FLIGHT; phase_started_=now; ROS_INFO("NMPC flight started"); }
        else phase_=ARMING;
      }
    } else if(phase_==ARMING) {
      if (!auto_manage_flight_) {
        if (armed) { phase_=FLIGHT; phase_started_=now; ROS_INFO("Manual OFFBOARD handoff complete; NMPC flight started"); }
        return;
      }
      mavros_msgs::CommandBool srv; srv.request.value=true;
      arm_client_.call(srv); last_service_call_=now;
      if(armed) { phase_=FLIGHT; phase_started_=now; ROS_INFO("NMPC flight started"); }
      else if(now-phase_started_>service_timeout_s_) { ROS_ERROR("Timed out arming"); phase_=FINISHED; }
    } else if(phase_==FLIGHT && auto_land_after_s_>0.0 && now-phase_started_>=auto_land_after_s_) {
      request_land();
    } else if(phase_==LANDING && !armed) {
      phase_=FINISHED; ROS_INFO("NMPC landing complete");
    }
  }
  void publish_hold(){Eigen::Vector4d c(mass_*gravity_,0,0,0);publish(c);}
  void publish(const Eigen::Vector4d &c){
    if(shadow_mode_)return;
    mavros_msgs::AttitudeTarget m;
    m.header.stamp=ros::Time::now();
    m.type_mask=mavros_msgs::AttitudeTarget::IGNORE_ATTITUDE;
    // The controller works in PX4 aircraft FRD. MAVROS expects ROS base_link FLU
    // here and performs FLU -> FRD before emitting SET_ATTITUDE_TARGET, so apply
    // the inverse conversion exactly once at this boundary.
    m.body_rate.x=c[1];
    m.body_rate.y=-c[2];
    m.body_rate.z=-c[3];
    m.thrust=std::clamp(hover_throttle_*c[0]/(mass_*gravity_),throttle_min_,throttle_max_);
    rates_pub_.publish(m);
  }
  bool request_message_rate(uint32_t id) {
    mavros_msgs::MessageInterval srv; srv.request.message_id=id; srv.request.message_rate=mavlink_rate_hz_;
    return message_interval_client_.call(srv) && srv.response.success;
  }
  void configure_rates() {
    if(!configure_mavlink_rates_ || mavlink_rates_configured_ || !fcu_connected_) return;
    const double now=stamp_seconds(Clock::now());
    if(now-last_rate_request_<1.0) return;
    last_rate_request_=now;
    // ATTITUDE_QUATERNION, HIGHRES_IMU and LOCAL_POSITION_NED.
    mavlink_rates_configured_=request_message_rate(31)&&request_message_rate(105)&&request_message_rate(32);
    if(mavlink_rates_configured_) ROS_INFO("Requested MAVLink IMU, attitude and fused odometry at %.1f Hz",mavlink_rate_hz_);
    else ROS_WARN_THROTTLE(5.0,"Unable to set all required MAVLink streams to %.1f Hz; retrying",mavlink_rate_hz_);
  }
  void heartbeat(const ros::TimerEvent&){configure_rates();flight_tick();}
  enum FlightPhase { WAIT, STABILIZE, PRESTREAM, READY, OFFBOARD, ENGAGING, ARMING, FLIGHT, LANDING, FINISHED, FAULT_HOLD };
  ros::NodeHandle nh_,pnh_; ros::Subscriber odom_sub_,trajectory_sub_,enable_sub_,state_sub_,estimator_sub_,rc_sub_,rc_input_sub_; ros::Publisher rates_pub_; ros::ServiceClient set_mode_client_,arm_client_,message_interval_client_; ros::Timer timer_; std::mutex mutex_;
  Trajectory trajectory_{}; bool trajectory_valid_{false},enabled_{false},shadow_mode_{true};
  bool fcu_connected_{false},fcu_armed_{false},estimator_ready_{false},rc_valid_{false},rc_aux_enabled_{false},rc_mode_active_{false},rc_neutral_latched_{false};
  double trajectory_rx_{0},last_state_rx_{0},last_odom_sample_stamp_{0},mass_{},gravity_{},dt_{},timeout_{}; int horizon_{};
  double hover_throttle_{},throttle_min_{},throttle_max_{},last_command_thrust_{0}; double thrust_min_{3.3278},thrust_max_{27.7314}; Eigen::Vector3d body_rate_max_{Eigen::Vector3d::Ones()};
  bool eso_enabled_{true},eso_active_{false}; double eso_activation_delay_s_{3.0},control_enable_time_{0}; Eigen::Vector3d last_disturbance_{Eigen::Vector3d::Zero()}; std::unique_ptr<VelocityLeso> eso_;
  std::array<float,4> rc_sticks_{}; double rc_rx_{0},rc_timeout_s_{0.5},rc_deadzone_{0.08},rc_aux_enable_threshold_{0.5}; int rc_aux_channel_{6};
  double rc_max_horizontal_speed_{2},rc_max_vertical_speed_up_{3},rc_max_vertical_speed_down_{1.5},rc_max_horizontal_acceleration_{2},rc_max_vertical_acceleration_up_{4},rc_max_vertical_acceleration_down_{3},rc_max_yaw_rate_{0.5},rc_max_yaw_acceleration_{1},rc_hold_max_horizontal_speed_{0.8},rc_hold_max_vertical_speed_{0.6},rc_max_horizontal_position_lead_{0.8},rc_max_vertical_position_lead_{0.4};
  Eigen::Vector3d rc_reference_position_{Eigen::Vector3d::Zero()},rc_reference_velocity_{Eigen::Vector3d::Zero()}; double rc_reference_yaw_{0},rc_reference_yaw_rate_{0}; uint32_t rc_sequence_{0};
  bool auto_manage_flight_{false}; double prestream_s_{1.5},service_timeout_s_{8},state_timeout_s_{1},auto_land_after_s_{0},phase_started_{0},last_service_call_{0}; std::string offboard_mode_,position_mode_; FlightPhase phase_{WAIT}; std::string fcu_mode_;
  bool configure_mavlink_rates_{true},mavlink_rates_configured_{false}; double mavlink_rate_hz_{100.0},last_rate_request_{0};
  AcadosController controller_; Eigen::Vector4d last_cmd_{Eigen::Vector4d::Zero()}; std::string timing_path_; std::ofstream timing_;
  double timing_flush_period_s_{0.25}; std::size_t timing_queue_size_{4096},timing_dropped_{0};
  std::mutex timing_mutex_; std::condition_variable timing_cv_; std::deque<std::string> timing_queue_;
  bool timing_stop_{false}; std::thread timing_thread_;
  int max_consecutive_solve_failures_{3},consecutive_solve_failures_{0};
  double fallback_hold_time_s_{0.05},solver_recovery_time_s_{2.0},recovery_since_s_{0.0},last_valid_command_time_s_{0.0};
  double handoff_stable_time_s_{1.0},offboard_entry_delay_s_{0.2},handoff_max_horizontal_speed_{0.15},handoff_max_vertical_speed_{0.10},handoff_max_position_drift_{0.25},handoff_stable_since_s_{0.0},offboard_entry_time_s_{0.0};
  Eigen::Vector3d handoff_anchor_position_{Eigen::Vector3d::Zero()};
  bool fault_active_{false},manual_reenable_required_{false},recovery_ready_{false},offboard_exit_seen_{false},position_mode_requested_{false},handoff_anchor_valid_{false},handoff_stable_{false};
  Eigen::Vector4d last_valid_command_{Eigen::Vector4d::Zero()};
  bool last_valid_command_available_{false};
};
}  // namespace eso_nmpc_cpp

int main(int argc,char **argv){ros::init(argc,argv,"eso_nmpc_cpp");try{eso_nmpc_cpp::Node node;ros::spin();}catch(const std::exception &e){ROS_FATAL("eso_nmpc_cpp: %s",e.what());return 1;}return 0;}
