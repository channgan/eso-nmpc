#include "eso_nmpc_node/eso_nmpc_node.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <acados_c/ocp_nlp_interface.h>
#include <acados/utils/types.h>
#include "acados_solver_config.hpp"
#include ESO_NMPC_ACADOS_HEADER

using namespace std::chrono_literals;

namespace eso_nmpc_node
{

namespace
{
constexpr int kGeneratedN = ESO_NMPC_GENERATED_N;
constexpr double kEpsilon = 1.0e-12;
using State = Eigen::Matrix<double, kNx, 1>;
using States = Eigen::Matrix<double, kNx, kMaxPoints>;
using Controls = Eigen::Matrix<double, kNu, kMaxPoints - 1>;

Eigen::Vector4d normalize_quaternion(Eigen::Vector4d q)
{
  const double norm = q.norm();
  if (norm < kEpsilon) {
    return Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
  }
  return q / norm;
}

Eigen::Matrix3d quaternion_to_rotation(const Eigen::Vector4d & q_in)
{
  const Eigen::Vector4d q = normalize_quaternion(q_in);
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  Eigen::Matrix3d r;
  r << 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
    2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
    2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y);
  return r;
}

Eigen::Vector4d rotation_to_quaternion(const Eigen::Matrix3d & r)
{
  Eigen::Vector4d q;
  const double trace = r.trace();
  if (trace > 0.0) {
    const double s = 2.0 * std::sqrt(trace + 1.0);
    q << 0.25 * s, (r(2, 1) - r(1, 2)) / s,
      (r(0, 2) - r(2, 0)) / s, (r(1, 0) - r(0, 1)) / s;
  } else {
    int i = 0;
    if (r(1, 1) > r(0, 0)) i = 1;
    if (r(2, 2) > r(i, i)) i = 2;
    if (i == 0) {
      const double s = 2.0 * std::sqrt(std::max(0.0, 1.0 + r(0, 0) - r(1, 1) - r(2, 2)));
      q << (r(2, 1) - r(1, 2)) / s, 0.25 * s,
        (r(0, 1) + r(1, 0)) / s, (r(0, 2) + r(2, 0)) / s;
    } else if (i == 1) {
      const double s = 2.0 * std::sqrt(std::max(0.0, 1.0 + r(1, 1) - r(0, 0) - r(2, 2)));
      q << (r(0, 2) - r(2, 0)) / s, (r(0, 1) + r(1, 0)) / s,
        0.25 * s, (r(1, 2) + r(2, 1)) / s;
    } else {
      const double s = 2.0 * std::sqrt(std::max(0.0, 1.0 + r(2, 2) - r(0, 0) - r(1, 1)));
      q << (r(1, 0) - r(0, 1)) / s, (r(0, 2) + r(2, 0)) / s,
        (r(1, 2) + r(2, 1)) / s, 0.25 * s;
    }
  }
  return normalize_quaternion(q);
}

Eigen::Vector3d quaternion_delta_rate(const Eigen::Vector4d & current,
                                      Eigen::Vector4d next, double dt)
{
  Eigen::Vector4d q = normalize_quaternion(current);
  next = normalize_quaternion(next);
  if (q.dot(next) < 0.0) next = -next;
  // conjugate(q) * next, scalar first Hamilton product
  const Eigen::Vector4d qc(q[0], -q[1], -q[2], -q[3]);
  Eigen::Vector4d d;
  d[0] = qc[0] * next[0] - qc[1] * next[1] - qc[2] * next[2] - qc[3] * next[3];
  d[1] = qc[0] * next[1] + qc[1] * next[0] + qc[2] * next[3] - qc[3] * next[2];
  d[2] = qc[0] * next[2] - qc[1] * next[3] + qc[2] * next[0] + qc[3] * next[1];
  d[3] = qc[0] * next[3] + qc[1] * next[2] - qc[2] * next[1] + qc[3] * next[0];
  d = normalize_quaternion(d);
  if (d[0] < 0.0) d = -d;
  const Eigen::Vector3d v = d.tail<3>();
  const double norm = v.norm();
  if (norm < kEpsilon || dt <= 0.0) return Eigen::Vector3d::Zero();
  const double angle = 2.0 * std::atan2(norm, std::clamp(d[0], -1.0, 1.0));
  return angle / dt * v / norm;
}

double apply_deadzone(double value, double deadzone)
{
  if (!std::isfinite(value)) return 0.0;
  value = std::clamp(value, -1.0, 1.0);
  const double magnitude = std::abs(value);
  if (magnitude <= deadzone) return 0.0;
  return std::copysign((magnitude - deadzone) / (1.0 - deadzone), value);
}

double wrap_angle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

double aux_value(const px4_msgs::msg::ManualControlSetpoint & message, int channel)
{
  switch (channel) {
    case 1: return message.aux1;
    case 2: return message.aux2;
    case 3: return message.aux3;
    case 4: return message.aux4;
    case 5: return message.aux5;
    case 6: return message.aux6;
    default: return -1.0;
  }
}

std::string default_flight_log_root()
{
  const char * repository_root = std::getenv("ESO_NMPC_ROOT");
  if (repository_root != nullptr && repository_root[0] != '\0') {
    return (std::filesystem::path(repository_root) / "nmpc" / "logs").string();
  }
  return (std::filesystem::current_path() / "nmpc" / "logs").string();
}

std::string flight_session_stamp()
{
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm local_time{};
#if defined(_WIN32)
  localtime_s(&local_time, &now_time);
#else
  localtime_r(&now_time, &local_time);
#endif
  std::ostringstream stamp;
  const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
    now.time_since_epoch()).count() % 1000;
  stamp << std::put_time(&local_time, "%Y%m%d_%H%M%S") << '_'
        << std::setfill('0') << std::setw(3) << milliseconds;
  return stamp.str();
}

}  // namespace

VelocityLeso::VelocityLeso(double bandwidth, double clamp, double innovation_limit)
: beta1_(2.0 * bandwidth), beta2_(bandwidth * bandwidth), clamp_(clamp),
  innovation_limit_(innovation_limit)
{
  if (!(bandwidth > 0.0) || !(clamp > 0.0) || !(innovation_limit > 0.0)) {
    throw std::invalid_argument("ESO parameters must be positive");
  }
}

void VelocityLeso::reset(const Eigen::Vector3d & velocity, const Eigen::Vector3d & disturbance)
{
  velocity_hat_ = velocity;
  disturbance_hat_ = disturbance.cwiseMax(-clamp_).cwiseMin(clamp_);
  initialized_ = true;
}

Eigen::Vector3d VelocityLeso::update(const Eigen::Vector3d & velocity,
                                     const Eigen::Vector3d & model_acceleration, double dt)
{
  if (!(dt > 0.0) || !std::isfinite(dt)) return disturbance_hat_;
  if (!initialized_) {
    reset(velocity, Eigen::Vector3d::Zero());
    return disturbance_hat_;
  }
  const double step = std::min(dt, 0.1);
  Eigen::Vector3d innovation = velocity - velocity_hat_;
  for (int i = 0; i < 3; ++i) innovation[i] = std::clamp(innovation[i], -innovation_limit_, innovation_limit_);
  velocity_hat_ += step * (model_acceleration + disturbance_hat_ + beta1_ * innovation);
  disturbance_hat_ += step * (beta2_ * innovation);
  disturbance_hat_ = disturbance_hat_.cwiseMax(-clamp_).cwiseMin(clamp_);
  return disturbance_hat_;
}

struct AcadosController::Impl
{
  using Capsule = ESO_NMPC_SOLVER_SYMBOL(ESO_NMPC_ACADOS_HASH, _solver_capsule);
  Capsule * capsule{nullptr};
  ocp_nlp_config * config{nullptr};
  ocp_nlp_dims * dims{nullptr};
  ocp_nlp_in * in{nullptr};
  ocp_nlp_out * out{nullptr};
  int n{kGeneratedN};
  double mass{0.0};
  double gravity{0.0};
  double sample_time{0.01};
  double thrust_min{0.0};
  double thrust_max{0.0};
  Eigen::Vector3d body_rate_max{Eigen::Vector3d::Ones()};
  bool warm_start{true};
  bool have_last{false};
  std::array<double, kNx * (kGeneratedN + 1)> last_x{};
  std::array<double, kNu * kGeneratedN> last_u{};
  AcadosController::Timing timing{};
  double previous_time_qpscaling_ms{0.0};
  double previous_time_sim_ms{0.0};
};

AcadosController::AcadosController(double mass, double gravity, double rate_tau,
                                   double sample_time, int horizon, double thrust_min,
                                   double thrust_max, const Eigen::Vector3d & body_rate_max,
                                   bool warm_start)
: impl_(std::make_unique<Impl>())
{
  (void)rate_tau;  // rate_tau is compiled into the generated formulation.
  if (horizon != kGeneratedN) {
    throw std::invalid_argument("C++ node horizon must match generated acados horizon (30)");
  }
  impl_->mass = mass;
  impl_->gravity = gravity;
  impl_->sample_time = sample_time;
  impl_->thrust_min = thrust_min;
  impl_->thrust_max = thrust_max;
  impl_->body_rate_max = body_rate_max;
  impl_->warm_start = warm_start;
  impl_->capsule = ESO_NMPC_SOLVER_SYMBOL(ESO_NMPC_ACADOS_HASH, _acados_create_capsule)();
  std::vector<double> time_steps(kGeneratedN, sample_time);
  if (impl_->capsule == nullptr ||
      ESO_NMPC_SOLVER_SYMBOL(ESO_NMPC_ACADOS_HASH, _acados_create_with_discretization)(
        impl_->capsule, kGeneratedN, time_steps.data()) != ACADOS_SUCCESS) {
    throw std::runtime_error("failed to create generated acados solver");
  }
  impl_->config = ESO_NMPC_SOLVER_SYMBOL(ESO_NMPC_ACADOS_HASH, _acados_get_nlp_config)(impl_->capsule);
  impl_->dims = ESO_NMPC_SOLVER_SYMBOL(ESO_NMPC_ACADOS_HASH, _acados_get_nlp_dims)(impl_->capsule);
  impl_->in = ESO_NMPC_SOLVER_SYMBOL(ESO_NMPC_ACADOS_HASH, _acados_get_nlp_in)(impl_->capsule);
  impl_->out = ESO_NMPC_SOLVER_SYMBOL(ESO_NMPC_ACADOS_HASH, _acados_get_nlp_out)(impl_->capsule);
}

AcadosController::~AcadosController()
{
  if (impl_ && impl_->capsule) {
    ESO_NMPC_SOLVER_SYMBOL(ESO_NMPC_ACADOS_HASH, _acados_free)(impl_->capsule);
    ESO_NMPC_SOLVER_SYMBOL(ESO_NMPC_ACADOS_HASH, _acados_free_capsule)(impl_->capsule);
  }
}

bool AcadosController::solve(const State & state, const States & reference_states,
                             const Controls & feedforward, int points,
                             const Eigen::Vector3d & disturbance,
                             Eigen::Matrix<double, kNu, 1> & command)
{
  const auto wrapper_started = std::chrono::steady_clock::now();
  if (points != kGeneratedN + 1) return false;
  std::array<double, kNx * (kGeneratedN + 1)> x_guess{};
  std::array<double, kNu * kGeneratedN> u_guess{};
  for (int i = 0; i <= kGeneratedN; ++i) {
    int source = i;
    if (impl_->warm_start && impl_->have_last) source = std::min(i + 1, kGeneratedN);
    for (int j = 0; j < kNx; ++j) {
      x_guess[i * kNx + j] = (impl_->warm_start && impl_->have_last) ?
        impl_->last_x[source * kNx + j] : reference_states(j, i);
    }
  }
  for (int i = 0; i < kGeneratedN; ++i) {
    int source = i;
    if (impl_->warm_start && impl_->have_last) source = std::min(i + 1, kGeneratedN - 1);
    for (int j = 0; j < kNu; ++j) {
      u_guess[i * kNu + j] = (impl_->warm_start && impl_->have_last) ?
        impl_->last_u[source * kNu + j] : feedforward(j, i);
    }
  }
  for (int j = 0; j < kNx; ++j) x_guess[j] = state[j];
  for (int i = 0; i < kGeneratedN; ++i) {
    ocp_nlp_out_set(impl_->config, impl_->dims, impl_->out, impl_->in, i, "x", &x_guess[i * kNx]);
    ocp_nlp_out_set(impl_->config, impl_->dims, impl_->out, impl_->in, i, "u", &u_guess[i * kNu]);
  }
  ocp_nlp_out_set(impl_->config, impl_->dims, impl_->out, impl_->in, kGeneratedN, "x",
                  &x_guess[kGeneratedN * kNx]);
  std::array<double, kNx> x0{};
  std::copy(state.data(), state.data() + kNx, x0.data());
  ocp_nlp_constraints_model_set(impl_->config, impl_->dims, impl_->in, impl_->out, 0, "lbx", x0.data());
  ocp_nlp_constraints_model_set(impl_->config, impl_->dims, impl_->in, impl_->out, 0, "ubx", x0.data());

  std::array<double, kNp> parameters{};
  for (int i = 0; i < kGeneratedN; ++i) {
    parameters[0] = disturbance[0]; parameters[1] = disturbance[1]; parameters[2] = disturbance[2];
    for (int j = 0; j < 4; ++j) parameters[3 + j] = reference_states(6 + j, i);
    ESO_NMPC_SOLVER_SYMBOL(ESO_NMPC_ACADOS_HASH, _acados_update_params)(impl_->capsule, i, parameters.data(), kNp);
    std::array<double, 13> yref{};
    for (int j = 0; j < 6; ++j) yref[j] = reference_states(j, i);
    for (int j = 0; j < 4; ++j) yref[9 + j] = feedforward(j, i);
    ocp_nlp_cost_model_set(impl_->config, impl_->dims, impl_->in, i, "yref", yref.data());
  }
  parameters[0] = disturbance[0]; parameters[1] = disturbance[1]; parameters[2] = disturbance[2];
  for (int j = 0; j < 4; ++j) parameters[3 + j] = reference_states(6 + j, kGeneratedN);
  ESO_NMPC_SOLVER_SYMBOL(ESO_NMPC_ACADOS_HASH, _acados_update_params)(impl_->capsule, kGeneratedN, parameters.data(), kNp);
  std::array<double, 9> yref_terminal{};
  for (int j = 0; j < 6; ++j) yref_terminal[j] = reference_states(j, kGeneratedN);
  ocp_nlp_cost_model_set(impl_->config, impl_->dims, impl_->in, kGeneratedN, "yref", yref_terminal.data());

  const auto solve_started = std::chrono::steady_clock::now();
  impl_->timing.set_end_steady_s = std::chrono::duration<double>(
    solve_started.time_since_epoch()).count();
  impl_->timing.solve_0_steady_s = impl_->timing.set_end_steady_s;
  const int status = ESO_NMPC_SOLVER_SYMBOL(ESO_NMPC_ACADOS_HASH, _acados_solve)(impl_->capsule);
  const auto solve_finished = std::chrono::steady_clock::now();
  impl_->timing.solve_1_steady_s = std::chrono::duration<double>(
    solve_finished.time_since_epoch()).count();
  impl_->timing.set_ms = 1.0e3 * std::chrono::duration<double>(solve_started - wrapper_started).count();
  impl_->timing.solve_wall_ms = 1.0e3 * std::chrono::duration<double>(solve_finished - solve_started).count();
  auto read_stat_ms = [this](const char * field) {
    double value = 0.0;
    ocp_nlp_get(impl_->capsule->nlp_solver, field, &value);
    return std::isfinite(value) && value >= 0.0 ? 1.0e3 * value : 0.0;
  };
  impl_->timing.time_tot_ms = read_stat_ms("time_tot");
  impl_->timing.time_qp_ms = read_stat_ms("time_qp");
  impl_->timing.time_qp_xcond_ms = read_stat_ms("time_qp_xcond");
  impl_->timing.time_qp_solver_call_ms = read_stat_ms("time_qp_solver_call");
  // SQP_RTI resets the QP, linearization and regularization sub-timers on
  // every solve, but acados currently leaves qpscaling and sim cumulative.
  // Convert those two counters to per-solve durations before logging them.
  const double cumulative_qpscaling_ms = read_stat_ms("time_qpscaling");
  impl_->timing.time_qpscaling_ms = std::max(
    0.0, cumulative_qpscaling_ms - impl_->previous_time_qpscaling_ms);
  impl_->previous_time_qpscaling_ms = cumulative_qpscaling_ms;
  impl_->timing.time_lin_ms = read_stat_ms("time_lin");
  const double cumulative_sim_ms = read_stat_ms("time_sim");
  impl_->timing.time_sim_ms = std::max(
    0.0, cumulative_sim_ms - impl_->previous_time_sim_ms);
  impl_->previous_time_sim_ms = cumulative_sim_ms;
  impl_->timing.time_reg_ms = read_stat_ms("time_reg");
  if (status != ACADOS_SUCCESS) {
    impl_->have_last = false;
    return false;
  }
  for (int i = 0; i <= kGeneratedN; ++i) {
    ocp_nlp_out_get(impl_->config, impl_->dims, impl_->out, i, "x", &impl_->last_x[i * kNx]);
  }
  for (int i = 0; i < kGeneratedN; ++i) {
    ocp_nlp_out_get(impl_->config, impl_->dims, impl_->out, i, "u", &impl_->last_u[i * kNu]);
  }
  impl_->have_last = true;
  for (int j = 0; j < kNu; ++j) command[j] = impl_->last_u[j];
  command[0] = std::clamp(command[0], impl_->thrust_min, impl_->thrust_max);
  for (int j = 0; j < 3; ++j) command[j + 1] = std::clamp(command[j + 1], -impl_->body_rate_max[j], impl_->body_rate_max[j]);
  return true;
}

const AcadosController::Timing & AcadosController::last_timing() const
{
  return impl_->timing;
}

void AcadosController::reset_warm_start()
{
  impl_->have_last = false;
}

EsoNmpcNode::EsoNmpcNode()
: Node("eso_nmpc_cpp"),
  mass_(declare_parameter("mass", 2.0643076923)),
  gravity_(declare_parameter("gravity", 9.80665)),
  rate_tau_(declare_parameter("rate_tau", 0.15)),
  sample_time_(declare_parameter("sample_time", 0.01)),
  control_period_(declare_parameter("control_period", 0.01)),
  enforce_reference_sample_time_(declare_parameter("enforce_reference_sample_time", true)),
  horizon_steps_(declare_parameter("horizon_steps", 30)),
  reference_timeout_s_(declare_parameter("reference_timeout", 0.20)),
  rc_timeout_s_(declare_parameter("rc_timeout", 0.50)),
  odometry_timestamp_gap_threshold_s_(declare_parameter(
    "odometry_timestamp_gap_threshold", 0.10)),
  manual_control_topic_(declare_parameter(
    "manual_control_topic", std::string("/fmu/out/manual_control_setpoint"))),
  rc_deadzone_(declare_parameter("rc_deadzone", 0.08)),
  // Zero disables RC source selection until an explicit AUX channel is
  // configured.  The channel is a deployment parameter, never a code-level
  // assumption about the transmitter mapping.
  rc_aux_channel_(declare_parameter("rc_aux_channel", 6)),
  rc_aux_enable_threshold_(declare_parameter("rc_aux_enable_threshold", 0.50)),
  rc_max_horizontal_speed_(declare_parameter("rc_max_horizontal_speed", 2.0)),
  rc_max_vertical_speed_up_(declare_parameter("rc_max_vertical_speed_up", 3.0)),
  rc_max_vertical_speed_down_(declare_parameter("rc_max_vertical_speed_down", 1.5)),
  rc_max_horizontal_acceleration_(declare_parameter("rc_max_horizontal_acceleration", 2.0)),
  rc_max_vertical_acceleration_up_(declare_parameter("rc_max_vertical_acceleration_up", 4.0)),
  rc_max_vertical_acceleration_down_(declare_parameter("rc_max_vertical_acceleration_down", 3.0)),
  rc_max_yaw_rate_(declare_parameter("rc_max_yaw_rate", 0.5)),
  rc_max_yaw_acceleration_(declare_parameter("rc_max_yaw_acceleration", 1.0)),
  rc_hold_max_horizontal_speed_(declare_parameter("rc_hold_max_horizontal_speed", 0.8)),
  rc_hold_max_vertical_speed_(declare_parameter("rc_hold_max_vertical_speed", 0.6)),
  rc_max_horizontal_position_lead_(declare_parameter("rc_max_horizontal_position_lead", 0.8)),
  rc_max_vertical_position_lead_(declare_parameter("rc_max_vertical_position_lead", 0.4)),
  thrust_min_(declare_parameter("thrust_min", 3.3278)),
  thrust_max_(declare_parameter("thrust_max", 27.7314)),
  hover_throttle_(declare_parameter("hover_throttle", 0.73)),
  throttle_min_(declare_parameter("throttle_min", 0.12)),
  throttle_max_(declare_parameter("throttle_max", 1.0)),
  body_rate_max_(Eigen::Vector3d(
    declare_parameter("body_rate_max_x", 1.0), declare_parameter("body_rate_max_y", 1.0),
    declare_parameter("body_rate_max_z", 1.0))),
  eso_enabled_(declare_parameter("eso_enabled", true)),
  solve_enabled_(declare_parameter("solve_enabled", true)),
  publish_rates_enabled_(declare_parameter("publish_rates_enabled", true)),
  eso_activation_delay_s_(declare_parameter("eso_activation_delay", 3.0)),
  eso_(declare_parameter("eso_bandwidth", 2.5), declare_parameter("eso_clamp", 1.0),
       declare_parameter("eso_innovation_limit", 0.5)),
  controller_(mass_, gravity_, rate_tau_, sample_time_, horizon_steps_, thrust_min_, thrust_max_,
              body_rate_max_, declare_parameter("warm_start", true))
{
  flight_log_root_ = declare_parameter("flight_log_root", default_flight_log_root());
  timing_log_path_ = declare_parameter("timing_log_path", std::string{});
  flight_log_path_ = declare_parameter("flight_log_path", std::string{});
  if (flight_log_path_.empty()) {
    const std::filesystem::path session_directory =
      std::filesystem::path(flight_log_root_.empty() ? default_flight_log_root() : flight_log_root_) /
      flight_session_stamp();
    flight_log_path_ = (session_directory / "nmpc_flight.csv").string();
    if (timing_log_path_.empty()) timing_log_path_ = (session_directory / "nmpc_timing.csv").string();
  } else if (timing_log_path_.empty()) {
    timing_log_path_ = (std::filesystem::path(flight_log_path_).parent_path() / "nmpc_timing.csv").string();
  }
  flight_log_buffer_size_ = static_cast<std::size_t>(declare_parameter("flight_log_buffer_size", 4096));
  flight_log_flush_period_ms_ = declare_parameter("flight_log_flush_period_ms", 250);
  if (flight_log_buffer_size_ == 0 || flight_log_flush_period_ms_ <= 0) {
    throw std::invalid_argument("flight logger buffer size and flush period must be positive");
  }
  if (!enforce_reference_sample_time_) {
    RCLCPP_WARN(
      get_logger(),
      "reference sample-time validation is DISABLED for diagnostic use; "
      "solver discretization remains %.6f s", sample_time_);
  }
  control_enabled_.store(declare_parameter("control_enabled_at_start", false));
  if (control_enabled_.load()) {
    control_enable_time_s_ = std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  if (horizon_steps_ + 1 > kMaxPoints) throw std::invalid_argument("horizon exceeds trajectory protocol capacity");
  if (rc_aux_channel_ < 0 || rc_aux_channel_ > 6) {
    throw std::invalid_argument("rc_aux_channel must be in [0, 6] (0 disables RC)");
  }
  if (!(odometry_timestamp_gap_threshold_s_ > 0.0) ||
      !std::isfinite(odometry_timestamp_gap_threshold_s_)) {
    throw std::invalid_argument("odometry timestamp gap threshold must be finite and positive");
  }
  if (!(rc_timeout_s_ > 0.0) || !(rc_deadzone_ >= 0.0 && rc_deadzone_ < 1.0) ||
      !(rc_max_horizontal_speed_ > 0.0) || !(rc_max_vertical_speed_up_ > 0.0) ||
      !(rc_max_vertical_speed_down_ > 0.0) || !(rc_max_horizontal_acceleration_ > 0.0) ||
      !(rc_max_vertical_acceleration_up_ > 0.0) || !(rc_max_vertical_acceleration_down_ > 0.0) ||
      !(rc_hold_max_horizontal_speed_ > 0.0) || !(rc_hold_max_vertical_speed_ > 0.0)) {
    throw std::invalid_argument("invalid RC-NMPC limits or timeout");
  }
  // VehicleOdometry is a high-rate best-effort stream.  Keep enough samples
  // to absorb a short executor/DDS scheduling burst without losing the state
  // chain; the controller still consumes only the newest delivered sample.
  // Keep the PX4 sensor queue aligned with the known-good C++ baseline.  PX4's
  // bare-DDS odometry publisher itself is KEEP_LAST(1); a larger ROS queue can
  // preserve stale samples after a transport burst and worsen reordering.
  auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
  // High-rate PX4 setpoints are latest-value data.  A reliable queue can
  // build back-pressure in the uXRCE bridge when the C++ controller publishes
  // at 100 Hz; dropping an old setpoint is safer than delaying it.
  // PX4's input bridge and the previously verified C++ baseline use reliable
  // delivery for control setpoints.  Keep the latest-value depth, but retain
  // reliable delivery so a transient DDS burst cannot create a long PX4 gap.
  auto px4_setpoint_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
  // A horizon is a complete reference for the next control interval.  It is
  // latest-value data: keep only the newest sample and let the C++ controller
  // continue with its freshness guard if a transient DDS sample is dropped.
  // Isolate its callback from the mutually-exclusive solve/state group.
  auto trajectory_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
  // Keep state/control inputs serialized.  Trajectory reception is separate
  // so a queued odometry/solve callback cannot starve reference updates.
  control_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  trajectory_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  heartbeat_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions control_options;
  control_options.callback_group = control_callback_group_;
  rclcpp::SubscriptionOptions trajectory_options;
  trajectory_options.callback_group = trajectory_callback_group_;
  rclcpp::SubscriptionOptions manual_options;
  manual_options.callback_group = control_callback_group_;
  odometry_subscription_ = create_subscription<px4_msgs::msg::VehicleOdometry>(
    "/fmu/out/vehicle_odometry", sensor_qos,
    std::bind(&EsoNmpcNode::odometry_callback, this, std::placeholders::_1), control_options);
  status_subscription_ = create_subscription<px4_msgs::msg::VehicleStatus>(
    "/fmu/out/vehicle_status_v1", sensor_qos,
    std::bind(&EsoNmpcNode::status_callback, this, std::placeholders::_1), control_options);
  trajectory_subscription_ = create_subscription<px4_msgs::msg::NmpcTrajectorySetpoint>(
    "/nmpc/in/trajectory_setpoint", trajectory_qos,
    std::bind(&EsoNmpcNode::trajectory_callback, this, std::placeholders::_1), trajectory_options);
  manual_control_subscription_ = create_subscription<px4_msgs::msg::ManualControlSetpoint>(
    manual_control_topic_, sensor_qos,
    std::bind(&EsoNmpcNode::manual_control_callback, this, std::placeholders::_1), manual_options);
  enable_subscription_ = create_subscription<std_msgs::msg::Bool>(
    "/nmpc/control_enabled", rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
    std::bind(&EsoNmpcNode::enable_callback, this, std::placeholders::_1), control_options);
  heartbeat_publisher_ = create_publisher<px4_msgs::msg::OffboardControlMode>(
    "/fmu/in/offboard_control_mode", px4_setpoint_qos);
  rc_timeout_publisher_ = create_publisher<std_msgs::msg::Bool>(
    "/nmpc/rc_timeout", rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
  odometry_timestamp_fault_publisher_ = create_publisher<std_msgs::msg::Bool>(
    "/nmpc/odometry_timestamp_fault", rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
  rates_publisher_ = create_publisher<px4_msgs::msg::VehicleRatesSetpoint>(
    "/fmu/in/vehicle_rates_setpoint", px4_setpoint_qos);
  heartbeat_timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(std::max(0.01, sample_time_))),
    std::bind(&EsoNmpcNode::heartbeat_callback, this), heartbeat_callback_group_);
  timestamp_epoch_us_ = static_cast<uint64_t>(get_clock()->now().nanoseconds() / 1000);
  timestamp_monotonic_origin_ = std::chrono::steady_clock::now();
  if (!flight_log_path_.empty() || !timing_log_path_.empty()) {
    flight_log_thread_ = std::thread(&EsoNmpcNode::flight_log_worker, this);
  }
  RCLCPP_INFO(get_logger(), "flight logs: %s, timing: %s",
              flight_log_path_.c_str(), timing_log_path_.c_str());
  publish_rc_timeout(false);
  publish_odometry_timestamp_fault(false);
  RCLCPP_INFO(get_logger(), "C++ single-process ESO+Reference+Acados+Publish ready (N=%d, mpc_dt=%.3f, control_period=%.3f)",
              horizon_steps_, sample_time_, control_period_);
}

EsoNmpcNode::~EsoNmpcNode()
{
  stop_flight_logger();
}

void EsoNmpcNode::enable_callback(std_msgs::msg::Bool::ConstSharedPtr message)
{
  std::lock_guard<std::mutex> lock(mutex_);
  const bool previous = control_enabled_.load();
  if (message->data && !previous) {
    if (rc_timeout_fault_.load() || odometry_timestamp_fault_.load()) {
      control_enabled_.store(false);
      RCLCPP_WARN(
        get_logger(),
        "rejecting NMPC re-enable while a safety fault is latched; restart the node for recovery");
      return;
    }
    control_enabled_.store(true);
    control_enable_time_s_ = std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
    eso_active_ = false;
    last_disturbance_.setZero();
    last_command_thrust_ = mass_ * gravity_;
    rc_mode_active_ = false;
    rc_hold_active_ = false;
    rc_neutral_latched_ = false;
    rc_timeout_fault_.store(false);
    odometry_timestamp_fault_.store(false);
    last_odometry_receive_time_s_ = 0.0;
    last_odometry_timestamp_ = 0;
    publish_rc_timeout(false);
    publish_odometry_timestamp_fault(false);
    RCLCPP_INFO(get_logger(), "NMPC control enabled; ESO activation delay started");
  } else if (!message->data && previous) {
    control_enabled_.store(false);
    eso_active_ = false;
    rc_mode_active_ = false;
    rc_hold_active_ = false;
    rc_neutral_latched_ = false;
    reset_controller_warm_start();
    RCLCPP_INFO(get_logger(), "NMPC control disabled");
  }
}

void EsoNmpcNode::status_callback(px4_msgs::msg::VehicleStatus::ConstSharedPtr message)
{
  vehicle_status_received_.store(true);
  vehicle_armed_.store(message->arming_state == px4_msgs::msg::VehicleStatus::ARMING_STATE_ARMED);
}

void EsoNmpcNode::trajectory_callback(px4_msgs::msg::NmpcTrajectorySetpoint::ConstSharedPtr message)
{
  if (message->points != horizon_steps_ + 1 ||
      (enforce_reference_sample_time_ &&
       std::abs(static_cast<double>(message->sample_time) - sample_time_) > 1.0e-4)) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "rejecting trajectory: points=%u sample_time=%.6f", message->points, message->sample_time);
    return;
  }
  for (int i = 0; i < horizon_steps_ + 1; ++i) {
    if (!std::isfinite(message->yaw[i])) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "rejecting trajectory: non-finite yaw");
      return;
    }
    for (int j = 0; j < 3; ++j) {
      const int index = 3 * i + j;
      if (!std::isfinite(message->position[index]) ||
          !std::isfinite(message->velocity[index]) ||
          !std::isfinite(message->acceleration[index])) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "rejecting trajectory: non-finite state sample");
        return;
      }
    }
  }
  Trajectory copy;
  copy.timestamp = message->timestamp;
  copy.sequence = message->sequence;
  copy.points = message->points;
  copy.sample_time = message->sample_time;
  copy.position = message->position;
  copy.velocity = message->velocity;
  copy.acceleration = message->acceleration;
  copy.yaw = message->yaw;
  std::lock_guard<std::mutex> lock(mutex_);
  trajectory_ = copy;
  trajectory_valid_ = true;
  last_receive_time_s_ = std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

void EsoNmpcNode::manual_control_callback(
  px4_msgs::msg::ManualControlSetpoint::ConstSharedPtr message)
{
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
  const double aux = aux_value(*message, rc_aux_channel_);
  const bool aux_enabled = std::isfinite(aux) && aux >= rc_aux_enable_threshold_;
  const bool source_rc = message->data_source == px4_msgs::msg::ManualControlSetpoint::SOURCE_RC;
  const bool valid = source_rc && message->valid &&
    std::isfinite(message->roll) && std::isfinite(message->pitch) &&
    std::isfinite(message->yaw) && std::isfinite(message->throttle);
  std::lock_guard<std::mutex> lock(mutex_);
  rc_aux_enabled_ = aux_enabled;
  if (!valid) {
    manual_control_valid_ = false;
    if (aux_enabled && !source_rc) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "AUX%d selected but ManualControlSetpoint source is not SOURCE_RC (%u)",
        rc_aux_channel_, static_cast<unsigned int>(message->data_source));
    }
    return;
  }
  rc_sticks_ = {message->roll, message->pitch, message->yaw, message->throttle};
  manual_control_valid_ = true;
  last_manual_receive_time_s_ = now;
}

void EsoNmpcNode::reset_controller_warm_start()
{
  std::lock_guard<std::mutex> lock(controller_mutex_);
  controller_.reset_warm_start();
}

bool EsoNmpcNode::build_rc_trajectory(const Eigen::Vector3d & measured_position,
                                      const Eigen::Vector3d & measured_velocity,
                                      double measured_yaw, double dt,
                                      Trajectory & trajectory)
{
  // This function runs from the odometry callback while the heartbeat timer
  // can concurrently run the RC timeout watchdog.  Keep the complete RC
  // reference state transition under the same mutex as the watchdog so a
  // timeout cannot race a reference update or warm-start reset.
  std::lock_guard<std::mutex> lock(mutex_);
  if (!(dt > 0.0) || !std::isfinite(dt)) dt = sample_time_;
  dt = std::clamp(dt, 1.0e-3, 0.1);
  const std::array<float, 4> sticks = rc_sticks_;
  const double roll = apply_deadzone(sticks[0], rc_deadzone_);
  const double pitch = apply_deadzone(sticks[1], rc_deadzone_);
  const double yaw_stick = apply_deadzone(sticks[2], rc_deadzone_);
  const double throttle = apply_deadzone(sticks[3], rc_deadzone_);
  Eigen::Vector2d stick_horizontal(pitch, roll);
  const double stick_norm = stick_horizontal.norm();

  const bool sticks_neutral = stick_norm < kEpsilon &&
    std::abs(throttle) < kEpsilon && std::abs(yaw_stick) < kEpsilon;
  if (!rc_mode_active_) {
    rc_reference_position_ = measured_position;
    rc_reference_velocity_.setZero();
    rc_reference_yaw_ = measured_yaw;
    rc_reference_yaw_rate_ = 0.0;
    rc_hold_active_ = false;
    rc_neutral_latched_ = sticks_neutral;
    rc_mode_active_ = true;
    reset_controller_warm_start();
    RCLCPP_INFO(get_logger(), "RC-NMPC reference enabled (AUX%d)", rc_aux_channel_);
  }

  Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
  if (sticks_neutral && rc_neutral_latched_) {
    // Already in hold: keep a fixed target, exactly like PX4 Position/Hold.
    rc_reference_velocity_.setZero();
    rc_reference_yaw_rate_ = 0.0;
  } else {
    if (!sticks_neutral && rc_neutral_latched_) {
      // Open the position loop for a new manual segment from the actual state.
      rc_reference_position_ = measured_position;
      rc_reference_velocity_.setZero();
      rc_reference_yaw_ = measured_yaw;
      rc_reference_yaw_rate_ = 0.0;
      rc_neutral_latched_ = false;
      reset_controller_warm_start();
    }
    const double cosine = std::cos(rc_reference_yaw_);
    const double sine = std::sin(rc_reference_yaw_);
    if (stick_norm > 1.0) stick_horizontal /= stick_norm;
    Eigen::Vector2d acceleration_xy(
      cosine * stick_horizontal[0] - sine * stick_horizontal[1],
      sine * stick_horizontal[0] + cosine * stick_horizontal[1]);
    if (sticks_neutral) {
      const double speed = rc_reference_velocity_.head<2>().norm();
      if (speed > kEpsilon) acceleration_xy = -rc_max_horizontal_acceleration_ *
        rc_reference_velocity_.head<2>() / speed;
    } else {
      acceleration_xy *= rc_max_horizontal_acceleration_;
    }
    acceleration = Eigen::Vector3d(acceleration_xy[0], acceleration_xy[1], 0.0);
    if (std::abs(throttle) > kEpsilon) {
      acceleration[2] = -throttle *
        (throttle >= 0.0 ? rc_max_vertical_acceleration_up_ : rc_max_vertical_acceleration_down_);
    } else if (std::abs(rc_reference_velocity_[2]) > kEpsilon) {
      acceleration[2] = rc_reference_velocity_[2] > 0.0 ?
        -rc_max_vertical_acceleration_down_ : rc_max_vertical_acceleration_up_;
    }

    rc_reference_velocity_ += acceleration * dt;
    const double horizontal_speed = rc_reference_velocity_.head<2>().norm();
    if (horizontal_speed > rc_max_horizontal_speed_)
      rc_reference_velocity_.head<2>() *= rc_max_horizontal_speed_ / horizontal_speed;
    rc_reference_velocity_[2] = std::clamp(
      rc_reference_velocity_[2], -rc_max_vertical_speed_up_, rc_max_vertical_speed_down_);
    rc_reference_position_ += rc_reference_velocity_ * dt;
    Eigen::Vector2d horizontal_lead = rc_reference_position_.head<2>() - measured_position.head<2>();
    const double lead_norm = horizontal_lead.norm();
    if (lead_norm > rc_max_horizontal_position_lead_)
      rc_reference_position_.head<2>() = measured_position.head<2>() +
        rc_max_horizontal_position_lead_ * horizontal_lead / lead_norm;
    rc_reference_position_[2] = std::clamp(
      rc_reference_position_[2], measured_position[2] - rc_max_vertical_position_lead_,
      measured_position[2] + rc_max_vertical_position_lead_);

    const double target_yaw_rate = yaw_stick * rc_max_yaw_rate_;
    rc_reference_yaw_rate_ += std::clamp(
      target_yaw_rate - rc_reference_yaw_rate_,
      -rc_max_yaw_acceleration_ * dt, rc_max_yaw_acceleration_ * dt);
    rc_reference_yaw_ = wrap_angle(rc_reference_yaw_ + rc_reference_yaw_rate_ * dt);

    // Match PX4's lockPosition(): wait for both the commanded and measured
    // velocity to be below the hold thresholds before latching current pose.
    if (sticks_neutral && rc_reference_velocity_.head<2>().norm() <= rc_hold_max_horizontal_speed_ &&
        measured_velocity.head<2>().norm() <= rc_hold_max_horizontal_speed_ &&
        std::abs(rc_reference_velocity_[2]) <= rc_hold_max_vertical_speed_ &&
        std::abs(measured_velocity[2]) <= rc_hold_max_vertical_speed_) {
      rc_reference_position_ = measured_position;
      rc_reference_velocity_.setZero();
      rc_reference_yaw_ = measured_yaw;
      rc_reference_yaw_rate_ = 0.0;
      rc_neutral_latched_ = true;
      acceleration.setZero();
      reset_controller_warm_start();
      RCLCPP_INFO(get_logger(), "RC velocity stopped; latched position hold");
    }
  }
  trajectory = Trajectory{};
  trajectory.timestamp = px4_timestamp_us();
  trajectory.sequence = ++rc_sequence_;
  trajectory.points = static_cast<uint8_t>(horizon_steps_ + 1);
  trajectory.sample_time = static_cast<float>(sample_time_);
  Eigen::Vector3d position = rc_reference_position_;
  Eigen::Vector3d velocity = rc_reference_velocity_;
  double yaw = rc_reference_yaw_;
  for (int i = 0; i <= horizon_steps_; ++i) {
    const Eigen::Vector3d a = acceleration;
    for (int j = 0; j < 3; ++j) {
      trajectory.position[3 * i + j] = static_cast<float>(position[j]);
      trajectory.velocity[3 * i + j] = static_cast<float>(velocity[j]);
      trajectory.acceleration[3 * i + j] = static_cast<float>(a[j]);
    }
    trajectory.yaw[i] = static_cast<float>(yaw);
    if (i == horizon_steps_) break;
    position += velocity * sample_time_ + 0.5 * a * sample_time_ * sample_time_;
    velocity += a * sample_time_;
    const double speed = velocity.head<2>().norm();
    if (speed > rc_max_horizontal_speed_) velocity.head<2>() *= rc_max_horizontal_speed_ / speed;
    velocity[2] = std::clamp(velocity[2], -rc_max_vertical_speed_up_, rc_max_vertical_speed_down_);
    yaw = wrap_angle(yaw + rc_reference_yaw_rate_ * sample_time_);
  }
  return true;
}

void EsoNmpcNode::odometry_callback(px4_msgs::msg::VehicleOdometry::ConstSharedPtr message)
{
  const auto timing_rx = std::chrono::steady_clock::now();
  const auto timing_control_start = timing_rx;
  if (message->pose_frame != px4_msgs::msg::VehicleOdometry::POSE_FRAME_NED ||
      message->velocity_frame != px4_msgs::msg::VehicleOdometry::VELOCITY_FRAME_NED) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "rejecting odometry: expected NED pose and velocity frames");
    return;
  }
  if (!control_enabled_.load()) return;
  const double receive_time = std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
  uint64_t previous_px4 = last_px4_timestamp_us_.load();
  while (previous_px4 < message->timestamp &&
         !last_px4_timestamp_us_.compare_exchange_weak(previous_px4, message->timestamp)) {}

  // PX4's timestamp fields are useful diagnostics, but they are not a safe
  // transport clock here: a DDS-delivered sample can arrive out of order, and
  // PX4 time synchronisation can change the timestamp epoch.  Use the local
  // monotonic receive clock for the safety gate and controller step instead.
  const uint64_t odometry_timestamp = message->timestamp;
  const uint64_t previous_odometry_timestamp = last_odometry_timestamp_;
  const double previous_odometry_receive_time_s = last_odometry_receive_time_s_;
  if (odometry_timestamp == 0) {
    ++odometry_timestamp_reorder_count_;
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "rejecting odometry with zero PX4 timestamp; waiting for a valid sample");
    return;
  }
  if (previous_odometry_timestamp > 0 && odometry_timestamp <= previous_odometry_timestamp) {
    ++odometry_timestamp_reorder_count_;
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "out-of-order odometry timestamp %llu after %llu; accepting sample and using receive clock",
      static_cast<unsigned long long>(odometry_timestamp),
      static_cast<unsigned long long>(previous_odometry_timestamp));
  }
  const double receive_step_s = previous_odometry_receive_time_s > 0.0 ?
    receive_time - previous_odometry_receive_time_s : sample_time_;
  if (!(receive_step_s > 0.0) || receive_step_s > odometry_timestamp_gap_threshold_s_) {
    // During prestream/arming, the vehicle is not airborne yet.  A DDS
    // participant may still be discovering or resynchronizing, so establish
    // a fresh receive baseline without latching an in-flight fault.  Once
    // PX4 reports ARMED, the same 0.10 s gate is strict.
    const double control_age_s = control_enable_time_s_ > 0.0 ?
      receive_time - control_enable_time_s_ : 0.0;
    const bool preflight_resync_window =
      !vehicle_armed_.load() &&
      control_age_s < 5.0;
    if (preflight_resync_window) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "pre-flight odometry receive gap %.3f s; re-baselining before arming",
        receive_step_s);
    } else {
      ++odometry_timestamp_gap_count_;
      reset_controller_warm_start();
      eso_active_ = false;
      last_disturbance_.setZero();
      control_enabled_.store(false);
      if (!odometry_timestamp_fault_.exchange(true)) {
        publish_odometry_timestamp_fault(true);
        RCLCPP_ERROR(
          get_logger(),
          "odometry receive gap %.3f s (threshold %.3f s); NMPC output latched off",
          receive_step_s,
          odometry_timestamp_gap_threshold_s_);
      }
      return;
    }
  }
  // This watchdog is updated for every valid odometry sample, even when the
  // external trajectory is stale.  A trajectory DDS pause must not be
  // misclassified as an odometry transport gap.
  last_odometry_receive_time_s_ = receive_time;
  last_odometry_timestamp_ = std::max(previous_odometry_timestamp, odometry_timestamp);
  const double dt = std::clamp(receive_step_s, 0.001, odometry_timestamp_gap_threshold_s_);

  Trajectory trajectory;
  bool use_rc_reference = false;
  bool rc_input_fresh = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    rc_input_fresh = manual_control_valid_ &&
      (receive_time - last_manual_receive_time_s_ <= rc_timeout_s_);
    if (rc_timeout_fault_.load()) return;
    use_rc_reference = rc_aux_enabled_ && rc_input_fresh;
    if (rc_aux_enabled_ && !use_rc_reference) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "RC-NMPC selected but no valid ManualControlSetpoint");
      return;
    }
    if (!use_rc_reference && (!trajectory_valid_ ||
      receive_time - last_receive_time_s_ > reference_timeout_s_)) return;
    if (!use_rc_reference) trajectory = trajectory_;
  }
  const auto timing_state_start = std::chrono::steady_clock::now();
  State state;
  for (int i = 0; i < 3; ++i) {
    state[i] = message->position[i];
    state[3 + i] = message->velocity[i];
    state[10 + i] = message->angular_velocity[i];
  }
  Eigen::Vector4d q(message->q[0], message->q[1], message->q[2], message->q[3]);
  q = normalize_quaternion(q);
  for (int i = 0; i < 4; ++i) state[6 + i] = q[i];
  const auto timing_state = std::chrono::steady_clock::now();
  const uint64_t sample = message->timestamp_sample > 0 ? message->timestamp_sample : message->timestamp;
  const bool sample_age_valid =
    message->timestamp >= sample && (message->timestamp - sample) <= 100000ULL;
  const double sample_age_ms = sample_age_valid ?
    1.0e-3 * static_cast<double>(message->timestamp - sample) : 0.0;
  if (!sample_age_valid) ++timestamp_sample_age_invalid_count_;
  Eigen::Vector3d disturbance = last_disturbance_;
  if (eso_enabled_) {
    const Eigen::Matrix3d rotation = quaternion_to_rotation(q);
    const Eigen::Vector3d model_acceleration(0.0, 0.0, gravity_);
    const Eigen::Vector3d model = model_acceleration -
      (last_command_thrust_ / mass_) * rotation.col(2);
    const bool activation_delay_complete =
      receive_time - control_enable_time_s_ >= eso_activation_delay_s_;
    if (!activation_delay_complete) {
      eso_.reset(state.segment<3>(3), disturbance);
      eso_active_ = false;
    } else if (!eso_active_) {
      eso_.reset(state.segment<3>(3), disturbance);
      eso_active_ = true;
    } else {
      disturbance = eso_.update(state.segment<3>(3), model, dt);
      last_disturbance_ = disturbance;
    }
  }
  const auto timing_eso = std::chrono::steady_clock::now();

  if (use_rc_reference) {
    const double yaw = std::atan2(
      2.0 * (q[0] * q[3] + q[1] * q[2]),
      1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]));
    if (!build_rc_trajectory(state.head<3>(), state.segment<3>(3), yaw, dt, trajectory)) return;
  } else {
    std::lock_guard<std::mutex> lock(mutex_);
    if (rc_mode_active_) {
      rc_mode_active_ = false;
      rc_hold_active_ = false;
      reset_controller_warm_start();
      RCLCPP_INFO(get_logger(), "RC-NMPC reference disabled; external trajectory selected");
    }
  }
  States references = States::Zero();
  Controls controls = Controls::Zero();
  if (!build_reference(trajectory, q, disturbance, references, controls)) return;
  const auto timing_reference = std::chrono::steady_clock::now();
  if (!solve_enabled_) return;
  Eigen::Matrix<double, kNu, 1> command;
  command.setConstant(std::numeric_limits<double>::quiet_NaN());
  AcadosController::Timing solver_timing{};
  const auto seconds = [](const std::chrono::steady_clock::time_point & value) {
    return value == std::chrono::steady_clock::time_point{} ? 0.0 :
      std::chrono::duration<double>(value.time_since_epoch()).count();
  };
  auto enqueue_record = [&](bool solve_success,
                             const std::chrono::steady_clock::time_point & timing_pub_start,
                             const std::chrono::steady_clock::time_point & timing_pub) {
    TimingRecord timing;
    timing.t_rx_steady_s = seconds(timing_rx);
    timing.t_control_start_steady_s = seconds(timing_control_start);
    timing.t_state_steady_s = seconds(timing_state);
    timing.t_eso_steady_s = seconds(timing_eso);
    timing.t_ref_steady_s = seconds(timing_reference);
    timing.t_pre_end_steady_s = timing.t_ref_steady_s;
    timing.t_pub_start_steady_s = seconds(timing_pub_start);
    timing.t_pub_end_steady_s = seconds(timing_pub);
    timing.executor_wait_ms = 1.0e3 * std::chrono::duration<double>(
      timing_control_start - timing_rx).count();
    timing.preparation_ms = 1.0e3 * std::chrono::duration<double>(
      timing_state_start - timing_control_start).count();
    timing.state_conversion_ms = 1.0e3 * std::chrono::duration<double>(
      timing_state - timing_state_start).count();
    timing.disturbance_estimation_ms = 1.0e3 * std::chrono::duration<double>(
      timing_eso - timing_state).count();
    timing.reference_construction_ms = 1.0e3 * std::chrono::duration<double>(
      timing_reference - timing_eso).count();
    timing.command_publish_ms = timing_pub_start == std::chrono::steady_clock::time_point{} ? 0.0 :
      1.0e3 * std::chrono::duration<double>(timing_pub - timing_pub_start).count();
    timing.control_callback_total_ms = timing_pub == std::chrono::steady_clock::time_point{} ? 0.0 :
      1.0e3 * std::chrono::duration<double>(timing_pub - timing_control_start).count();
    timing.sample_age_ms = sample_age_ms;
    timing.sample_timestamp_age_valid = sample_age_valid;
    timing.sample_to_command_latency_ms = timing_pub_start == std::chrono::steady_clock::time_point{} ? 0.0 :
      1.0e3 * std::chrono::duration<double>(timing_pub_start - timing_rx).count() + sample_age_ms;
    timing.rx_to_pub_ms = timing_pub == std::chrono::steady_clock::time_point{} ? 0.0 :
      1.0e3 * std::chrono::duration<double>(timing_pub - timing_rx).count();
    timing.eso_ms = timing.disturbance_estimation_ms;
    timing.reference_ms = 1.0e3 * std::chrono::duration<double>(timing_reference - timing_eso).count();
    timing.solver = solver_timing;

    FlightLogRecord log;
    log.px4_timestamp_us = message->timestamp;
    log.px4_timestamp_sample_us = sample;
    log.trajectory_timestamp_us = trajectory.timestamp;
    log.trajectory_sequence = trajectory.sequence;
    log.trajectory_points = trajectory.points;
    // At this point either a validated external trajectory or a freshly
    // generated RC trajectory is being consumed by NMPC.
    log.trajectory_valid = true;
    log.solve_success = solve_success;
    log.control_enabled = control_enabled_.load();
    log.eso_enabled = eso_enabled_;
    log.eso_active = eso_active_;
    log.rc_mode_active = use_rc_reference;
    log.rc_aux_enabled = rc_aux_enabled_;
    for (int i = 0; i < kNx; ++i) {
      log.measured_state[static_cast<std::size_t>(i)] = state[i];
      log.reference_state[static_cast<std::size_t>(i)] = references(i, 0);
    }
    for (int i = 0; i < kNu; ++i) {
      log.feedforward[static_cast<std::size_t>(i)] = controls(i, 0);
      log.command[static_cast<std::size_t>(i)] = command[i];
    }
    for (int i = 0; i < 3; ++i) log.disturbance[static_cast<std::size_t>(i)] = disturbance[i];
    for (int i = 0; i < 4; ++i) log.rc_sticks[static_cast<std::size_t>(i)] = rc_sticks_[static_cast<std::size_t>(i)];
    log.timing = timing;
    enqueue_flight_log(std::move(log));
  };
  bool solve_success = false;
  {
    std::lock_guard<std::mutex> lock(controller_mutex_);
    solve_success = controller_.solve(
      state, references, controls, trajectory.points, disturbance, command);
    solver_timing = controller_.last_timing();
  }
  if (!solve_success) {
    enqueue_record(false, std::chrono::steady_clock::time_point{}, std::chrono::steady_clock::time_point{});
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "acados solve failed; keeping heartbeat only");
    return;
  }
  // A watchdog may have latched the output off while this solve was running.
  // Never publish a command after that latch, even if the solve ended well.
  if (!control_enabled_.load() || rc_timeout_fault_.load()) return;
  last_command_thrust_ = command[0];
  const auto timing_pub_start = std::chrono::steady_clock::now();
  publish_rates(command);
  const auto timing_pub = std::chrono::steady_clock::now();
  enqueue_record(true, timing_pub_start, timing_pub);
}

bool EsoNmpcNode::build_reference(const Trajectory & trajectory, const Eigen::Vector4d & anchor,
                                  const Eigen::Vector3d & disturbance, States & states,
                                  Controls & controls) const
{
  std::array<Eigen::Vector4d, kMaxPoints> quaternions{};
  for (int i = 0; i <= horizon_steps_; ++i) {
    Eigen::Vector3d acceleration;
    for (int j = 0; j < 3; ++j) acceleration[j] = trajectory.acceleration[3 * i + j];
    const double yaw = trajectory.yaw[i];
    Eigen::Vector3d body_z(0.0, 0.0, gravity_);
    body_z += disturbance - acceleration;
    if (body_z.norm() < kEpsilon) return false;
    body_z.normalize();
    Eigen::Vector3d heading(std::cos(yaw), std::sin(yaw), 0.0);
    Eigen::Vector3d body_y = body_z.cross(heading);
    if (body_y.norm() < kEpsilon) body_y = body_z.cross(Eigen::Vector3d(-std::sin(yaw), std::cos(yaw), 0.0));
    body_y.normalize();
    const Eigen::Vector3d body_x = body_y.cross(body_z);
    Eigen::Matrix3d rotation;
    rotation.col(0) = body_x; rotation.col(1) = body_y; rotation.col(2) = body_z;
    quaternions[i] = rotation_to_quaternion(rotation);
    if (i == 0 && quaternions[i].dot(anchor) < 0.0) quaternions[i] = -quaternions[i];
    if (i > 0 && quaternions[i].dot(quaternions[i - 1]) < 0.0) quaternions[i] = -quaternions[i];
    for (int j = 0; j < 3; ++j) {
      states(j, i) = trajectory.position[3 * i + j];
      states(3 + j, i) = trajectory.velocity[3 * i + j];
    }
    for (int j = 0; j < 4; ++j) states(6 + j, i) = quaternions[i][j];
    if (i < horizon_steps_) {
      controls(0, i) =
        std::clamp(mass_ * (Eigen::Vector3d(0.0, 0.0, gravity_) + disturbance - acceleration).norm(),
                   thrust_min_, thrust_max_);
    }
  }
  for (int i = 0; i < horizon_steps_; ++i) {
    const Eigen::Vector3d rate = quaternion_delta_rate(quaternions[i], quaternions[i + 1], sample_time_);
    controls(1, i) = rate[0];
    controls(2, i) = rate[1];
    controls(3, i) = rate[2];
    for (int j = 1; j < 4; ++j) controls(j, i) = std::clamp(controls(j, i), -body_rate_max_[j - 1], body_rate_max_[j - 1]);
  }
  for (int j = 0; j < 3; ++j) states(10 + j, horizon_steps_) = controls(1 + j, horizon_steps_ - 1);
  for (int i = 0; i < horizon_steps_; ++i)
    for (int j = 0; j < 3; ++j) states(10 + j, i) = controls(1 + j, i);
  return true;
}

void EsoNmpcNode::rc_timeout_watchdog()
{
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
  bool timeout_event = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool input_fresh = manual_control_valid_ &&
      (now - last_manual_receive_time_s_ <= rc_timeout_s_);
    // AUX selection is the operator's request to use RC-NMPC.  The timeout
    // must be armed even when no valid RC frame has ever activated the local
    // reference, otherwise selecting AUX6 with a missing/wrong source would
    // leave the controller silently waiting forever.
    if (control_enabled_.load() && rc_aux_enabled_ && !input_fresh &&
        !rc_timeout_fault_.exchange(true)) {
      control_enabled_.store(false);
      rc_mode_active_ = false;
      rc_hold_active_ = false;
      rc_neutral_latched_ = false;
      timeout_event = true;
    }
  }
  if (timeout_event) {
    // A latched NMPC fault must not leave a stale prediction in Acados.  The
    // controller lock serializes this reset with any solve already in flight.
    reset_controller_warm_start();
    publish_rc_timeout(true);
    RCLCPP_ERROR(get_logger(),
                 "RC input timeout; NMPC output latched off, awaiting external PX4 fallback");
  }
}

void EsoNmpcNode::heartbeat_callback()
{
  rc_timeout_watchdog();
  publish_heartbeat();
}

void EsoNmpcNode::publish_heartbeat()
{
  px4_msgs::msg::OffboardControlMode message;
  message.timestamp = px4_timestamp_us();
  message.position = false; message.velocity = false; message.acceleration = false;
  message.attitude = false; message.body_rate = true; message.thrust_and_torque = false;
  message.direct_actuator = false;
  heartbeat_publisher_->publish(message);
}

void EsoNmpcNode::publish_rc_timeout(bool active)
{
  if (!rc_timeout_publisher_) return;
  std_msgs::msg::Bool message;
  message.data = active;
  rc_timeout_publisher_->publish(message);
}

void EsoNmpcNode::publish_odometry_timestamp_fault(bool active)
{
  if (!odometry_timestamp_fault_publisher_) return;
  std_msgs::msg::Bool message;
  message.data = active;
  odometry_timestamp_fault_publisher_->publish(message);
}

void EsoNmpcNode::publish_rates(const Eigen::Matrix<double, kNu, 1> & command)
{
  if (!publish_rates_enabled_) return;
  px4_msgs::msg::VehicleRatesSetpoint message;
  message.timestamp = px4_timestamp_us();
  message.roll = static_cast<float>(command[1]);
  message.pitch = static_cast<float>(command[2]);
  message.yaw = static_cast<float>(command[3]);
  const double hover_thrust = mass_ * gravity_;
  const double throttle = std::clamp(hover_throttle_ * command[0] / hover_thrust, throttle_min_, throttle_max_);
  message.thrust_body = {0.0F, 0.0F, static_cast<float>(-throttle)};
  message.reset_integral = false;
  rates_publisher_->publish(message);
}

uint64_t EsoNmpcNode::px4_timestamp_us()
{
  const uint64_t elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now() - timestamp_monotonic_origin_).count());
  const uint64_t now = timestamp_epoch_us_ + elapsed;
  uint64_t old = last_px4_timestamp_us_.load();
  while (old < now && !last_px4_timestamp_us_.compare_exchange_weak(old, now)) {}
  return std::max(now, last_px4_timestamp_us_.load());
}

void EsoNmpcNode::write_timing_log() const
{
  if (timing_log_path_.empty() || timing_records_.empty()) return;
  std::ofstream stream(timing_log_path_);
  if (!stream) {
    RCLCPP_ERROR(get_logger(), "cannot write C++ timing log: %s", timing_log_path_.c_str());
    return;
  }
  stream << "t_rx_steady_s,t_control_start_steady_s,t_state_steady_s,t_eso_steady_s,"
            "t_ref_steady_s,t_pre_end_steady_s,t_set_steady_s,t_solve_0_steady_s,"
            "t_solve_1_steady_s,t_pub_start_steady_s,t_pub_end_steady_s,"
            "executor_wait_ms,preparation_ms,state_conversion_ms,"
            "disturbance_estimation_ms,reference_construction_ms,command_publish_ms,"
            "control_callback_total_ms,sample_age_ms,sample_timestamp_age_valid,"
            "sample_to_command_latency_ms,rx_to_pre_end_ms,pre_end_to_solve_0_ms,"
            "solve_ms_from_timestamps,solve_1_to_pub_ms,rx_to_pub_ms,"
            "acados_set_ms,acados_solve_wall_ms,"
            "time_tot_ms,time_qp_ms,time_qp_xcond_ms,time_qp_solver_call_ms,"
            "time_qp_framework_ms,time_qpscaling_ms,time_lin_ms,time_sim_ms,time_reg_ms\n";
  stream << std::setprecision(9);
  for (const auto & record : timing_records_) {
    const auto & timing = record.solver;
    // QP scaling is performed outside the time_qp timer.
    const double qp_framework = timing.time_qp_ms - timing.time_qp_xcond_ms -
      timing.time_qp_solver_call_ms;
    const double t_set = timing.set_end_steady_s;
    const double t_solve_0 = timing.solve_0_steady_s;
    const double t_solve_1 = timing.solve_1_steady_s;
    const double rx_to_pre_end = 1.0e3 * (record.t_pre_end_steady_s - record.t_rx_steady_s);
    const double pre_end_to_solve_0 = 1.0e3 * (t_solve_0 - record.t_pre_end_steady_s);
    const double solve_from_timestamps = 1.0e3 * (t_solve_1 - t_solve_0);
    const double solve_1_to_pub = 1.0e3 * (record.t_pub_start_steady_s - t_solve_1);
    stream << record.t_rx_steady_s << ',' << record.t_control_start_steady_s << ','
           << record.t_state_steady_s << ',' << record.t_eso_steady_s << ','
           << record.t_ref_steady_s << ',' << record.t_pre_end_steady_s << ','
           << t_set << ',' << t_solve_0 << ',' << t_solve_1 << ','
           << record.t_pub_start_steady_s << ',' << record.t_pub_end_steady_s << ','
           << record.executor_wait_ms << ',' << record.preparation_ms << ','
           << record.state_conversion_ms << ',' << record.disturbance_estimation_ms << ','
           << record.reference_construction_ms << ',' << record.command_publish_ms << ','
           << record.control_callback_total_ms << ',' << record.sample_age_ms << ','
           << (record.sample_timestamp_age_valid ? 1 : 0) << ','
           << record.sample_to_command_latency_ms << ',' << rx_to_pre_end << ','
           << pre_end_to_solve_0 << ',' << solve_from_timestamps << ',' << solve_1_to_pub << ','
           << record.rx_to_pub_ms << ',' << timing.set_ms << ',' << timing.solve_wall_ms << ','
           << timing.time_tot_ms << ','
           << timing.time_qp_ms << ',' << timing.time_qp_xcond_ms << ','
           << timing.time_qp_solver_call_ms << ',' << qp_framework << ','
           << timing.time_qpscaling_ms << ',' << timing.time_lin_ms << ',' << timing.time_sim_ms << ','
           << timing.time_reg_ms << '\n';
  }
  RCLCPP_INFO(get_logger(), "wrote %zu C++ timing samples to %s", timing_records_.size(),
              timing_log_path_.c_str());
  RCLCPP_INFO(get_logger(), "timestamp_sample age invalid count: %llu",
              static_cast<unsigned long long>(timestamp_sample_age_invalid_count_));
}

void EsoNmpcNode::enqueue_flight_log(FlightLogRecord record)
{
  if (!flight_log_thread_.joinable()) return;
  std::lock_guard<std::mutex> lock(flight_log_mutex_);
  if (flight_log_queue_.size() >= flight_log_buffer_size_) {
    flight_log_queue_.pop_front();
    record.logger_dropped_samples = flight_log_dropped_samples_.fetch_add(1) + 1;
  } else {
    record.logger_dropped_samples = flight_log_dropped_samples_.load();
  }
  flight_log_queue_.push_back(std::move(record));
  flight_log_cv_.notify_one();
}

void EsoNmpcNode::stop_flight_logger()
{
  if (!flight_log_thread_.joinable()) return;
  {
    std::lock_guard<std::mutex> lock(flight_log_mutex_);
    flight_log_stop_ = true;
  }
  flight_log_cv_.notify_one();
  flight_log_thread_.join();
}

void EsoNmpcNode::write_timing_log_header(std::ofstream & stream) const
{
  stream << "t_rx_steady_s,t_control_start_steady_s,t_state_steady_s,t_eso_steady_s,"
            "t_ref_steady_s,t_pre_end_steady_s,t_set_steady_s,t_solve_0_steady_s,"
            "t_solve_1_steady_s,t_pub_start_steady_s,t_pub_end_steady_s,"
            "executor_wait_ms,preparation_ms,state_conversion_ms,"
            "disturbance_estimation_ms,reference_construction_ms,command_publish_ms,"
            "control_callback_total_ms,sample_age_ms,sample_timestamp_age_valid,"
            "sample_to_command_latency_ms,rx_to_pre_end_ms,pre_end_to_solve_0_ms,"
            "solve_ms_from_timestamps,solve_1_to_pub_ms,rx_to_pub_ms,"
            "acados_set_ms,acados_solve_wall_ms,time_tot_ms,time_qp_ms,time_qp_xcond_ms,"
            "time_qp_solver_call_ms,time_qp_framework_ms,time_qpscaling_ms,time_lin_ms,"
            "time_sim_ms,time_reg_ms,solve_success,logger_dropped_samples\n";
}

void EsoNmpcNode::write_timing_log_record(std::ofstream & stream,
                                          const FlightLogRecord & flight_record) const
{
  const auto & record = flight_record.timing;
  const auto & timing = record.solver;
  const double qp_framework = timing.time_qp_ms - timing.time_qp_xcond_ms -
    timing.time_qp_solver_call_ms;
  const auto elapsed_ms = [](double end, double start) {
    return end >= start && end > 0.0 && start > 0.0 ? 1.0e3 * (end - start) : 0.0;
  };
  const double rx_to_pre_end = elapsed_ms(record.t_pre_end_steady_s, record.t_rx_steady_s);
  const double pre_end_to_solve_0 = elapsed_ms(timing.solve_0_steady_s, record.t_pre_end_steady_s);
  const double solve_from_timestamps = elapsed_ms(timing.solve_1_steady_s, timing.solve_0_steady_s);
  const double solve_1_to_pub = elapsed_ms(record.t_pub_start_steady_s, timing.solve_1_steady_s);
  stream << std::setprecision(9)
         << record.t_rx_steady_s << ',' << record.t_control_start_steady_s << ','
         << record.t_state_steady_s << ',' << record.t_eso_steady_s << ','
         << record.t_ref_steady_s << ',' << record.t_pre_end_steady_s << ','
         << timing.set_end_steady_s << ',' << timing.solve_0_steady_s << ','
         << timing.solve_1_steady_s << ',' << record.t_pub_start_steady_s << ','
         << record.t_pub_end_steady_s << ',' << record.executor_wait_ms << ','
         << record.preparation_ms << ',' << record.state_conversion_ms << ','
         << record.disturbance_estimation_ms << ',' << record.reference_construction_ms << ','
         << record.command_publish_ms << ',' << record.control_callback_total_ms << ','
         << record.sample_age_ms << ',' << (record.sample_timestamp_age_valid ? 1 : 0) << ','
         << record.sample_to_command_latency_ms << ',' << rx_to_pre_end << ','
         << pre_end_to_solve_0 << ',' << solve_from_timestamps << ',' << solve_1_to_pub << ','
         << record.rx_to_pub_ms << ',' << timing.set_ms << ',' << timing.solve_wall_ms << ','
         << timing.time_tot_ms << ',' << timing.time_qp_ms << ',' << timing.time_qp_xcond_ms << ','
         << timing.time_qp_solver_call_ms << ',' << qp_framework << ',' << timing.time_qpscaling_ms << ','
         << timing.time_lin_ms << ',' << timing.time_sim_ms << ',' << timing.time_reg_ms << ','
         << (flight_record.solve_success ? 1 : 0) << ',' << flight_record.logger_dropped_samples << '\n';
}

void EsoNmpcNode::write_flight_log_header(std::ofstream & stream) const
{
  stream << "px4_timestamp_us,px4_timestamp_sample_us,trajectory_timestamp_us,"
            "trajectory_sequence,trajectory_points,trajectory_valid,solve_success,"
            "control_enabled,eso_enabled,eso_active,rc_mode_active,rc_aux_enabled,"
            "logger_dropped_samples,";
  const std::array<const char *, 13> state_names = {
    "p_x", "p_y", "p_z", "v_x", "v_y", "v_z", "q_w", "q_x", "q_y", "q_z",
    "rate_x", "rate_y", "rate_z"};
  for (const char * name : state_names) stream << "measured_" << name << ',';
  for (const char * name : state_names) stream << "reference_" << name << ',';
  stream << "feedforward_thrust,feedforward_rate_x,feedforward_rate_y,feedforward_rate_z,"
            "command_thrust,command_rate_x,command_rate_y,command_rate_z,"
            "disturbance_x,disturbance_y,disturbance_z,"
            "rc_roll,rc_pitch,rc_yaw,rc_throttle,"
            "t_rx_steady_s,t_control_start_steady_s,t_state_steady_s,t_eso_steady_s,"
            "t_ref_steady_s,t_pre_end_steady_s,t_set_steady_s,t_solve_0_steady_s,"
            "t_solve_1_steady_s,t_pub_start_steady_s,t_pub_end_steady_s,"
            "executor_wait_ms,preparation_ms,state_conversion_ms,disturbance_estimation_ms,"
            "reference_construction_ms,command_publish_ms,control_callback_total_ms,"
            "sample_age_ms,sample_timestamp_age_valid,sample_to_command_latency_ms,"
            "rx_to_pub_ms,acados_set_ms,acados_solve_wall_ms,time_tot_ms,time_qp_ms,"
            "time_qp_xcond_ms,time_qp_solver_call_ms,time_qpscaling_ms,time_lin_ms,"
            "time_sim_ms,time_reg_ms\n";
}

void EsoNmpcNode::write_flight_log_record(std::ofstream & stream,
                                          const FlightLogRecord & flight_record) const
{
  const auto & record = flight_record.timing;
  const auto & timing = record.solver;
  stream << std::setprecision(9)
         << flight_record.px4_timestamp_us << ',' << flight_record.px4_timestamp_sample_us << ','
         << flight_record.trajectory_timestamp_us << ',' << flight_record.trajectory_sequence << ','
         << static_cast<unsigned int>(flight_record.trajectory_points) << ','
         << (flight_record.trajectory_valid ? 1 : 0) << ',' << (flight_record.solve_success ? 1 : 0) << ','
         << (flight_record.control_enabled ? 1 : 0) << ',' << (flight_record.eso_enabled ? 1 : 0) << ','
         << (flight_record.eso_active ? 1 : 0) << ',' << (flight_record.rc_mode_active ? 1 : 0) << ','
         << (flight_record.rc_aux_enabled ? 1 : 0) << ',' << flight_record.logger_dropped_samples;
  for (double value : flight_record.measured_state) stream << ',' << value;
  for (double value : flight_record.reference_state) stream << ',' << value;
  for (double value : flight_record.feedforward) stream << ',' << value;
  for (double value : flight_record.command) stream << ',' << value;
  for (double value : flight_record.disturbance) stream << ',' << value;
  for (double value : flight_record.rc_sticks) stream << ',' << value;
  stream << ',' << record.t_rx_steady_s << ',' << record.t_control_start_steady_s << ','
         << record.t_state_steady_s << ',' << record.t_eso_steady_s << ',' << record.t_ref_steady_s << ','
         << record.t_pre_end_steady_s << ',' << timing.set_end_steady_s << ',' << timing.solve_0_steady_s << ','
         << timing.solve_1_steady_s << ',' << record.t_pub_start_steady_s << ',' << record.t_pub_end_steady_s << ','
         << record.executor_wait_ms << ',' << record.preparation_ms << ',' << record.state_conversion_ms << ','
         << record.disturbance_estimation_ms << ',' << record.reference_construction_ms << ','
         << record.command_publish_ms << ',' << record.control_callback_total_ms << ',' << record.sample_age_ms << ','
         << (record.sample_timestamp_age_valid ? 1 : 0) << ',' << record.sample_to_command_latency_ms << ','
         << record.rx_to_pub_ms << ',' << timing.set_ms << ',' << timing.solve_wall_ms << ','
         << timing.time_tot_ms << ',' << timing.time_qp_ms << ',' << timing.time_qp_xcond_ms << ','
         << timing.time_qp_solver_call_ms << ',' << timing.time_qpscaling_ms << ',' << timing.time_lin_ms << ','
         << timing.time_sim_ms << ',' << timing.time_reg_ms << '\n';
}

void EsoNmpcNode::flight_log_worker()
{
  auto open_stream = [this](const std::string & path, const char * label) {
    std::ofstream stream;
    if (path.empty()) return stream;
    std::error_code error;
    const std::filesystem::path file_path(path);
    if (!file_path.parent_path().empty()) {
      std::filesystem::create_directories(file_path.parent_path(), error);
      if (error) RCLCPP_ERROR(get_logger(), "cannot create %s log directory: %s", label, error.message().c_str());
    }
    stream.open(file_path, std::ios::out | std::ios::trunc);
    if (!stream) RCLCPP_ERROR(get_logger(), "cannot open %s log: %s", label, path.c_str());
    return stream;
  };
  std::ofstream flight_stream = open_stream(flight_log_path_, "flight");
  std::ofstream timing_stream = open_stream(timing_log_path_, "timing");
  if (flight_stream) write_flight_log_header(flight_stream);
  if (timing_stream) write_timing_log_header(timing_stream);
  const auto flush_period = std::chrono::milliseconds(flight_log_flush_period_ms_);
  auto next_flush = std::chrono::steady_clock::now() + flush_period;
  while (true) {
    std::deque<FlightLogRecord> pending;
    {
      std::unique_lock<std::mutex> lock(flight_log_mutex_);
      flight_log_cv_.wait_until(lock, next_flush, [this] {
        return flight_log_stop_ || !flight_log_queue_.empty();
      });
      pending.swap(flight_log_queue_);
      if (flight_log_stop_ && pending.empty()) break;
    }
    for (const auto & record : pending) {
      if (flight_stream) write_flight_log_record(flight_stream, record);
      if (timing_stream) write_timing_log_record(timing_stream, record);
    }
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_flush) {
      if (flight_stream) flight_stream.flush();
      if (timing_stream) timing_stream.flush();
      next_flush = now + flush_period;
    }
  }
  if (flight_stream) flight_stream.flush();
  if (timing_stream) timing_stream.flush();
}

}  // namespace eso_nmpc_node

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  auto node = std::make_shared<eso_nmpc_node::EsoNmpcNode>();
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
