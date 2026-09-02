#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/nmpc_trajectory_setpoint.hpp>
#include <px4_msgs/msg/manual_control_setpoint.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_rates_setpoint.hpp>
#include <std_msgs/msg/bool.hpp>

namespace eso_nmpc_node
{

constexpr int kNx = 13;
constexpr int kNu = 4;
constexpr int kNp = 7;
constexpr int kMaxPoints = 51;

struct Trajectory
{
  uint64_t timestamp{0};
  uint32_t sequence{0};
  uint8_t points{0};
  float sample_time{0.0F};
  std::array<float, 3 * kMaxPoints> position{};
  std::array<float, 3 * kMaxPoints> velocity{};
  std::array<float, 3 * kMaxPoints> acceleration{};
  std::array<float, kMaxPoints> yaw{};
};

class VelocityLeso
{
public:
  VelocityLeso(double bandwidth, double clamp, double innovation_limit);
  void reset(const Eigen::Vector3d & velocity, const Eigen::Vector3d & disturbance);
  Eigen::Vector3d update(const Eigen::Vector3d & velocity,
                         const Eigen::Vector3d & model_acceleration, double dt);

private:
  double beta1_;
  double beta2_;
  double clamp_;
  double innovation_limit_;
  Eigen::Vector3d velocity_hat_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d disturbance_hat_{Eigen::Vector3d::Zero()};
  bool initialized_{false};
};

class AcadosController
{
public:
  struct Timing
  {
    double set_end_steady_s{0.0};
    double solve_0_steady_s{0.0};
    double solve_1_steady_s{0.0};
    double set_ms{0.0};
    double solve_wall_ms{0.0};
    double time_tot_ms{0.0};
    double time_qp_ms{0.0};
    double time_qp_xcond_ms{0.0};
    double time_qp_solver_call_ms{0.0};
    double time_qpscaling_ms{0.0};
    double time_lin_ms{0.0};
    double time_sim_ms{0.0};
    double time_reg_ms{0.0};
  };
  AcadosController(double mass, double gravity, double rate_tau,
                   double sample_time, int horizon, double thrust_min,
                   double thrust_max, const Eigen::Vector3d & body_rate_max,
                   bool warm_start);
  ~AcadosController();
  AcadosController(const AcadosController &) = delete;
  AcadosController & operator=(const AcadosController &) = delete;

  bool solve(const Eigen::Matrix<double, kNx, 1> & state,
             const Eigen::Matrix<double, kNx, kMaxPoints> & reference_states,
             const Eigen::Matrix<double, kNu, kMaxPoints - 1> & feedforward,
             int points, const Eigen::Vector3d & disturbance,
             Eigen::Matrix<double, kNu, 1> & command);
  void reset_warm_start();
  const Timing & last_timing() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class EsoNmpcNode : public rclcpp::Node
{
public:
  EsoNmpcNode();
  ~EsoNmpcNode() override;

private:
  void trajectory_callback(px4_msgs::msg::NmpcTrajectorySetpoint::ConstSharedPtr message);
  void manual_control_callback(px4_msgs::msg::ManualControlSetpoint::ConstSharedPtr message);
  void enable_callback(std_msgs::msg::Bool::ConstSharedPtr message);
  void odometry_callback(px4_msgs::msg::VehicleOdometry::ConstSharedPtr message);
  void heartbeat_callback();
  bool build_reference(const Trajectory & trajectory, const Eigen::Vector4d & anchor,
                       const Eigen::Vector3d & disturbance,
                       Eigen::Matrix<double, kNx, kMaxPoints> & states,
                       Eigen::Matrix<double, kNu, kMaxPoints - 1> & controls) const;
  bool build_rc_trajectory(const Eigen::Vector3d & measured_position,
                           const Eigen::Vector3d & measured_velocity,
                           double measured_yaw, double dt, Trajectory & trajectory);
  void publish_heartbeat();
  void publish_rates(const Eigen::Matrix<double, kNu, 1> & command);
  uint64_t px4_timestamp_us();

  struct TimingRecord
  {
    double t_rx_steady_s{0.0};
    double t_control_start_steady_s{0.0};
    double t_state_steady_s{0.0};
    double t_eso_steady_s{0.0};
    double t_ref_steady_s{0.0};
    double t_pre_end_steady_s{0.0};
    double t_pub_start_steady_s{0.0};
    double t_pub_end_steady_s{0.0};
    double executor_wait_ms{0.0};
    double preparation_ms{0.0};
    double state_conversion_ms{0.0};
    double disturbance_estimation_ms{0.0};
    double reference_construction_ms{0.0};
    double command_publish_ms{0.0};
    double control_callback_total_ms{0.0};
    double sample_age_ms{0.0};
    double sample_to_command_latency_ms{0.0};
    bool sample_timestamp_age_valid{false};
    double rx_to_pub_ms{0.0};
    double eso_ms{0.0};
    double reference_ms{0.0};
    AcadosController::Timing solver{};
  };

  // One immutable sample copied out of the real-time callback.  The logger
  // thread owns all file I/O; this record is deliberately composed only of
  // fixed-size values so enqueueing cannot retain ROS message memory.
  struct FlightLogRecord
  {
    uint64_t px4_timestamp_us{0};
    uint64_t px4_timestamp_sample_us{0};
    uint64_t trajectory_timestamp_us{0};
    uint32_t trajectory_sequence{0};
    uint8_t trajectory_points{0};
    bool trajectory_valid{false};
    bool solve_success{false};
    bool control_enabled{false};
    bool eso_enabled{false};
    bool eso_active{false};
    bool rc_mode_active{false};
    bool rc_aux_enabled{false};
    uint64_t logger_dropped_samples{0};
    std::array<double, kNx> measured_state{};
    std::array<double, kNx> reference_state{};
    std::array<double, kNu> feedforward{};
    std::array<double, kNu> command{};
    std::array<double, 3> disturbance{};
    std::array<double, 4> rc_sticks{};
    TimingRecord timing{};
  };

  void enqueue_flight_log(FlightLogRecord record);
  void flight_log_worker();
  void write_flight_log_header(std::ofstream & stream) const;
  void write_flight_log_record(std::ofstream & stream, const FlightLogRecord & record) const;
  void write_timing_log_header(std::ofstream & stream) const;
  void write_timing_log_record(std::ofstream & stream, const FlightLogRecord & record) const;
  void stop_flight_logger();
  void write_timing_log() const;

  double mass_;
  double gravity_;
  double rate_tau_;
  double sample_time_;
  int horizon_steps_;
  double reference_timeout_s_;
  double rc_timeout_s_;
  double rc_deadzone_;
  int rc_aux_channel_;
  double rc_aux_enable_threshold_;
  double rc_max_horizontal_speed_;
  double rc_max_vertical_speed_up_;
  double rc_max_vertical_speed_down_;
  double rc_max_horizontal_acceleration_;
  double rc_max_vertical_acceleration_up_;
  double rc_max_vertical_acceleration_down_;
  double rc_max_yaw_rate_;
  double rc_max_yaw_acceleration_;
  double rc_hold_max_horizontal_speed_;
  double rc_hold_max_vertical_speed_;
  double rc_max_horizontal_position_lead_;
  double rc_max_vertical_position_lead_;
  double thrust_min_;
  double thrust_max_;
  double hover_throttle_;
  double throttle_min_;
  double throttle_max_;
  Eigen::Vector3d body_rate_max_;
  bool eso_enabled_;
  double eso_activation_delay_s_;
  VelocityLeso eso_;
  AcadosController controller_;

  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odometry_subscription_;
  rclcpp::Subscription<px4_msgs::msg::NmpcTrajectorySetpoint>::SharedPtr trajectory_subscription_;
  rclcpp::Subscription<px4_msgs::msg::ManualControlSetpoint>::SharedPtr manual_control_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enable_subscription_;
  rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr heartbeat_publisher_;
  rclcpp::Publisher<px4_msgs::msg::VehicleRatesSetpoint>::SharedPtr rates_publisher_;
  rclcpp::TimerBase::SharedPtr heartbeat_timer_;
  rclcpp::CallbackGroup::SharedPtr control_callback_group_;
  rclcpp::CallbackGroup::SharedPtr heartbeat_callback_group_;

  std::mutex mutex_;
  Trajectory trajectory_{};
  bool trajectory_valid_{false};
  bool manual_control_valid_{false};
  bool rc_mode_active_{false};
  bool rc_hold_active_{false};
  bool rc_neutral_latched_{false};
  bool rc_aux_enabled_{false};
  double last_manual_receive_time_s_{0.0};
  std::array<float, 4> rc_sticks_{};
  uint32_t rc_sequence_{0};
  Eigen::Vector3d rc_reference_position_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d rc_reference_velocity_{Eigen::Vector3d::Zero()};
  double rc_reference_yaw_{0.0};
  double rc_reference_yaw_rate_{0.0};
  Eigen::Vector4d reference_anchor_{1.0, 0.0, 0.0, 0.0};
  Eigen::Vector3d last_disturbance_{Eigen::Vector3d::Zero()};
  double last_receive_time_s_{0.0};  // CLOCK_MONOTONIC seconds
  uint64_t last_timestamp_sample_{0};
  uint64_t timestamp_sample_age_invalid_count_{0};
  double last_command_thrust_{0.0};
  bool have_odom_{false};
  bool eso_active_{false};
  std::atomic<bool> control_enabled_{true};
  double control_enable_time_s_{0.0};
  std::atomic<uint64_t> last_px4_timestamp_us_{0};
  uint64_t timestamp_epoch_us_{0};
  std::chrono::steady_clock::time_point timestamp_monotonic_origin_{};
  std::string flight_log_root_;
  std::string flight_log_path_;
  std::string timing_log_path_;
  std::size_t flight_log_buffer_size_{4096};
  int flight_log_flush_period_ms_{250};
  std::mutex flight_log_mutex_;
  std::condition_variable flight_log_cv_;
  std::deque<FlightLogRecord> flight_log_queue_;
  std::thread flight_log_thread_;
  bool flight_log_stop_{false};
  std::atomic<uint64_t> flight_log_dropped_samples_{0};
  // Kept for source compatibility with older callers; live logging no longer
  // accumulates timing samples in this vector.
  std::vector<TimingRecord> timing_records_;
};

}  // namespace eso_nmpc_node
