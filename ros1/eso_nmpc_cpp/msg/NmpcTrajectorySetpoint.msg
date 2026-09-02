# Complete NED/FRD-independent kinematic horizon.  Position, velocity and
# acceleration are NED; yaw is around the NED down axis.  Arrays are flattened
# point-major: [point0.x, point0.y, point0.z, point1.x, ...].
std_msgs/Header header
uint32 sequence
uint32 points
float64 sample_time
float64[] position
float64[] velocity
float64[] acceleration
float64[] yaw
