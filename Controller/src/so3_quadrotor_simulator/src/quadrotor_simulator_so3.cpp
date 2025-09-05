#include <Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/SO3Command.h>
#include <quadrotor_simulator/Quadrotor.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include <tf2_ros/transform_broadcaster.h>
#include <uav_utils/geometry_utils.h>
#include "visualization_msgs/Marker.h"

typedef struct _Control
{
  double rpm[4];
} Control;

// typedef struct _Command
// {
//   float force[3];
//   float qx, qy, qz, qw;
//   float kR[3];
//   float kOm[3];
//   float corrections[3];
//   float current_yaw;
//   bool  use_external_yaw;
// } Command;

typedef struct _Command
{
  float vx,vy,vz;
  float wz;
} Command;


typedef struct _Disturbance
{
  Eigen::Vector3d f;
  Eigen::Vector3d m;
} Disturbance;

static Command     command;
static Disturbance disturbance;
static QuadrotorSimulator::Quadrotor quad;

void stateToOdomMsg(const QuadrotorSimulator::Quadrotor::State& state,
                    nav_msgs::Odometry&                         odom);
void quadToImuMsg(const QuadrotorSimulator::Quadrotor& quad,
                  sensor_msgs::Imu&                    imu);
void odomToTF(const nav_msgs::Odometry& odom_msg,
              geometry_msgs::TransformStamped& transformStamped);
void odomToMesh(const nav_msgs::Odometry& odom_msg, 
                visualization_msgs::Marker& meshROS);

static Control
getControl(const QuadrotorSimulator::Quadrotor& quad, const Command& cmd)
{
  // 获取无人机参数
  const double kf = quad.getPropellerThrustCoefficient();
  const double km = quad.getPropellerMomentCoefficient();
  const double d = quad.getArmLength();
  const double mass = quad.getMass();
  const auto state = quad.getState();
  
  // 控制器增益 (需要根据实际调整)
  const double kv_xy = 2.0;   // XY速度控制增益
  const double kv_z = 5.0;    // Z速度控制增益
  const double kp_roll_pitch = 0.1; // 滚转/俯仰角控制增益
  const double kd_roll_pitch = 0.05; // 滚转/俯仰角速度控制增益
  const double kd_yaw = 0.05;  // 偏航角速度控制增益

  // 1. 速度控制器 (世界坐标系)
  // 计算速度误差
  Eigen::Vector3d vel_error(
      cmd.vx - state.v.x(),
      cmd.vy - state.v.y(),
      cmd.vz - state.v.z());
  
  // 计算期望加速度 (PI控制器)
  Eigen::Vector3d des_accel;
  des_accel.x() = kv_xy * vel_error.x();
  des_accel.y() = kv_xy * vel_error.y();
  des_accel.z() = kv_z * vel_error.z() + 9.81; // 重力补偿

  // 2. 计算期望姿态
  // 期望推力方向 (世界坐标系)
  Eigen::Vector3d des_thrust_dir = des_accel.normalized();
  
  // 计算当前偏航角
  double current_yaw = std::atan2(state.R(1, 0), state.R(0, 0));
  
  // 构造期望旋转矩阵
  Eigen::Matrix3d R_des;
  
  // 期望Z轴 = 推力方向
  Eigen::Vector3d Z_des = des_thrust_dir;
  
  // 期望X轴 (水平面投影，保持当前偏航)
  Eigen::Vector3d X_des(std::cos(current_yaw), std::sin(current_yaw), 0);
  X_des = X_des - Z_des.dot(X_des) * Z_des;
  if (X_des.norm() < 1e-6) {
    // 如果Z_des垂直向上/向下，使用默认方向
    X_des = Eigen::Vector3d(1, 0, 0);
  } else {
    X_des.normalize();
  }
  
  // 期望Y轴 = Z轴 × X轴
  Eigen::Vector3d Y_des = Z_des.cross(X_des);
  Y_des.normalize();
  
  R_des.col(0) = X_des;
  R_des.col(1) = Y_des;
  R_des.col(2) = Z_des;

  // 3. 姿态控制器 (SO(3)控制)
  // 姿态误差
  Eigen::Matrix3d R_err = R_des.transpose() * state.R - state.R.transpose() * R_des;
  Eigen::Vector3d e_R(
      R_err(2, 1) - R_err(1, 2),
      R_err(0, 2) - R_err(2, 0),
      R_err(1, 0) - R_err(0, 1));
  e_R *= 0.5;
  
  // 角速度误差
  Eigen::Vector3d des_omega(0, 0, cmd.wz); // 使用期望偏航角速度
  Eigen::Vector3d e_omega = state.omega - des_omega;
  
  // 陀螺力矩补偿
  Eigen::Vector3d J_omega = quad.getInertia() * state.omega;
  Eigen::Vector3d gyro_term = state.omega.cross(J_omega);
  
  // 计算控制力矩
  Eigen::Vector3d M;
  M.x() = -kp_roll_pitch * e_R.x() - kd_roll_pitch * e_omega.x() + gyro_term.x();
  M.y() = -kp_roll_pitch * e_R.y() - kd_roll_pitch * e_omega.y() + gyro_term.y();
  M.z() = -kd_yaw * e_omega.z() + gyro_term.z(); // 只使用角速度控制偏航

  // 4. 计算总推力
  double thrust = mass * des_accel.dot(Z_des);
  
  // 5. 使用原始混控器公式计算电机转速
  double w_sq[4];
  w_sq[0] = thrust / (4 * kf) - M.y() / (2 * d * kf) + M.z() / (4 * km); // 右前
  w_sq[1] = thrust / (4 * kf) + M.y() / (2 * d * kf) + M.z() / (4 * km); // 左后
  w_sq[2] = thrust / (4 * kf) + M.x() / (2 * d * kf) - M.z() / (4 * km); // 左前
  w_sq[3] = thrust / (4 * kf) - M.x() / (2 * d * kf) - M.z() / (4 * km); // 右后

  // 6. 转换为RPM并限幅
  Control control;
  for (int i = 0; i < 4; i++) {
    w_sq[i] = std::max(w_sq[i], 0.0);
    control.rpm[i] = std::sqrt(w_sq[i]);
  }
  
  return control;
}

static void
// cmd_callback(const quadrotor_msgs::SO3Command::ConstPtr& cmd)
// {
//   command.force[0]         = cmd->force.x;
//   command.force[1]         = cmd->force.y;
//   command.force[2]         = cmd->force.z;
//   command.qx               = cmd->orientation.x;
//   command.qy               = cmd->orientation.y;
//   command.qz               = cmd->orientation.z;
//   command.qw               = cmd->orientation.w;
//   command.kR[0]            = cmd->kR[0];
//   command.kR[1]            = cmd->kR[1];
//   command.kR[2]            = cmd->kR[2];
//   command.kOm[0]           = cmd->kOm[0];
//   command.kOm[1]           = cmd->kOm[1];
//   command.kOm[2]           = cmd->kOm[2];
//   command.corrections[0]   = cmd->aux.kf_correction;
//   command.corrections[1]   = cmd->aux.angle_corrections[0];
//   command.corrections[2]   = cmd->aux.angle_corrections[1];
//   command.current_yaw      = cmd->aux.current_yaw;
//   command.use_external_yaw = cmd->aux.use_external_yaw;
// }

cmd_callback(const geometry_msgs::Twist& cmd)
{ 
  command.vx = cmd.linear.x;
  command.vy = cmd.linear.y;
  command.vz = cmd.linear.z;
  command.wz = cmd.angular.z;
}

static void
reset_callback(const std_msgs::Float32 &msg)
{
  quad.initState();

  float _init_y = msg.data;
  Eigen::Vector3d position = Eigen::Vector3d(0, _init_y , 1);
  quad.setStatePos(position);

  command.vx = 0;
  command.vy = 0;
  command.vz = 0;
  command.wz = 0;
}

int
main(int argc, char** argv)
{
  ros::init(argc, argv, "quadrotor_simulator_controller");

  ros::NodeHandle n("~");

  ros::Publisher  odom_pub = n.advertise<nav_msgs::Odometry>("odom", 100);
  ros::Publisher  imu_pub  = n.advertise<sensor_msgs::Imu>("imu", 10);
  ros::Publisher  mesh_pub = n.advertise<visualization_msgs::Marker>("uav", 1);

  tf2_ros::TransformBroadcaster tf_broadcaster;

  ros::Subscriber cmd_sub =
    n.subscribe("cmd", 100, &cmd_callback, ros::TransportHints().tcpNoDelay());

  ros::Subscriber reset_pub = 
    n.subscribe("reset", 10, &reset_callback, ros::TransportHints().tcpNoDelay());
  // ros::Subscriber f_sub =
  //   n.subscribe("force_disturbance", 100, &force_disturbance_callback,
  //               ros::TransportHints().tcpNoDelay());
  // ros::Subscriber m_sub =
  //   n.subscribe("moment_disturbance", 100, &moment_disturbance_callback,
  //               ros::TransportHints().tcpNoDelay());

  double                        _init_x, _init_y, _init_z;
  n.param("simulator/init_state_x", _init_x, 0.0);
  n.param("simulator/init_state_y", _init_y, 0.0);
  n.param("simulator/init_state_z", _init_z, 1.0);

  Eigen::Vector3d position = Eigen::Vector3d(_init_x, _init_y, _init_z);
  quad.setStatePos(position);

  double simulation_rate;
  n.param("rate/simulation", simulation_rate, 1000.0);
  ROS_ASSERT(simulation_rate > 0);

  double odom_rate;
  n.param("rate/odom", odom_rate, 100.0);
  const ros::Duration odom_pub_duration(1 / odom_rate);

  std::string quad_name;
  n.param("quadrotor_name", quad_name, std::string("quadrotor"));

  QuadrotorSimulator::Quadrotor::State state = quad.getState();

  ros::Rate    r(simulation_rate);
  const double dt = 1 / simulation_rate;

  Control control;

  nav_msgs::Odometry odom_msg;
  odom_msg.header.frame_id = "world";
  odom_msg.child_frame_id  = "/" + quad_name;

  sensor_msgs::Imu imu;
  imu.header.frame_id = "world";
  
  geometry_msgs::TransformStamped transformStamped;

  visualization_msgs::Marker meshROS;

  ros::Time next_odom_pub_time = ros::Time::now();
  while (n.ok())
  {
    ros::spinOnce();

    auto last = control;
    control   = getControl(quad, command);
    for (int i = 0; i < 4; ++i)
    {
      //! @bug might have nan when the input is legal
      if (std::isnan(control.rpm[i]))
        control.rpm[i] = last.rpm[i];
    }
    quad.setInput(control.rpm[0], control.rpm[1], control.rpm[2],
                  control.rpm[3]);
    quad.setExternalForce(disturbance.f);
    quad.setExternalMoment(disturbance.m);
    quad.step(dt);

    ros::Time tnow = ros::Time::now();

    if (tnow >= next_odom_pub_time)
    {
      next_odom_pub_time += odom_pub_duration;
      odom_msg.header.stamp = tnow;
      state                 = quad.getState();
      stateToOdomMsg(state, odom_msg);
      quadToImuMsg(quad, imu);
      odomToTF(odom_msg, transformStamped);
      odom_pub.publish(odom_msg);
      imu_pub.publish(imu);
      tf_broadcaster.sendTransform(transformStamped);
      if (mesh_pub.getNumSubscribers() > 0) {
        odomToMesh(odom_msg, meshROS);
        mesh_pub.publish(meshROS);
      }
    }

    r.sleep();
  }

  return 0;
}

void
stateToOdomMsg(const QuadrotorSimulator::Quadrotor::State& state,
               nav_msgs::Odometry&                         odom)
{
  odom.pose.pose.position.x = state.x(0);
  odom.pose.pose.position.y = state.x(1);
  odom.pose.pose.position.z = state.x(2);

  Eigen::Quaterniond q(state.R);
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();

  odom.twist.twist.linear.x = state.v(0);
  odom.twist.twist.linear.y = state.v(1);
  odom.twist.twist.linear.z = state.v(2);

  odom.twist.twist.angular.x = state.omega(0);
  odom.twist.twist.angular.y = state.omega(1);
  odom.twist.twist.angular.z = state.omega(2);
}

void
quadToImuMsg(const QuadrotorSimulator::Quadrotor& quad, sensor_msgs::Imu& imu)

{
  QuadrotorSimulator::Quadrotor::State state = quad.getState();
  Eigen::Quaterniond                   q(state.R);
  imu.orientation.x = q.x();
  imu.orientation.y = q.y();
  imu.orientation.z = q.z();
  imu.orientation.w = q.w();

  imu.angular_velocity.x = state.omega(0);
  imu.angular_velocity.y = state.omega(1);
  imu.angular_velocity.z = state.omega(2);

  imu.linear_acceleration.x = quad.getAcc()[0];
  imu.linear_acceleration.y = quad.getAcc()[1];
  imu.linear_acceleration.z = quad.getAcc()[2];
}


void 
odomToTF(const nav_msgs::Odometry& odom_msg, geometry_msgs::TransformStamped& transformStamped) {
  transformStamped.header.stamp = odom_msg.header.stamp;
  transformStamped.header.frame_id = "world";
  transformStamped.child_frame_id = "odom";

  transformStamped.transform.translation.x = odom_msg.pose.pose.position.x;
  transformStamped.transform.translation.y = odom_msg.pose.pose.position.y;
  transformStamped.transform.translation.z = odom_msg.pose.pose.position.z;

  transformStamped.transform.rotation = odom_msg.pose.pose.orientation;
}

void 
odomToMesh(const nav_msgs::Odometry& odom_msg, visualization_msgs::Marker& meshROS) {
  meshROS.mesh_resource = "file://" + ros::package::getPath("so3_quadrotor_simulator") + "/config/uav.dae";
  meshROS.mesh_use_embedded_materials = true;

  meshROS.header = odom_msg.header;
  meshROS.header.frame_id = "world";

  meshROS.ns     = "mesh";
  meshROS.id     = 0;
  meshROS.type   = visualization_msgs::Marker::MESH_RESOURCE;
  meshROS.action = visualization_msgs::Marker::ADD;

  meshROS.pose = odom_msg.pose.pose;

  meshROS.scale.x = 2.0;
  meshROS.scale.y = 2.0;
  meshROS.scale.z = 2.0;
  meshROS.color.r = 1.0;
  meshROS.color.g = 1.0;
  meshROS.color.b = 1.0;
  meshROS.color.a = 1.0;
}

// getControl(const QuadrotorSimulator::Quadrotor& quad, const Command& cmd)
// {
//   const double _kf = quad.getPropellerThrustCoefficient();
//   const double _km = quad.getPropellerMomentCoefficient();
//   const double kf  = _kf - cmd.corrections[0];
//   const double km  = _km / _kf * kf;

//   const double          d       = quad.getArmLength();
//   const Eigen::Matrix3f J       = quad.getInertia().cast<float>();
//   const float           I[3][3] = { { J(0, 0), J(0, 1), J(0, 2) },
//                           { J(1, 0), J(1, 1), J(1, 2) },
//                           { J(2, 0), J(2, 1), J(2, 2) } };
//   const QuadrotorSimulator::Quadrotor::State state = quad.getState();

//   // Rotation, may use external yaw
//   Eigen::Vector3d _ypr = uav_utils::R_to_ypr(state.R);
//   Eigen::Vector3d ypr  = _ypr;
//   if (cmd.use_external_yaw)
//     ypr[0] = cmd.current_yaw;
//   Eigen::Matrix3d R;
//   R = Eigen::AngleAxisd(ypr[0], Eigen::Vector3d::UnitZ()) *
//       Eigen::AngleAxisd(ypr[1], Eigen::Vector3d::UnitY()) *
//       Eigen::AngleAxisd(ypr[2], Eigen::Vector3d::UnitX());
//   float R11 = R(0, 0);
//   float R12 = R(0, 1);
//   float R13 = R(0, 2);
//   float R21 = R(1, 0);
//   float R22 = R(1, 1);
//   float R23 = R(1, 2);
//   float R31 = R(2, 0);
//   float R32 = R(2, 1);
//   float R33 = R(2, 2);
//   /*
//     float R11 = state.R(0,0);
//     float R12 = state.R(0,1);
//     float R13 = state.R(0,2);
//     float R21 = state.R(1,0);
//     float R22 = state.R(1,1);
//     float R23 = state.R(1,2);
//     float R31 = state.R(2,0);
//     float R32 = state.R(2,1);
//     float R33 = state.R(2,2);
//   */
//   float Om1 = state.omega(0);
//   float Om2 = state.omega(1);
//   float Om3 = state.omega(2);

//   float Rd11 =
//     cmd.qw * cmd.qw + cmd.qx * cmd.qx - cmd.qy * cmd.qy - cmd.qz * cmd.qz;
//   float Rd12 = 2 * (cmd.qx * cmd.qy - cmd.qw * cmd.qz);
//   float Rd13 = 2 * (cmd.qx * cmd.qz + cmd.qw * cmd.qy);
//   float Rd21 = 2 * (cmd.qx * cmd.qy + cmd.qw * cmd.qz);
//   float Rd22 =
//     cmd.qw * cmd.qw - cmd.qx * cmd.qx + cmd.qy * cmd.qy - cmd.qz * cmd.qz;
//   float Rd23 = 2 * (cmd.qy * cmd.qz - cmd.qw * cmd.qx);
//   float Rd31 = 2 * (cmd.qx * cmd.qz - cmd.qw * cmd.qy);
//   float Rd32 = 2 * (cmd.qy * cmd.qz + cmd.qw * cmd.qx);
//   float Rd33 =
//     cmd.qw * cmd.qw - cmd.qx * cmd.qx - cmd.qy * cmd.qy + cmd.qz * cmd.qz;

//   float Psi = 0.5f * (3.0f - (Rd11 * R11 + Rd21 * R21 + Rd31 * R31 +
//                               Rd12 * R12 + Rd22 * R22 + Rd32 * R32 +
//                               Rd13 * R13 + Rd23 * R23 + Rd33 * R33));

//   float force = 0;
//   if (Psi < 1.0f) // Position control stability guaranteed only when Psi < 1
//     force = cmd.force[0] * R13 + cmd.force[1] * R23 + cmd.force[2] * R33;

//   float eR1 = 0.5f * (R12 * Rd13 - R13 * Rd12 + R22 * Rd23 - R23 * Rd22 +
//                       R32 * Rd33 - R33 * Rd32);
//   float eR2 = 0.5f * (R13 * Rd11 - R11 * Rd13 - R21 * Rd23 + R23 * Rd21 -
//                       R31 * Rd33 + R33 * Rd31);
//   float eR3 = 0.5f * (R11 * Rd12 - R12 * Rd11 + R21 * Rd22 - R22 * Rd21 +
//                       R31 * Rd32 - R32 * Rd31);

//   float eOm1 = Om1;
//   float eOm2 = Om2;
//   float eOm3 = Om3;

//   float in1 = Om2 * (I[2][0] * Om1 + I[2][1] * Om2 + I[2][2] * Om3) -
//               Om3 * (I[1][0] * Om1 + I[1][1] * Om2 + I[1][2] * Om3);
//   float in2 = Om3 * (I[0][0] * Om1 + I[0][1] * Om2 + I[0][2] * Om3) -
//               Om1 * (I[2][0] * Om1 + I[2][1] * Om2 + I[2][2] * Om3);
//   float in3 = Om1 * (I[1][0] * Om1 + I[1][1] * Om2 + I[1][2] * Om3) -
//               Om2 * (I[0][0] * Om1 + I[0][1] * Om2 + I[0][2] * Om3);
//   /*
//     // Robust Control --------------------------------------------
//     float c2       = 0.6;
//     float epsilonR = 0.04;
//     float deltaR   = 0.1;
//     float eA1 = eOm1 + c2 * 1.0/I[0][0] * eR1;
//     float eA2 = eOm2 + c2 * 1.0/I[1][1] * eR2;
//     float eA3 = eOm3 + c2 * 1.0/I[2][2] * eR3;
//     float neA = sqrt(eA1*eA1 + eA2*eA2 + eA3*eA3);
//     float muR1 = -deltaR*deltaR * eA1 / (deltaR * neA + epsilonR);
//     float muR2 = -deltaR*deltaR * eA2 / (deltaR * neA + epsilonR);
//     float muR3 = -deltaR*deltaR * eA3 / (deltaR * neA + epsilonR);
//     // Robust Control --------------------------------------------
//   */
//   float M1 = -cmd.kR[0] * eR1 - cmd.kOm[0] * eOm1 + in1; // - I[0][0]*muR1;
//   float M2 = -cmd.kR[1] * eR2 - cmd.kOm[1] * eOm2 + in2; // - I[1][1]*muR2;
//   float M3 = -cmd.kR[2] * eR3 - cmd.kOm[2] * eOm3 + in3; // - I[2][2]*muR3;

//   float w_sq[4];
//   w_sq[0] = force / (4 * kf) - M2 / (2 * d * kf) + M3 / (4 * km);
//   w_sq[1] = force / (4 * kf) + M2 / (2 * d * kf) + M3 / (4 * km);
//   w_sq[2] = force / (4 * kf) + M1 / (2 * d * kf) - M3 / (4 * km);
//   w_sq[3] = force / (4 * kf) - M1 / (2 * d * kf) - M3 / (4 * km);

//   Control control;
//   for (int i = 0; i < 4; i++)
//   {
//     if (w_sq[i] < 0)
//       w_sq[i] = 0;

//     control.rpm[i] = sqrtf(w_sq[i]);
//   }
//   return control;
// }
