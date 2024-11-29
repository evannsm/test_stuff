// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from px4_msgs:msg/VehicleMagnetometer.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__VEHICLE_MAGNETOMETER__TRAITS_HPP_
#define PX4_MSGS__MSG__DETAIL__VEHICLE_MAGNETOMETER__TRAITS_HPP_

#include "px4_msgs/msg/detail/vehicle_magnetometer__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

namespace rosidl_generator_traits
{

inline void to_yaml(
  const px4_msgs::msg::VehicleMagnetometer & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: timestamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "timestamp: ";
    value_to_yaml(msg.timestamp, out);
    out << "\n";
  }

  // member: timestamp_sample
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "timestamp_sample: ";
    value_to_yaml(msg.timestamp_sample, out);
    out << "\n";
  }

  // member: device_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "device_id: ";
    value_to_yaml(msg.device_id, out);
    out << "\n";
  }

  // member: magnetometer_ga
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.magnetometer_ga.size() == 0) {
      out << "magnetometer_ga: []\n";
    } else {
      out << "magnetometer_ga:\n";
      for (auto item : msg.magnetometer_ga) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: calibration_count
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "calibration_count: ";
    value_to_yaml(msg.calibration_count, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const px4_msgs::msg::VehicleMagnetometer & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<px4_msgs::msg::VehicleMagnetometer>()
{
  return "px4_msgs::msg::VehicleMagnetometer";
}

template<>
inline const char * name<px4_msgs::msg::VehicleMagnetometer>()
{
  return "px4_msgs/msg/VehicleMagnetometer";
}

template<>
struct has_fixed_size<px4_msgs::msg::VehicleMagnetometer>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<px4_msgs::msg::VehicleMagnetometer>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<px4_msgs::msg::VehicleMagnetometer>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PX4_MSGS__MSG__DETAIL__VEHICLE_MAGNETOMETER__TRAITS_HPP_
