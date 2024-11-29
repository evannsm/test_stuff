// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from px4_msgs:msg/QshellReq.idl
// generated code does not contain a copyright notice

#ifndef PX4_MSGS__MSG__DETAIL__QSHELL_REQ__TRAITS_HPP_
#define PX4_MSGS__MSG__DETAIL__QSHELL_REQ__TRAITS_HPP_

#include "px4_msgs/msg/detail/qshell_req__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

namespace rosidl_generator_traits
{

inline void to_yaml(
  const px4_msgs::msg::QshellReq & msg,
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

  // member: cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.cmd.size() == 0) {
      out << "cmd: []\n";
    } else {
      out << "cmd:\n";
      for (auto item : msg.cmd) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: strlen
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "strlen: ";
    value_to_yaml(msg.strlen, out);
    out << "\n";
  }

  // member: request_sequence
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "request_sequence: ";
    value_to_yaml(msg.request_sequence, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const px4_msgs::msg::QshellReq & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<px4_msgs::msg::QshellReq>()
{
  return "px4_msgs::msg::QshellReq";
}

template<>
inline const char * name<px4_msgs::msg::QshellReq>()
{
  return "px4_msgs/msg/QshellReq";
}

template<>
struct has_fixed_size<px4_msgs::msg::QshellReq>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<px4_msgs::msg::QshellReq>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<px4_msgs::msg::QshellReq>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PX4_MSGS__MSG__DETAIL__QSHELL_REQ__TRAITS_HPP_
