// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from px4_msgs:msg/EstimatorStatusFlags.idl
// generated code does not contain a copyright notice
#include "px4_msgs/msg/detail/estimator_status_flags__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>


bool
px4_msgs__msg__EstimatorStatusFlags__init(px4_msgs__msg__EstimatorStatusFlags * msg)
{
  if (!msg) {
    return false;
  }
  // timestamp
  // timestamp_sample
  // control_status_changes
  // cs_tilt_align
  // cs_yaw_align
  // cs_gps
  // cs_opt_flow
  // cs_mag_hdg
  // cs_mag_3d
  // cs_mag_dec
  // cs_in_air
  // cs_wind
  // cs_baro_hgt
  // cs_rng_hgt
  // cs_gps_hgt
  // cs_ev_pos
  // cs_ev_yaw
  // cs_ev_hgt
  // cs_fuse_beta
  // cs_mag_field_disturbed
  // cs_fixed_wing
  // cs_mag_fault
  // cs_fuse_aspd
  // cs_gnd_effect
  // cs_rng_stuck
  // cs_gps_yaw
  // cs_mag_aligned_in_flight
  // cs_ev_vel
  // cs_synthetic_mag_z
  // cs_vehicle_at_rest
  // cs_gps_yaw_fault
  // cs_rng_fault
  // cs_inertial_dead_reckoning
  // cs_wind_dead_reckoning
  // cs_rng_kin_consistent
  // cs_fake_pos
  // cs_fake_hgt
  // cs_gravity_vector
  // fault_status_changes
  // fs_bad_mag_x
  // fs_bad_mag_y
  // fs_bad_mag_z
  // fs_bad_hdg
  // fs_bad_mag_decl
  // fs_bad_airspeed
  // fs_bad_sideslip
  // fs_bad_optflow_x
  // fs_bad_optflow_y
  // fs_bad_vel_n
  // fs_bad_vel_e
  // fs_bad_vel_d
  // fs_bad_pos_n
  // fs_bad_pos_e
  // fs_bad_pos_d
  // fs_bad_acc_bias
  // fs_bad_acc_vertical
  // fs_bad_acc_clipping
  // innovation_fault_status_changes
  // reject_hor_vel
  // reject_ver_vel
  // reject_hor_pos
  // reject_ver_pos
  // reject_yaw
  // reject_airspeed
  // reject_sideslip
  // reject_hagl
  // reject_optflow_x
  // reject_optflow_y
  return true;
}

void
px4_msgs__msg__EstimatorStatusFlags__fini(px4_msgs__msg__EstimatorStatusFlags * msg)
{
  if (!msg) {
    return;
  }
  // timestamp
  // timestamp_sample
  // control_status_changes
  // cs_tilt_align
  // cs_yaw_align
  // cs_gps
  // cs_opt_flow
  // cs_mag_hdg
  // cs_mag_3d
  // cs_mag_dec
  // cs_in_air
  // cs_wind
  // cs_baro_hgt
  // cs_rng_hgt
  // cs_gps_hgt
  // cs_ev_pos
  // cs_ev_yaw
  // cs_ev_hgt
  // cs_fuse_beta
  // cs_mag_field_disturbed
  // cs_fixed_wing
  // cs_mag_fault
  // cs_fuse_aspd
  // cs_gnd_effect
  // cs_rng_stuck
  // cs_gps_yaw
  // cs_mag_aligned_in_flight
  // cs_ev_vel
  // cs_synthetic_mag_z
  // cs_vehicle_at_rest
  // cs_gps_yaw_fault
  // cs_rng_fault
  // cs_inertial_dead_reckoning
  // cs_wind_dead_reckoning
  // cs_rng_kin_consistent
  // cs_fake_pos
  // cs_fake_hgt
  // cs_gravity_vector
  // fault_status_changes
  // fs_bad_mag_x
  // fs_bad_mag_y
  // fs_bad_mag_z
  // fs_bad_hdg
  // fs_bad_mag_decl
  // fs_bad_airspeed
  // fs_bad_sideslip
  // fs_bad_optflow_x
  // fs_bad_optflow_y
  // fs_bad_vel_n
  // fs_bad_vel_e
  // fs_bad_vel_d
  // fs_bad_pos_n
  // fs_bad_pos_e
  // fs_bad_pos_d
  // fs_bad_acc_bias
  // fs_bad_acc_vertical
  // fs_bad_acc_clipping
  // innovation_fault_status_changes
  // reject_hor_vel
  // reject_ver_vel
  // reject_hor_pos
  // reject_ver_pos
  // reject_yaw
  // reject_airspeed
  // reject_sideslip
  // reject_hagl
  // reject_optflow_x
  // reject_optflow_y
}

bool
px4_msgs__msg__EstimatorStatusFlags__are_equal(const px4_msgs__msg__EstimatorStatusFlags * lhs, const px4_msgs__msg__EstimatorStatusFlags * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // timestamp
  if (lhs->timestamp != rhs->timestamp) {
    return false;
  }
  // timestamp_sample
  if (lhs->timestamp_sample != rhs->timestamp_sample) {
    return false;
  }
  // control_status_changes
  if (lhs->control_status_changes != rhs->control_status_changes) {
    return false;
  }
  // cs_tilt_align
  if (lhs->cs_tilt_align != rhs->cs_tilt_align) {
    return false;
  }
  // cs_yaw_align
  if (lhs->cs_yaw_align != rhs->cs_yaw_align) {
    return false;
  }
  // cs_gps
  if (lhs->cs_gps != rhs->cs_gps) {
    return false;
  }
  // cs_opt_flow
  if (lhs->cs_opt_flow != rhs->cs_opt_flow) {
    return false;
  }
  // cs_mag_hdg
  if (lhs->cs_mag_hdg != rhs->cs_mag_hdg) {
    return false;
  }
  // cs_mag_3d
  if (lhs->cs_mag_3d != rhs->cs_mag_3d) {
    return false;
  }
  // cs_mag_dec
  if (lhs->cs_mag_dec != rhs->cs_mag_dec) {
    return false;
  }
  // cs_in_air
  if (lhs->cs_in_air != rhs->cs_in_air) {
    return false;
  }
  // cs_wind
  if (lhs->cs_wind != rhs->cs_wind) {
    return false;
  }
  // cs_baro_hgt
  if (lhs->cs_baro_hgt != rhs->cs_baro_hgt) {
    return false;
  }
  // cs_rng_hgt
  if (lhs->cs_rng_hgt != rhs->cs_rng_hgt) {
    return false;
  }
  // cs_gps_hgt
  if (lhs->cs_gps_hgt != rhs->cs_gps_hgt) {
    return false;
  }
  // cs_ev_pos
  if (lhs->cs_ev_pos != rhs->cs_ev_pos) {
    return false;
  }
  // cs_ev_yaw
  if (lhs->cs_ev_yaw != rhs->cs_ev_yaw) {
    return false;
  }
  // cs_ev_hgt
  if (lhs->cs_ev_hgt != rhs->cs_ev_hgt) {
    return false;
  }
  // cs_fuse_beta
  if (lhs->cs_fuse_beta != rhs->cs_fuse_beta) {
    return false;
  }
  // cs_mag_field_disturbed
  if (lhs->cs_mag_field_disturbed != rhs->cs_mag_field_disturbed) {
    return false;
  }
  // cs_fixed_wing
  if (lhs->cs_fixed_wing != rhs->cs_fixed_wing) {
    return false;
  }
  // cs_mag_fault
  if (lhs->cs_mag_fault != rhs->cs_mag_fault) {
    return false;
  }
  // cs_fuse_aspd
  if (lhs->cs_fuse_aspd != rhs->cs_fuse_aspd) {
    return false;
  }
  // cs_gnd_effect
  if (lhs->cs_gnd_effect != rhs->cs_gnd_effect) {
    return false;
  }
  // cs_rng_stuck
  if (lhs->cs_rng_stuck != rhs->cs_rng_stuck) {
    return false;
  }
  // cs_gps_yaw
  if (lhs->cs_gps_yaw != rhs->cs_gps_yaw) {
    return false;
  }
  // cs_mag_aligned_in_flight
  if (lhs->cs_mag_aligned_in_flight != rhs->cs_mag_aligned_in_flight) {
    return false;
  }
  // cs_ev_vel
  if (lhs->cs_ev_vel != rhs->cs_ev_vel) {
    return false;
  }
  // cs_synthetic_mag_z
  if (lhs->cs_synthetic_mag_z != rhs->cs_synthetic_mag_z) {
    return false;
  }
  // cs_vehicle_at_rest
  if (lhs->cs_vehicle_at_rest != rhs->cs_vehicle_at_rest) {
    return false;
  }
  // cs_gps_yaw_fault
  if (lhs->cs_gps_yaw_fault != rhs->cs_gps_yaw_fault) {
    return false;
  }
  // cs_rng_fault
  if (lhs->cs_rng_fault != rhs->cs_rng_fault) {
    return false;
  }
  // cs_inertial_dead_reckoning
  if (lhs->cs_inertial_dead_reckoning != rhs->cs_inertial_dead_reckoning) {
    return false;
  }
  // cs_wind_dead_reckoning
  if (lhs->cs_wind_dead_reckoning != rhs->cs_wind_dead_reckoning) {
    return false;
  }
  // cs_rng_kin_consistent
  if (lhs->cs_rng_kin_consistent != rhs->cs_rng_kin_consistent) {
    return false;
  }
  // cs_fake_pos
  if (lhs->cs_fake_pos != rhs->cs_fake_pos) {
    return false;
  }
  // cs_fake_hgt
  if (lhs->cs_fake_hgt != rhs->cs_fake_hgt) {
    return false;
  }
  // cs_gravity_vector
  if (lhs->cs_gravity_vector != rhs->cs_gravity_vector) {
    return false;
  }
  // fault_status_changes
  if (lhs->fault_status_changes != rhs->fault_status_changes) {
    return false;
  }
  // fs_bad_mag_x
  if (lhs->fs_bad_mag_x != rhs->fs_bad_mag_x) {
    return false;
  }
  // fs_bad_mag_y
  if (lhs->fs_bad_mag_y != rhs->fs_bad_mag_y) {
    return false;
  }
  // fs_bad_mag_z
  if (lhs->fs_bad_mag_z != rhs->fs_bad_mag_z) {
    return false;
  }
  // fs_bad_hdg
  if (lhs->fs_bad_hdg != rhs->fs_bad_hdg) {
    return false;
  }
  // fs_bad_mag_decl
  if (lhs->fs_bad_mag_decl != rhs->fs_bad_mag_decl) {
    return false;
  }
  // fs_bad_airspeed
  if (lhs->fs_bad_airspeed != rhs->fs_bad_airspeed) {
    return false;
  }
  // fs_bad_sideslip
  if (lhs->fs_bad_sideslip != rhs->fs_bad_sideslip) {
    return false;
  }
  // fs_bad_optflow_x
  if (lhs->fs_bad_optflow_x != rhs->fs_bad_optflow_x) {
    return false;
  }
  // fs_bad_optflow_y
  if (lhs->fs_bad_optflow_y != rhs->fs_bad_optflow_y) {
    return false;
  }
  // fs_bad_vel_n
  if (lhs->fs_bad_vel_n != rhs->fs_bad_vel_n) {
    return false;
  }
  // fs_bad_vel_e
  if (lhs->fs_bad_vel_e != rhs->fs_bad_vel_e) {
    return false;
  }
  // fs_bad_vel_d
  if (lhs->fs_bad_vel_d != rhs->fs_bad_vel_d) {
    return false;
  }
  // fs_bad_pos_n
  if (lhs->fs_bad_pos_n != rhs->fs_bad_pos_n) {
    return false;
  }
  // fs_bad_pos_e
  if (lhs->fs_bad_pos_e != rhs->fs_bad_pos_e) {
    return false;
  }
  // fs_bad_pos_d
  if (lhs->fs_bad_pos_d != rhs->fs_bad_pos_d) {
    return false;
  }
  // fs_bad_acc_bias
  if (lhs->fs_bad_acc_bias != rhs->fs_bad_acc_bias) {
    return false;
  }
  // fs_bad_acc_vertical
  if (lhs->fs_bad_acc_vertical != rhs->fs_bad_acc_vertical) {
    return false;
  }
  // fs_bad_acc_clipping
  if (lhs->fs_bad_acc_clipping != rhs->fs_bad_acc_clipping) {
    return false;
  }
  // innovation_fault_status_changes
  if (lhs->innovation_fault_status_changes != rhs->innovation_fault_status_changes) {
    return false;
  }
  // reject_hor_vel
  if (lhs->reject_hor_vel != rhs->reject_hor_vel) {
    return false;
  }
  // reject_ver_vel
  if (lhs->reject_ver_vel != rhs->reject_ver_vel) {
    return false;
  }
  // reject_hor_pos
  if (lhs->reject_hor_pos != rhs->reject_hor_pos) {
    return false;
  }
  // reject_ver_pos
  if (lhs->reject_ver_pos != rhs->reject_ver_pos) {
    return false;
  }
  // reject_yaw
  if (lhs->reject_yaw != rhs->reject_yaw) {
    return false;
  }
  // reject_airspeed
  if (lhs->reject_airspeed != rhs->reject_airspeed) {
    return false;
  }
  // reject_sideslip
  if (lhs->reject_sideslip != rhs->reject_sideslip) {
    return false;
  }
  // reject_hagl
  if (lhs->reject_hagl != rhs->reject_hagl) {
    return false;
  }
  // reject_optflow_x
  if (lhs->reject_optflow_x != rhs->reject_optflow_x) {
    return false;
  }
  // reject_optflow_y
  if (lhs->reject_optflow_y != rhs->reject_optflow_y) {
    return false;
  }
  return true;
}

bool
px4_msgs__msg__EstimatorStatusFlags__copy(
  const px4_msgs__msg__EstimatorStatusFlags * input,
  px4_msgs__msg__EstimatorStatusFlags * output)
{
  if (!input || !output) {
    return false;
  }
  // timestamp
  output->timestamp = input->timestamp;
  // timestamp_sample
  output->timestamp_sample = input->timestamp_sample;
  // control_status_changes
  output->control_status_changes = input->control_status_changes;
  // cs_tilt_align
  output->cs_tilt_align = input->cs_tilt_align;
  // cs_yaw_align
  output->cs_yaw_align = input->cs_yaw_align;
  // cs_gps
  output->cs_gps = input->cs_gps;
  // cs_opt_flow
  output->cs_opt_flow = input->cs_opt_flow;
  // cs_mag_hdg
  output->cs_mag_hdg = input->cs_mag_hdg;
  // cs_mag_3d
  output->cs_mag_3d = input->cs_mag_3d;
  // cs_mag_dec
  output->cs_mag_dec = input->cs_mag_dec;
  // cs_in_air
  output->cs_in_air = input->cs_in_air;
  // cs_wind
  output->cs_wind = input->cs_wind;
  // cs_baro_hgt
  output->cs_baro_hgt = input->cs_baro_hgt;
  // cs_rng_hgt
  output->cs_rng_hgt = input->cs_rng_hgt;
  // cs_gps_hgt
  output->cs_gps_hgt = input->cs_gps_hgt;
  // cs_ev_pos
  output->cs_ev_pos = input->cs_ev_pos;
  // cs_ev_yaw
  output->cs_ev_yaw = input->cs_ev_yaw;
  // cs_ev_hgt
  output->cs_ev_hgt = input->cs_ev_hgt;
  // cs_fuse_beta
  output->cs_fuse_beta = input->cs_fuse_beta;
  // cs_mag_field_disturbed
  output->cs_mag_field_disturbed = input->cs_mag_field_disturbed;
  // cs_fixed_wing
  output->cs_fixed_wing = input->cs_fixed_wing;
  // cs_mag_fault
  output->cs_mag_fault = input->cs_mag_fault;
  // cs_fuse_aspd
  output->cs_fuse_aspd = input->cs_fuse_aspd;
  // cs_gnd_effect
  output->cs_gnd_effect = input->cs_gnd_effect;
  // cs_rng_stuck
  output->cs_rng_stuck = input->cs_rng_stuck;
  // cs_gps_yaw
  output->cs_gps_yaw = input->cs_gps_yaw;
  // cs_mag_aligned_in_flight
  output->cs_mag_aligned_in_flight = input->cs_mag_aligned_in_flight;
  // cs_ev_vel
  output->cs_ev_vel = input->cs_ev_vel;
  // cs_synthetic_mag_z
  output->cs_synthetic_mag_z = input->cs_synthetic_mag_z;
  // cs_vehicle_at_rest
  output->cs_vehicle_at_rest = input->cs_vehicle_at_rest;
  // cs_gps_yaw_fault
  output->cs_gps_yaw_fault = input->cs_gps_yaw_fault;
  // cs_rng_fault
  output->cs_rng_fault = input->cs_rng_fault;
  // cs_inertial_dead_reckoning
  output->cs_inertial_dead_reckoning = input->cs_inertial_dead_reckoning;
  // cs_wind_dead_reckoning
  output->cs_wind_dead_reckoning = input->cs_wind_dead_reckoning;
  // cs_rng_kin_consistent
  output->cs_rng_kin_consistent = input->cs_rng_kin_consistent;
  // cs_fake_pos
  output->cs_fake_pos = input->cs_fake_pos;
  // cs_fake_hgt
  output->cs_fake_hgt = input->cs_fake_hgt;
  // cs_gravity_vector
  output->cs_gravity_vector = input->cs_gravity_vector;
  // fault_status_changes
  output->fault_status_changes = input->fault_status_changes;
  // fs_bad_mag_x
  output->fs_bad_mag_x = input->fs_bad_mag_x;
  // fs_bad_mag_y
  output->fs_bad_mag_y = input->fs_bad_mag_y;
  // fs_bad_mag_z
  output->fs_bad_mag_z = input->fs_bad_mag_z;
  // fs_bad_hdg
  output->fs_bad_hdg = input->fs_bad_hdg;
  // fs_bad_mag_decl
  output->fs_bad_mag_decl = input->fs_bad_mag_decl;
  // fs_bad_airspeed
  output->fs_bad_airspeed = input->fs_bad_airspeed;
  // fs_bad_sideslip
  output->fs_bad_sideslip = input->fs_bad_sideslip;
  // fs_bad_optflow_x
  output->fs_bad_optflow_x = input->fs_bad_optflow_x;
  // fs_bad_optflow_y
  output->fs_bad_optflow_y = input->fs_bad_optflow_y;
  // fs_bad_vel_n
  output->fs_bad_vel_n = input->fs_bad_vel_n;
  // fs_bad_vel_e
  output->fs_bad_vel_e = input->fs_bad_vel_e;
  // fs_bad_vel_d
  output->fs_bad_vel_d = input->fs_bad_vel_d;
  // fs_bad_pos_n
  output->fs_bad_pos_n = input->fs_bad_pos_n;
  // fs_bad_pos_e
  output->fs_bad_pos_e = input->fs_bad_pos_e;
  // fs_bad_pos_d
  output->fs_bad_pos_d = input->fs_bad_pos_d;
  // fs_bad_acc_bias
  output->fs_bad_acc_bias = input->fs_bad_acc_bias;
  // fs_bad_acc_vertical
  output->fs_bad_acc_vertical = input->fs_bad_acc_vertical;
  // fs_bad_acc_clipping
  output->fs_bad_acc_clipping = input->fs_bad_acc_clipping;
  // innovation_fault_status_changes
  output->innovation_fault_status_changes = input->innovation_fault_status_changes;
  // reject_hor_vel
  output->reject_hor_vel = input->reject_hor_vel;
  // reject_ver_vel
  output->reject_ver_vel = input->reject_ver_vel;
  // reject_hor_pos
  output->reject_hor_pos = input->reject_hor_pos;
  // reject_ver_pos
  output->reject_ver_pos = input->reject_ver_pos;
  // reject_yaw
  output->reject_yaw = input->reject_yaw;
  // reject_airspeed
  output->reject_airspeed = input->reject_airspeed;
  // reject_sideslip
  output->reject_sideslip = input->reject_sideslip;
  // reject_hagl
  output->reject_hagl = input->reject_hagl;
  // reject_optflow_x
  output->reject_optflow_x = input->reject_optflow_x;
  // reject_optflow_y
  output->reject_optflow_y = input->reject_optflow_y;
  return true;
}

px4_msgs__msg__EstimatorStatusFlags *
px4_msgs__msg__EstimatorStatusFlags__create()
{
  px4_msgs__msg__EstimatorStatusFlags * msg = (px4_msgs__msg__EstimatorStatusFlags *)malloc(sizeof(px4_msgs__msg__EstimatorStatusFlags));
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(px4_msgs__msg__EstimatorStatusFlags));
  bool success = px4_msgs__msg__EstimatorStatusFlags__init(msg);
  if (!success) {
    free(msg);
    return NULL;
  }
  return msg;
}

void
px4_msgs__msg__EstimatorStatusFlags__destroy(px4_msgs__msg__EstimatorStatusFlags * msg)
{
  if (msg) {
    px4_msgs__msg__EstimatorStatusFlags__fini(msg);
  }
  free(msg);
}


bool
px4_msgs__msg__EstimatorStatusFlags__Sequence__init(px4_msgs__msg__EstimatorStatusFlags__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  px4_msgs__msg__EstimatorStatusFlags * data = NULL;
  if (size) {
    data = (px4_msgs__msg__EstimatorStatusFlags *)calloc(size, sizeof(px4_msgs__msg__EstimatorStatusFlags));
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = px4_msgs__msg__EstimatorStatusFlags__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        px4_msgs__msg__EstimatorStatusFlags__fini(&data[i - 1]);
      }
      free(data);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
px4_msgs__msg__EstimatorStatusFlags__Sequence__fini(px4_msgs__msg__EstimatorStatusFlags__Sequence * array)
{
  if (!array) {
    return;
  }
  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      px4_msgs__msg__EstimatorStatusFlags__fini(&array->data[i]);
    }
    free(array->data);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

px4_msgs__msg__EstimatorStatusFlags__Sequence *
px4_msgs__msg__EstimatorStatusFlags__Sequence__create(size_t size)
{
  px4_msgs__msg__EstimatorStatusFlags__Sequence * array = (px4_msgs__msg__EstimatorStatusFlags__Sequence *)malloc(sizeof(px4_msgs__msg__EstimatorStatusFlags__Sequence));
  if (!array) {
    return NULL;
  }
  bool success = px4_msgs__msg__EstimatorStatusFlags__Sequence__init(array, size);
  if (!success) {
    free(array);
    return NULL;
  }
  return array;
}

void
px4_msgs__msg__EstimatorStatusFlags__Sequence__destroy(px4_msgs__msg__EstimatorStatusFlags__Sequence * array)
{
  if (array) {
    px4_msgs__msg__EstimatorStatusFlags__Sequence__fini(array);
  }
  free(array);
}

bool
px4_msgs__msg__EstimatorStatusFlags__Sequence__are_equal(const px4_msgs__msg__EstimatorStatusFlags__Sequence * lhs, const px4_msgs__msg__EstimatorStatusFlags__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!px4_msgs__msg__EstimatorStatusFlags__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
px4_msgs__msg__EstimatorStatusFlags__Sequence__copy(
  const px4_msgs__msg__EstimatorStatusFlags__Sequence * input,
  px4_msgs__msg__EstimatorStatusFlags__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(px4_msgs__msg__EstimatorStatusFlags);
    px4_msgs__msg__EstimatorStatusFlags * data =
      (px4_msgs__msg__EstimatorStatusFlags *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!px4_msgs__msg__EstimatorStatusFlags__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          px4_msgs__msg__EstimatorStatusFlags__fini(&data[i]);
        }
        free(data);
        return false;
      }
    }
    output->data = data;
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!px4_msgs__msg__EstimatorStatusFlags__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}