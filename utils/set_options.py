'''
option.visual_preset,
option.frames_queue_size,
option.error_polling_enabled,
option.depth_units,
option.inter_cam_sync_mode,
option.ldd_temperature,
option.mc_temperature,
option.ma_temperature,
option.global_time_enabled,
option.apd_temperature,
option.depth_offset,
option.zero_order_enabled,
option.freefall_detection_enabled,
option.sensor_mode,
option.trigger_camera_accuracy_health,
option.reset_camera_accuracy_health,
option.host_performance,
option.humidity_temperature,
option.enable_max_usable_range,
option.alternate_ir,
option.noise_estimation,
option.enable_ir_reflectivity,
option.digital_gain,
option.laser_power,
option.confidence_threshold,
option.min_distance,
option.receiver_gain,
option.post_processing_sharpening,
option.pre_processing_sharpening,
option.noise_filtering,
option.invalidation_bypass
'''
import time

from pyrealsense2.pyrealsense2 import option


def set_short_range_small(depth_sensor):
    depth_sensor.set_option(option.min_distance, 190)
    depth_sensor.set_option(option.noise_filtering, 4)
    depth_sensor.set_option(option.post_processing_sharpening, 1)
    depth_sensor.set_option(option.visual_preset, 5)
    depth_sensor.set_option(option.noise_filtering, 4)
    depth_sensor.set_option(option.laser_power, 71)

    return depth_sensor


def set_short_range(depth_sensor):
    depth_sensor.set_option(option.alternate_ir, 0.0)
    # depth_sensor.set_option(option.apd_temperature, -9999)
    # depth_sensor.set_option(option.depth_offset, 4.5)
    # depth_sensor.set_option(option.depth_units, 0.000250000011874363)
    depth_sensor.set_option(option.digital_gain, 2.0)
    depth_sensor.set_option(option.enable_ir_reflectivity, 0.0)
    depth_sensor.set_option(option.enable_max_usable_range, 0.0)
    depth_sensor.set_option(option.error_polling_enabled, 1.0)
    depth_sensor.set_option(option.frames_queue_size, 16.0)
    depth_sensor.set_option(option.freefall_detection_enabled, 1.0)
    depth_sensor.set_option(option.global_time_enabled, 0.0)
    depth_sensor.set_option(option.host_performance, 0.0)
    # depth_sensor.set_option(option.humidity_temperature, 36.6105880737305)
    depth_sensor.set_option(option.inter_cam_sync_mode, 0.0)
    depth_sensor.set_option(option.invalidation_bypass, 0.0)
    # depth_sensor.set_option(option.ldd_temperature, 36.6820793151855)
    depth_sensor.set_option(option.laser_power, 71)
    # depth_sensor.set_option(option.ma_temperature, 36.6820793151855)
    # depth_sensor.set_option(option.mc_temperature, 36.570125579834)
    depth_sensor.set_option(option.min_distance, 190)
    # depth_sensor.set_option(option.noise_estimation, 0.0)
    depth_sensor.set_option(option.noise_filtering, 4.0)
    depth_sensor.set_option(option.post_processing_sharpening, 1)
    depth_sensor.set_option(option.pre_processing_sharpening, 0.0)
    depth_sensor.set_option(option.receiver_gain, 18)
    depth_sensor.set_option(option.reset_camera_accuracy_health, 0.0)
    depth_sensor.set_option(option.sensor_mode, 0.0)
    depth_sensor.set_option(option.trigger_camera_accuracy_health, 0.0)
    depth_sensor.set_option(option.visual_preset, 5)
    depth_sensor.set_option(option.zero_order_enabled, 0.0)

    depth_sensor.set_option(option.confidence_threshold, 1.0)

    time.sleep(10)

    return depth_sensor
