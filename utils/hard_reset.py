import time

import pyrealsense2.pyrealsense2 as rs


def hardware_reset():
    ctx = rs.context()
    list = ctx.query_devices()
    for dev in list:
        serial = dev.query_sensors()[0].get_info(rs.camera_info.serial_number)
        print("Reset start")
        print(serial)
        dev.hardware_reset()
        time.sleep(15)
        print("Reset finish")
