import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pyrealsense2.pyrealsense2 as rs
from google.protobuf.json_format import MessageToDict
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from pynput import keyboard

from utils.common import get_filtered_values, draw_cam_out, get_right_index
from utils.hard_reset import hardware_reset
from utils.set_options import set_short_range

pyautogui.FAILSAFE = False

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.9)


def on_press(key):
    if key == keyboard.Key.ctrl:
        pyautogui.leftClick()
    if key == keyboard.Key.alt:
        pyautogui.rightClick()


def get_color_depth(pipeline, align, colorizer):
    frames = pipeline.wait_for_frames(timeout_ms=15000)
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None, None

    depth_image = np.asanyarray(depth_frame.get_data())
    depth_color_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    color_image = np.asanyarray(color_frame.get_data())

    depth_color_image = cv2.cvtColor(cv2.flip(cv2.flip(depth_color_image, 1), 0), cv2.COLOR_BGR2RGB)
    color_image = cv2.cvtColor(cv2.flip(cv2.flip(color_image, 1), 0), cv2.COLOR_BGR2RGB)
    depth_image = np.flipud(np.fliplr(depth_image))

    depth_color_image = cv2.resize(depth_color_image, (1280 * 2, 720 * 2))
    color_image = cv2.resize(color_image, (1280 * 2, 720 * 2))
    depth_image = cv2.resize(depth_image, (1280 * 2, 720 * 2))

    return color_image, depth_color_image, depth_image


def get_right_hand_coords(color_image, depth_color_image):
    color_image.flags.writeable = False
    results = hands.process(color_image)

    color_image.flags.writeable = True
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    handedness_dict = []

    idx_to_coordinates = {}
    xy0, xy1 = None, None
    if results.multi_hand_landmarks:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            handedness_dict.append(MessageToDict(hand_handedness))

        right_hand_index = get_right_index(handedness_dict)

        if right_hand_index != -1:
            for i, landmark_list in enumerate(results.multi_hand_landmarks):
                if i == right_hand_index:
                    image_rows, image_cols, _ = color_image.shape
                    for idx, landmark in enumerate(landmark_list.landmark):
                        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                       image_cols, image_rows)
                        if landmark_px:
                            idx_to_coordinates[idx] = landmark_px

            for i, landmark_px in enumerate(idx_to_coordinates.values()):
                if i == 5:
                    xy0 = landmark_px
                if i == 7:
                    xy1 = landmark_px
                    break
    return color_image, depth_color_image, xy0, xy1, idx_to_coordinates


def start():
    pipeline = rs.pipeline()
    config = rs.config()

    print("Start load conf")
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    set_short_range(depth_sensor)
    colorizer = rs.colorizer()
    print("Conf loaded")
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            color_image, depth_color_image, depth_image = get_color_depth(pipeline, align, colorizer)
            if color_image is None and color_image is None and color_image is None:
                continue

            color_image, depth_color_image, xy0, xy1, idx_to_coordinates = get_right_hand_coords(color_image,
                                                                                                 depth_color_image)
            if xy0 is not None or xy1 is not None:
                z_val_f, z_val_s, m_xy, c_xy, xy0_f, xy1_f, x, y, z = get_filtered_values(depth_image, xy0, xy1)
                pyautogui.moveTo(int(x), int(3500 - z))  # , duration=0.05
                if draw_cam_out(color_image, depth_color_image, xy0_f, xy1_f, c_xy, m_xy):
                    break
    finally:
        hands.close()
        pipeline.stop()


hardware_reset()
listener = keyboard.Listener(on_press=on_press)
listener.start()
start()
