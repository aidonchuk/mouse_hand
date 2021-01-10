import collections
from statistics import mean

import cv2
import numpy as np
from mediapipe.python.solutions.drawing_utils import DrawingSpec, RED_COLOR
from numpy.linalg import lstsq
from scipy import stats

landmark_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR)

deque_l = 5

x0_d = collections.deque(deque_l * [0.], deque_l)
y0_d = collections.deque(deque_l * [0.], deque_l)

x1_d = collections.deque(deque_l * [0.], deque_l)
y1_d = collections.deque(deque_l * [0.], deque_l)

z_val_f_d = collections.deque(deque_l * [0.], deque_l)
z_val_s_d = collections.deque(deque_l * [0.], deque_l)

m_xy_d = collections.deque(deque_l * [0.], deque_l)
c_xy_d = collections.deque(deque_l * [0.], deque_l)

x_d = collections.deque(deque_l * [0.], deque_l)
y_d = collections.deque(deque_l * [0.], deque_l)
z_d = collections.deque(deque_l * [0.], deque_l)


def read_depth_png(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def calc_z_score(v):
    r = []
    z = stats.zscore(v)
    for i, j in enumerate(z):
        if -1.3 <= j <= 1.3:
            r.append(v[i])

    return r


def get_area_mean_z_val(depth_image, a, b):
    all_depth = []
    for i in range(-14, 14):
        for j in range(-14, 14):
            all_depth.append(depth_image[a + 0 if a + i >= depth_image.shape[0] else a + i,
                                         b + 0 if b + j >= depth_image.shape[1] else b + j])
    all_depth = np.array(all_depth, dtype=np.float32)
    # all_depth = calc_z_score(all_depth)
    return np.mean(all_depth)


def get_filtered_values(depth_image, xy0, xy1):
    global x0_d, y0_d, x1_d, y1_d, m_xy_d, c_xy_d, z_val_f_d, z_val_s_d, x_d, y_d, z_d

    x0_d.append(float(xy0[1]))
    x0_f = round(mean(x0_d))

    y0_d.append(float(xy0[0]))
    y0_f = round(mean(y0_d))

    x1_d.append(float(xy1[1]))
    x1_f = round(mean(x1_d))

    y1_d.append(float(xy1[0]))
    y1_f = round(mean(y1_d))

    z_val_f = get_area_mean_z_val(depth_image, x0_f, y0_f)
    z_val_f_d.append(float(z_val_f))
    z_val_f = mean(z_val_f_d)

    z_val_s = get_area_mean_z_val(depth_image, x1_f, y1_f)
    z_val_s_d.append(float(z_val_s))
    z_val_s = mean(z_val_s_d)

    points = [(y0_f, x0_f), (y1_f, x1_f)]
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    m_xy_d.append(float(m))
    m_xy = mean(m_xy_d)
    c_xy_d.append(float(c))
    c_xy = mean(c_xy_d)

    a0, a1, a2, a3 = equation_plane()
    x, y, z = line_plane_intersection(y0_f, x0_f, z_val_s, y1_f, x1_f, z_val_f, a0, a1, a2, a3)

    x_d.append(float(x))
    x = round(mean(x_d))
    y_d.append(float(y))
    y = round(mean(y_d))
    z_d.append(float(z))
    z = round(mean(z_d))

    return z_val_f, z_val_s, m_xy, c_xy, (y0_f, x0_f), (y1_f, x1_f), x, y, z


def get_right_index(handedness_dict):
    right_hand_index = -1

    if len(handedness_dict) > 1:
        hand_0 = handedness_dict[0]['classification'][0]
        hand_1 = handedness_dict[1]['classification'][0]
        if hand_0['label'] == hand_1['label']:
            if hand_0['label'] == 'Left':
                if hand_0['score'] > hand_1['score']:
                    right_hand_index = 0
                else:
                    right_hand_index = 1
        else:
            if hand_0['label'] == 'Left':
                right_hand_index = 0
            else:
                right_hand_index = 1
    else:
        hand_0 = handedness_dict[0]['classification'][0]
        if hand_0['label'] == 'Left':
            right_hand_index = 0
        else:
            right_hand_index = 1

    return right_hand_index


def draw_circles(color_image, depth_color_image, landmark_px):
    cv2.circle(color_image, landmark_px, landmark_drawing_spec.circle_radius,
               landmark_drawing_spec.color, 2)
    cv2.circle(depth_color_image, landmark_px, landmark_drawing_spec.circle_radius,
               (0, 0, 0), 2)


def line_plane_intersection(px, py, pz, qx, qy, qz, a, b, c, d):
    """
    Points p with px py pz and q that define a line, and the plane
    of formula ax+by+cz+d = 0, returns the intersection point or null if none.
    """

    tDenom = a * (qx - px) + b * (qy - py) + c * (qz - pz)
    if tDenom == 0:
        return None

    t = - (a * px + b * py + c * pz + d) / tDenom
    return (px + t * (qx - px)), (py + t * (qy - py)), (pz + t * (qz - pz))  # x, y, z


def equation_plane(x1=1, y1=0, z1=1, x2=3, y2=0, z2=5, x3=7, y3=0, z3=3):
    p1 = np.array([x1, y1, z1])
    p2 = np.array([x2, y2, z2])
    p3 = np.array([x3, y3, z3])

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    return a, b, c, d


def draw_cam_out(color_image, depth_color_image, xy0_f, xy1_f, c_xy, m_xy, scale=2.5):
    ci = cv2.resize(color_image, (int(color_image.shape[1] / scale), int(color_image.shape[0] / scale)))
    dci = cv2.resize(depth_color_image,
                     (int(depth_color_image.shape[1] / scale), int(depth_color_image.shape[0] / scale)))

    draw_circles(ci, dci, (int(xy0_f[0] / scale), int(xy0_f[1] / scale)))
    draw_circles(ci, dci, (int(xy1_f[0] / scale), int(xy1_f[1] / scale)))

    y_max = 2
    x = (y_max - 1 * c_xy) / m_xy
    if x > 0:
        ci = cv2.line(ci, (int(xy1_f[0] / scale), int(xy1_f[1] / scale)), (round(x / scale), y_max), (0, 255, 0), 2)

    images_stacked = np.vstack((ci, dci))
    cv2.imshow('Crazy Hands', images_stacked)
    if cv2.waitKey(5) & 0xFF == 27:
        return True

    return False
