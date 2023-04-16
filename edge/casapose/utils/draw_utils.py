import cv2
import matplotlib as plt
import numpy as np


def draw_bb(
    xy,
    img,
    color_lines=(0, 0, 255),
    color_points=(0, 255, 255),
    width=2,
    line_type=cv2.LINE_4,
):
    xy = xy.astype(int)
    xy = tuple(map(tuple, xy))
    cv2.line(img, xy[0], xy[1], color_lines, width, line_type)
    cv2.line(img, xy[1], xy[3], color_lines, width, line_type)
    cv2.line(img, xy[3], xy[2], color_lines, width, line_type)
    cv2.line(img, xy[2], xy[0], color_lines, width, line_type)
    cv2.line(img, xy[0], xy[4], color_lines, width, line_type)
    cv2.line(img, xy[4], xy[5], color_lines, width, line_type)
    cv2.line(img, xy[5], xy[7], color_lines, width, line_type)
    cv2.line(img, xy[7], xy[6], color_lines, width, line_type)
    cv2.line(img, xy[6], xy[4], color_lines, width, line_type)
    cv2.line(img, xy[2], xy[6], color_lines, width, line_type)
    cv2.line(img, xy[7], xy[3], color_lines, width, line_type)
    cv2.line(img, xy[1], xy[5], color_lines, width, line_type)

    for p in xy:
        cv2.circle(img, p, 1, color_points, -1)
    return img

def draw_axes(img, keypoints, colors = [(0, 0, 255), (255, 0, 0),(255, 255, 255)], thickness = 1 ):
    # args: image, projected_cuboid keypoints, list of 3 colors to use, tickenss
    # returns the image with the line drawn
    
    # finds the center point
    center = np.mean(keypoints, axis=0)
    center = [int(i) for i in center]
    
    # finds the top of the object    
    point1_top = [(keypoints[1][0] + keypoints[2][0])/2, (keypoints[1][1] + keypoints[2][1])/2]
    point2_top = [(keypoints[0][0] + keypoints[3][0])/2, (keypoints[0][1] + keypoints[3][1])/2]
    top_coords = [(point1_top[0] + point2_top[0])/2, (point1_top[1] + point2_top[1])/2]
    
    
    # finds the right of the top of the object
    point1_right = [(keypoints[3][0] + keypoints[6][0])/2, (keypoints[3][1] + keypoints[6][1])/2]
    point2_right = [(keypoints[2][0] + keypoints[7][0])/2, (keypoints[2][1] + keypoints[7][1])/2]
    right_coords = [(point1_right[0] + point2_right[0])/2, (point1_right[1] + point2_right[1])/2]
    
    # finds the center of the front of the object
    point1_front = [(keypoints[1][0] + keypoints[7][0])/2, (keypoints[1][1] + keypoints[7][1])/2]
    point2_front = [(keypoints[3][0] + keypoints[5][0])/2, (keypoints[3][1] + keypoints[5][1])/2]
    front_coords = [(point1_front[0] + point2_front[0])/2, (point1_front[1] + point2_front[1])/2]
    
    # draws lines
    img_test2 = cv2.line(img, center, top_coords, colors[0], thickness)
    img_test2 = cv2.line(img, center, right_coords, colors[1], thickness)
    img_test2 = cv2.line(img, center, front_coords, colors[2], thickness)

    return img

def draw_keypointnums(img, keypoints):
    p = 0
    for point in keypoints:
        img = cv2.putText(img, str(p), [int(point[0]), int(point[1])], font = cv2.FONT_HERSHEY_SIMPLEX, 
                          fontScale = 1, color = (0, 0, 0) thickness = 1)
        p+=1
    
    return img

def draw_points(xy, img, color_points=(255, 0, 0), size=1, thickness=-1, line_type=cv2.LINE_AA):

    xy = tuple(map(tuple, xy))
    for p in xy:
        cv2.circle(
            img,
            (int(p[0]), int(p[1])),
            size,
            color_points,
            thickness,
            lineType=line_type,
        )
    return img


def draw_lines(xy_1, xy_2, img, color_points=(255, 255, 255)):
    xy_1 = xy_1.astype(int)
    xy_1 = tuple(map(tuple, xy_1))
    xy_2 = xy_2.astype(int)
    xy_2 = tuple(map(tuple, xy_2))
    for idx, p1 in enumerate(xy_1):
        p2 = xy_2[idx]
        cv2.line(img, p1, p2, color_points)
    return img


def pseudocolor_dir(x_dir, y_dir, mask):
    dir_map = np.arctan2(x_dir, y_dir) * 180.0 / np.pi
    dir_map[dir_map < 0.0] += 360.0
    dir_map[dir_map >= 360.0] -= 360.0
    dir_map[mask == 0] = 0.0
    dir_map = dir_map / 360.0  # * 179.0
    # dir_map = dir_map.astype('uint8')
    ones = np.full(mask.shape, 1.0, dtype="float")

    len_map = np.stack([x_dir, y_dir], -1)
    len_map = np.linalg.norm(len_map, axis=-1)
    len_map = np.clip(len_map, 0, 1)
    len_map = np.stack((len_map, len_map, len_map), axis=2) * 255.0

    dir_map = np.stack((dir_map, ones, ones), axis=2)
    dir_map = plt.colors.hsv_to_rgb(dir_map) * 255.0

    dir_map = dir_map.astype("uint8")
    len_map = len_map.astype("uint8")

    dir_map[mask == 0] = 0.0
    return dir_map


def grayscale_dist(dist, mask, clip_max):
    dist = (dist / clip_max) * 255
    dist = np.stack((dist, dist, dist), axis=2)
    dist = 255 - dist.astype("uint8")
    dist[mask == 0] = 0
    return dist
