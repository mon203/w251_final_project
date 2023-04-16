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
    """
    inputs:
        xy: 8x2 np array, the 8 points of a rectangle
        img: single image, can be the frame of a video
    output:
        img: returns img with bounding box drawn
    """
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