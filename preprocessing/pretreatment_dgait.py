# -*- coding: utf-8 -*-
# Adapted for D-Gait dataset

import os
import cv2
import numpy as np
from warnings import warn
from time import sleep
import argparse

from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"

SILHOUETTE_ANGLES_SUBDIR = [
    "/A1-back/", "/A1-front/", "/A2-back/", "/A2-front/",
    "/B1-back/", "/B1-front/", "/B2-back/", "/B2-front/",
    "/C1-back/", "/C1-front/", "/C2-back/", "/C2-front/",
    "/D1-back/", "/D1-front/", "/D2-back/", "/D2-front/",
    "/E1-back/", "/E1-front/", "/E2-back/", "/E2-front/",
    "/F1-back/", "/F1-front/", "/F2-back/", "/F2-front/",
    "/G1-back/", "/G1-front/", "/G2-back/", "/G2-front/",
    "/H1-back/", "/H1-front/", "/H2-back/", "/H2-front/"
]

CLOTHING_TYPES = ["bg-cl", "cl", "nm"]


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='D-Gait Pretreatment')
parser.add_argument('--input_path', default='/home/docker/DATA/D-Gait/D-Gait-Silhouette', type=str,
                    help='Root path of raw D-Gait dataset.')
parser.add_argument('--output_path', default='/home/docker/DATA/D-Gait/D-Gait-Silhouette-preprocess', type=str,
                    help='Root path for output.')
parser.add_argument('--log_file', default='./pretreatment_dgait.log', type=str,
                    help='Log file path. Default: ./pretreatment_dgait.log')
parser.add_argument('--log', default=False, type=boolean_string,
                    help='If set as True, all logs will be saved. '
                         'Otherwise, only warnings and errors will be saved.'
                         'Default: False')
parser.add_argument('--worker_num', default=4, type=int,
                    help='How many subprocesses to use for data pretreatment. '
                         'Default: 4')
opt = parser.parse_args()

INPUT_PATH = opt.input_path
OUTPUT_PATH = opt.output_path
IF_LOG = opt.log
LOG_PATH = opt.log_file
WORKERS = opt.worker_num

T_H = 64
T_W = 64


def log2str(pid, comment, logs):
    str_log = ''
    if type(logs) is str:
        logs = [logs]
    for log in logs:
        str_log += "# JOB %d : --%s-- %s\n" % (
            pid, comment, log)
    return str_log


def log_print(pid, comment, logs):
    str_log = log2str(pid, comment, logs)
    if comment in [WARNING, FAIL]:
        with open(LOG_PATH, 'a') as log_f:
            log_f.write(str_log)
    if comment in [START, FINISH]:
        if pid % 500 != 0:
            return
    print(str_log, end='')


def cut_img(img, seq_info, frame_name, pid):
    # A silhouette contains too little white pixels
    # might be not valid for identification.
    if img.sum() <= 10000:
        message = 'seq:%s, frame:%s, no data, %d.' % (
            '-'.join(seq_info), frame_name, img.sum())
        warn(message)
        log_print(pid, WARNING, message)
        return None
    # Get the top and bottom point
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
    # Get the median of x axis and regard it as the x center of the person.
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    if x_center < 0:
        message = 'seq:%s, frame:%s, no center.' % (
            '-'.join(seq_info), frame_name)
        warn(message)
        log_print(pid, WARNING, message)
        return None
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right]
    return img.astype('uint8')


def cut_pickle(seq_info, pid):
    """Process a single sequence (patient-clothing-angle)"""
    seq_name = '-'.join(seq_info)
    log_print(pid, START, seq_name)
    seq_path = os.path.join(INPUT_PATH, *seq_info)
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    frame_list = os.listdir(seq_path)
    frame_list.sort()
    count_frame = 0
    
    for _frame_name in frame_list:
        frame_path = os.path.join(seq_path, _frame_name)
        
        # Skip if not an image file
        if not _frame_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
            
        try:
            img = cv2.imread(frame_path)
            if img is None:
                continue
            img = img[:, :, 0]  # Use only one channel for silhouette
            img = cut_img(img, seq_info, _frame_name, pid)
            
            if img is not None:
                # Save the cut img
                save_path = os.path.join(out_dir, _frame_name)
                cv2.imwrite(save_path, img)
                count_frame += 1
        except Exception as e:
            message = 'seq:%s, frame:%s, error: %s' % (
                '-'.join(seq_info), _frame_name, str(e))
            warn(message)
            log_print(pid, WARNING, message)
    
    # Warn if the sequence contains less than 5 frames
    if count_frame < 5:
        message = 'seq:%s, less than 5 valid data.' % (
            '-'.join(seq_info))
        warn(message)
        log_print(pid, WARNING, message)

    log_print(pid, FINISH,
              'Contain %d valid frames. Saved to %s.'
              % (count_frame, out_dir))


if __name__ == '__main__':
    pool = Pool(WORKERS)
    results = list()
    pid = 0

    print('D-Gait Pretreatment Start.\n'
          'Input path: %s\n'
          'Output path: %s\n'
          'Log file: %s\n'
          'Worker num: %d' % (
              INPUT_PATH, OUTPUT_PATH, LOG_PATH, WORKERS))

    # Create output directory structure
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    patient_list = os.listdir(INPUT_PATH)
    patient_list.sort()
    
    # Walk through: patients -> clothing types -> angles
    for _patient in patient_list:
        patient_path = os.path.join(INPUT_PATH, _patient)
        
        if not os.path.isdir(patient_path):
            continue
        
        for _clothing in CLOTHING_TYPES:
            clothing_path = os.path.join(patient_path, _clothing)
            
            if not os.path.isdir(clothing_path):
                continue
            
            for _angle in SILHOUETTE_ANGLES_SUBDIR:
                angle_name = _angle.strip('/')
                angle_path = os.path.join(clothing_path, angle_name)
                
                if not os.path.isdir(angle_path):
                    continue
                
                seq_info = [_patient, _clothing, angle_name]
                out_dir = os.path.join(OUTPUT_PATH, *seq_info)
                os.makedirs(out_dir, exist_ok=True)
                
                results.append(
                    pool.apply_async(
                        cut_pickle,
                        args=(seq_info, pid)))
                sleep(0.02)
                pid += 1

    pool.close()
    unfinish = 1
    while unfinish > 0:
        unfinish = 0
        for i, res in enumerate(results):
            try:
                res.get(timeout=0.1)
            except Exception as e:
                if type(e) == MP_TimeoutError:
                    unfinish += 1
                    continue
                else:
                    print('\n\n\nERROR OCCUR: PID ##%d##, ERRORTYPE: %s\n\n\n' %
                          (i, type(e)))
                    raise e
    pool.join()

    print('\nPretreatment Complete!')
