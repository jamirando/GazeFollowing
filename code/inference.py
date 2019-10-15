import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from gazenet import GazeNet

import time
import os
import numpy as np
import json
import cv2
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import operator
import itertools
from scipy.io import  loadmat
import logging

from scipy import signal

from utils import data_transforms
from utils import get_paste_kernel, kernel_map

import pickle
import time

def generate_data_field(eye_point):
    """eye_point is (x, y) and between 0 and 1"""
    height, width = 224, 224
    x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
    y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
    grid = np.stack((x_grid, y_grid)).astype(np.float32)

    x, y = eye_point
    x, y = x * width, y * height

    grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
    norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
    # avoid zero norm
    norm = np.maximum(norm, 0.1)
    grid /= norm
    return grid

def preprocess_image(image_path, eye):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # crop face
    x_c, y_c = eye
    x_0 = x_c - 0.15
    y_0 = y_c - 0.15
    x_1 = x_c + 0.15
    y_1 = y_c + 0.15
    if x_0 < 0:
        x_0 = 0
    if y_0 < 0:
        y_0 = 0
    if x_1 > 1:
        x_1 = 1
    if y_1 > 1:
        y_1 = 1

    h, w = image.shape[:2]
    face_image = image[int(y_0 * h):int(y_1 * h), int(x_0 * w):int(x_1 * w), :]
    # process face_image for face net
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = Image.fromarray(face_image)
    face_image = data_transforms['test'](face_image)
    # process image for saliency net
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = data_transforms['test'](image)

    # generate gaze field
    gaze_field = generate_data_field(eye_point=eye)
    sample = {'image' : image,
              'face_image': face_image,
              'eye_position': torch.FloatTensor(eye),
              'gaze_field': torch.from_numpy(gaze_field)}

    return sample


def test(net, test_image_path, eye):
    net.eval()
    heatmaps = []

    data = preprocess_image(test_image_path, eye)

    image, face_image, gaze_field, eye_position = data['image'], data['face_image'], data['gaze_field'], data['eye_position']
    image, face_image, gaze_field, eye_position = map(lambda x: Variable(x.unsqueeze(0).cuda(), volatile=True), [image, face_image, gaze_field, eye_position])

    _, predict_heatmap = net([image, face_image, gaze_field, eye_position])

    final_output = predict_heatmap.cpu().data.numpy()

    heatmap = final_output.reshape([224 // 4, 224 // 4])

    h_index, w_index = np.unravel_index(heatmap.argmax(), heatmap.shape)
    f_point = np.array([w_index / 56., h_index / 56.])


    return heatmap, f_point[0], f_point[1]

def draw_result(image_path, eye, heatmap, gaze_point, results_path='tmp.png'):
    x1, y1 = eye
    x2, y2 = gaze_point
    im = cv2.imread(image_path)
    image_height, image_width = im.shape[:2]
    x1, y1 = image_width * x1, y1 * image_height
    x2, y2 = image_width * x2, y2 * image_height
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.circle(im, (x1, y1), 5, [255, 255, 255], -1)
    cv2.circle(im, (x2, y2), 5, [255, 255, 255], -1)
    cv2.line(im, (x1, y1), (x2, y2), [255, 0, 0], 3)

    # heatmap visualization
    heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
    heatmap = np.stack([heatmap, heatmap, heatmap], axis=2)
    heatmap = cv2.resize(heatmap, (image_width, image_height))

    heatmap = (0.8 * heatmap.astype(np.float32) + 0.2 * im.astype(np.float32)).astype(np.uint8)
    img = np.concatenate((im, heatmap), axis=1)
    cv2.imwrite(results_path, img)

    return img

def main():

    # DEFINE PARAMETERS

    CHECKPOINT_PATH = '/media/samsung2080pc/New Volume/SAMSUNG/gazefollowing/trial1_Adam'
    EPOCH = 25
    LOAD_PATH = os.path.join(CHECKPOINT_PATH, 'model_epoch'+str(EPOCH)+'.pkl')
    DATASET_PATH = '/home/samsung2080pc/Documents/ObjectOfInterestV22Dataset'
    TEST_PATH = '/home/samsung2080pc/Documents/ObjectOfInterestV22Dataset/test.pickle'
    NUM_TEST = 0


    net = GazeNet()
    net = DataParallel(net)
    net.cuda()

    # pretrained_dict = torch.load('../model/pretrained_model.pkl')

    pretrained_dict = torch.load(LOAD_PATH)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    f = open(TEST_PATH, 'rb')
    test_data = pickle.load(f)
    test_img_path = test_data[0]['filename']
    test_img_path = os.path.join(DATASET_PATH, test_img_path)
    h,w = cv2.imread(test_img_path).shape[:2]
    if NUM_TEST == 0:
        NUM_TEST = len(test_data)

    save_path = os.path.join(CHECKPOINT_PATH, 'epoch_'+str(EPOCH))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    start_time = time.time()
    results_pkl = []
    for i in tqdm(range(NUM_TEST)):

        save_path = os.path.join(CHECKPOINT_PATH, 'epoch_'+str(EPOCH), 'out'+str(i).zfill(4)+'.png')

        # test_image_path = sys.argv[1]
        # x = float(sys.argv[2])
        # y = float(sys.argv[3])
        test_image_path = test_data[i]['filename']
        test_image_path = os.path.join(DATASET_PATH, test_image_path)
        x = test_data[i]['hx']/w
        y = test_data[i]['hy']/h
        # print(test_image_path,x,y)
        heatmap, p_x, p_y = test(net, test_image_path, (x, y))
        output = {  # PREDICTIONS
                    'predictions':{
                        'heatmap': heatmap,
                        'p_x': p_x,
                        'p_y': p_y
                        },
                    # INPUTS
                    'inputs':{
                        'image_path': test_data[i]['filename'],
                        'eye_x': x,
                        'eye_y': y,
                        },
                    # GROUND TRUTH
                    'gt':{
                        'gaze_cx': test_data[i]['gaze_cx'],
                        'gaze_cy': test_data[i]['gaze_cy']
                        }
                    }
        results_pkl.append(output)
        draw_result(test_image_path, (x, y), heatmap, (p_x, p_y), save_path)

    end_time = time.time()
    process_time = end_time - start_time

    outfilename = os.path.join(CHECKPOINT_PATH, 'epoch_'+str(EPOCH), 'allresults.pkl')


    with open(outfilename, 'wb') as outfile:
        pickle.dump(results_pkl, outfile)
    print(results_pkl)
    print('Processed %i images in %f seconds.' %(NUM_TEST, process_time))
    # print(p_x, p_y)


if __name__ == '__main__':
    main()
