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


CHECKPOINT_PATH = '/media/samsung2080pc/New Volume/SAMSUNG/gazefollowing/trial1'
# log setting
log_dir = os.path.join(CHECKPOINT_PATH,'log')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = log_dir + 'train.log'

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s',
                    filename=log_file,
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

class GazeDataset(Dataset):
    def __init__(self, root_dir, mat_file, training='train'):
        assert (training in set(['train', 'test']))
        self.root_dir = root_dir
        self.mat_file = mat_file
        self.training = training

        # anns = loadmat(self.mat_file)
        # self.bboxes = anns[self.training + '_bbox']
        # self.gazes = anns[self.training + '_gaze']
        # self.paths = anns[self.training + '_path']
        # self.eyes = anns[self.training + '_eyes']
        # self.meta = anns[self.training + '_meta']
        # self.image_num = self.paths.shape[0]

        import pickle

        # self.bboxes = []
        # self.gazes = []
        # self.paths = []
        # self.eyes = []

        with open(mat_file, 'rb') as f:
            self.data = pickle.load(f)
            self.image_num = len(self.data)

            # for i in range(self.image_num):
                # print(data[i]['filename'])
                # print(data[i]['ann']['bboxes'][-1,:])
                # print(data[i]['gaze_cx'])
                # print(data[i]['gaze_cy'])

        #         self.paths.append(data[i]['filename'])
        #         self.bboxes.append(data[i]['ann']['bboxes'][-1,:])
        #         self.gazes.append([data[i]['gaze_cx'],data[i]['gaze_cy']])
        #         self.eyes.append([data[i]['hx'],data[i]['hy']])
        # self.bboxes = np.array(self.bboxes)
        # self.gazes = np.array(self.gazes)
        # self.eyes = np.array(self.eyes)
        # self.bboxes = np.expand_dims(self.bboxes, axis=0)
        # self.eyes = np.expand_dims(self.eyes, axis=0)
        # self.gazes = np.expand_dims(self.gazes, axis=0)

        #

        logging.info('%s contains %d images' % (self.mat_file, self.image_num))

    def generate_data_field(self, eye_point):
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

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        # print(self.data)
        get_data = self.data[idx]
        image_path = get_data['filename']
        image_path = os.path.join(self.root_dir, image_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.height, self.width, _ = image.shape


        #### NORMALIZE VARIABLES
        x_0 = get_data['ann']['bboxes'][-1,0] / self.width # xmin_headbb
        y_0 = get_data['ann']['bboxes'][-1,1] / self.height # ymin_headbb
        x_1 = get_data['ann']['bboxes'][-1,2] / self.width # xmax_headbb
        y_1 = get_data['ann']['bboxes'][-1,3] / self.height # ymax_headbb

        x_c = get_data['hx']/ self.width   # use head coordinates for eyes
        y_c = get_data['hy']/ self.height
        eye = np.array([x_c, y_c])

        gaze_x = get_data['gaze_cx']/ self.width
        gaze_y = get_data['gaze_cy']/ self.height
        gaze = np.array([gaze_x, gaze_y])

        if random.random() > 0.5 and self.training == 'train':
            eye = [1.0 - eye[0], eye[1]]
            gaze = [1.0 - gaze[0], gaze[1]]
            image = cv2.flip(image, 1)

            # box[0] = w-box[0]
            # box[2] = w-box[2]
        # # crop face
        # x_c, y_c = eye
        # x_0 = x_c - 0.15
        # y_0 = y_c - 0.15
        # x_1 = x_c + 0.15
        # y_1 = y_c + 0.15
        # if x_0 < 0:
        #     x_0 = 0
        # if y_0 < 0:
        #     y_0 = 0
        # if x_1 > 1:
        #     x_1 = 1
        # if y_1 > 1:
        #     y_1 = 1
        h, w = image.shape[:2]
        # print('h,w:',h,w, eye, x_c, y_c)
        face_image = image[int(y_0 * h):int(y_1 * h), int(x_0 * w):int(x_1 * w), :]
        # face_image = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        # print('\n\n\n\n',face_image.shape, box)
        # process face_image for face net
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)
        face_image = data_transforms[self.training](face_image)
        # process image for saliency net
        #image = image_preprocess(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = data_transforms[self.training](image)

        # generate gaze field
        gaze_field = self.generate_data_field(eye_point=eye)
        # generate heatmap
        heatmap = get_paste_kernel((224 // 4, 224 // 4), gaze, kernel_map, (224 // 4, 224 // 4))
        '''
        direction = gaze - eye
        norm = (direction[0] ** 2.0 + direction[1] ** 2.0) ** 0.5
        if norm <= 0.0:
            norm = 1.0

        direction = direction / norm
        '''
        sample = {'image' : image,
                  'face_image': face_image,
                  'eye_position': torch.FloatTensor(eye),
                  'gaze_field': torch.from_numpy(gaze_field),
                  'gt_position': torch.FloatTensor(gaze),
                  'gt_heatmap': torch.FloatTensor(heatmap).unsqueeze(0)}

        return sample


cosine_similarity = nn.CosineSimilarity()
mse_distance = nn.MSELoss()
bce_loss = nn.BCELoss()


def F_loss(direction, predict_heatmap, eye_position, gt_position, gt_heatmap):
    # point loss
    heatmap_loss = bce_loss(predict_heatmap, gt_heatmap)

    # angle loss
    gt_direction = gt_position - eye_position
    middle_angle_loss = torch.mean(1 - cosine_similarity(direction, gt_direction))

    return heatmap_loss, middle_angle_loss


def test(net, test_data_loader):
    net.eval()
    total_loss = []
    total_error = []
    info_list = []
    heatmaps = []

    for data in test_data_loader:
        image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
            data['image'], data['face_image'], data['gaze_field'], data['eye_position'], data['gt_position'], data['gt_heatmap']
        image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
            map(lambda x: Variable(x.cuda(), volatile=True), [image, face_image, gaze_field, eye_position, gt_position, gt_heatmap])

        direction, predict_heatmap = net([image, face_image, gaze_field, eye_position])

        heatmap_loss, m_angle_loss = \
            F_loss(direction, predict_heatmap, eye_position, gt_position, gt_heatmap)

        loss = heatmap_loss + m_angle_loss


        # total_loss.append([heatmap_loss.data[0],
        #                   m_angle_loss.data[0], loss.data[0]])
        # logging.info('loss: %.5lf, %.5lf, %.5lf'%( \
        #       heatmap_loss.data[0], m_angle_loss.data[0], loss.data[0]))


        total_loss.append([heatmap_loss.item(),
                          m_angle_loss.item(), loss.item()])
        logging.info('loss: %.5lf, %.5lf, %.5lf'%( \
              heatmap_loss.item(), m_angle_loss.item(), loss.item()))

        middle_output = direction.cpu().data.numpy()
        final_output = predict_heatmap.cpu().data.numpy()
        target = gt_position.cpu().data.numpy()
        eye_position = eye_position.cpu().data.numpy()
        for m_direction, f_point, gt_point, eye_point in \
            zip(middle_output, final_output, target, eye_position):
            f_point = f_point.reshape([224 // 4, 224 // 4])
            heatmaps.append(f_point)

            h_index, w_index = np.unravel_index(f_point.argmax(), f_point.shape)
            f_point = np.array([w_index / 56., h_index / 56.])

            f_error = f_point - gt_point
            f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)

            # angle
            f_direction = f_point - eye_point
            gt_direction = gt_point - eye_point

            norm_m = (m_direction[0] **2 + m_direction[1] ** 2 ) ** 0.5
            norm_f = (f_direction[0] **2 + f_direction[1] ** 2 ) ** 0.5
            norm_gt = (gt_direction[0] **2 + gt_direction[1] ** 2 ) ** 0.5

            m_cos_sim = (m_direction[0]*gt_direction[0] + m_direction[1]*gt_direction[1]) / \
                        (norm_gt * norm_m + 1e-6)
            m_cos_sim = np.maximum(np.minimum(m_cos_sim, 1.0), -1.0)
            m_angle = np.arccos(m_cos_sim) * 180 / np.pi

            f_cos_sim = (f_direction[0]*gt_direction[0] + f_direction[1]*gt_direction[1]) / \
                        (norm_gt * norm_f + 1e-6)
            f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
            f_angle = np.arccos(f_cos_sim) * 180 / np.pi


            total_error.append([f_dist, m_angle, f_angle])
            info_list.append(list(f_point))
    info_list = np.array(info_list)
    #np.savez('multi_scale_concat_prediction.npz', info_list=info_list)

    #heatmaps = np.stack(heatmaps)
    #np.savez('multi_scale_concat_heatmaps.npz', heatmaps=heatmaps)

    logging.info('average loss : %s'%str(np.mean(np.array(total_loss), axis=0)))
    logging.info('average error: %s'%str(np.mean(np.array(total_error), axis=0)))

    net.train()
    return


def main():
    train_set = GazeDataset(root_dir='/home/samsung2080pc/Documents/ObjectOfInterestV22Dataset/',
                            mat_file='/home/samsung2080pc/Documents/ObjectOfInterestV22Dataset/train.pickle',
                            training='train')

    train_data_loader = DataLoader(train_set, batch_size=3 * 4,
                                   shuffle=True, num_workers=16)

    # train_data_loader = DataLoader(train_set, batch_size=1,
    #                                shuffle=True, num_workers=1)

    test_set = GazeDataset(root_dir='/home/samsung2080pc/Documents/ObjectOfInterestV22Dataset/',
                            mat_file='/home/samsung2080pc/Documents/ObjectOfInterestV22Dataset/test.pickle',
                            training='test')
    test_data_loader = DataLoader(test_set, batch_size=1 * 4,
                                  shuffle=False, num_workers=8)

    net = GazeNet()
    net = DataParallel(net)
    net.cuda()

    resume_training = False
    if resume_training :
        pretrained_dict = torch.load('../model/pretrained_model.pkl')
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        test(net, test_data_loader)
        exit()

    method = 'Adam'
    learning_rate = 0.0001

    optimizer_s1 = optim.Adam([{'params': net.module.face_net.parameters(),
                                'initial_lr': learning_rate},
                               {'params': net.module.face_process.parameters(),
                                'initial_lr': learning_rate},
                               {'params': net.module.eye_position_transform.parameters(),
                                'initial_lr': learning_rate},
                               {'params': net.module.fusion.parameters(),
                                'initial_lr': learning_rate}],
                               lr=learning_rate, weight_decay=0.0001)
    optimizer_s2 = optim.Adam([{'params': net.module.fpn_net.parameters(),
                                'initial_lr': learning_rate}],
                               lr=learning_rate, weight_decay=0.0001)

    optimizer_s3 = optim.Adam([{'params': net.parameters(), 'initial_lr': learning_rate}],
                           lr=learning_rate*0.1, weight_decay=0.0001)

    lr_scheduler_s1 = optim.lr_scheduler.StepLR(optimizer_s1, step_size=5, gamma=0.1, last_epoch=-1)
    lr_scheduler_s2 = optim.lr_scheduler.StepLR(optimizer_s2, step_size=5, gamma=0.1, last_epoch=-1)
    lr_scheduler_s3 = optim.lr_scheduler.StepLR(optimizer_s3, step_size=5, gamma=0.1, last_epoch=-1)


    max_epoch = 25

    epoch = 0
    while epoch < max_epoch:
        if epoch == 0:
            lr_scheduler = lr_scheduler_s1
            optimizer = optimizer_s1
        elif epoch == 7:
            lr_scheduler = lr_scheduler_s2
            optimizer = optimizer_s2
        elif epoch == 15:
            lr_scheduler = lr_scheduler_s3
            optimizer = optimizer_s3

        lr_scheduler.step()

        running_loss = []
        for i, data in tqdm(enumerate(train_data_loader)):
            image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
                data['image'], data['face_image'], data['gaze_field'], data['eye_position'], data['gt_position'], data['gt_heatmap']
            image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
                map(lambda x: Variable(x.cuda()), [image, face_image, gaze_field, eye_position, gt_position, gt_heatmap])
            #for var in [image, face_image, gaze_field, eye_position, gt_position]:
            #    print var.shape

            optimizer.zero_grad()

            direction, predict_heatmap = net([image, face_image, gaze_field, eye_position])

            heatmap_loss, m_angle_loss = \
                F_loss(direction, predict_heatmap, eye_position, gt_position, gt_heatmap)

            if epoch == 0:
                loss = m_angle_loss
            elif epoch >= 7 and epoch <= 14:
                loss = heatmap_loss
            else:
                loss = m_angle_loss + heatmap_loss

            loss.backward()
            optimizer.step()

            # print('\n\n\n\nmeow',heatmap_loss, m_angle_loss)
            # print(heatmap_loss.data)
            # running_loss.append([heatmap_loss.data[0],
            #                      m_angle_loss.data[0], loss.data[0]])

            running_loss.append([heatmap_loss.item(), m_angle_loss.item(), loss.item()])

            if i % 10 == 9:
                logging.info('%s %s %s'%(str(np.mean(running_loss, axis=0)), method, str(lr_scheduler.get_lr())))
                running_loss = []

        epoch += 1

        # save_path = '../model/two_stage_fpn_concat_multi_scale_'+method
        save_path = CHECKPOINT_PATH + '_' + method
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(net.state_dict(), save_path+'/model_epoch{}.pkl'.format(epoch))

        test(net, test_data_loader)


if __name__ == '__main__':
    main()
