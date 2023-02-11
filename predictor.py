from typing import Tuple

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.body import Body
from src.model import bodypose_model
from src.util import transfer

import math
import numpy as np


def make_gaussian_kernel(radius, sigma=2., order=0):
    assert order == 0, 'Non-zero order is not implemented'
    # The gaussian kernel is the product of the gaussian function of each dimension.
    # kernel_size should be an odd number.

    sigma2 = sigma * sigma
    x = torch.arange(-radius, radius + 1)
    phi_x = torch.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    return phi_x


def pad_symmetric(tensor: torch.Tensor, pad_left: int, pad_right: int):

    _x_indices = [i for i in range(tensor.shape[-1])]  # [0, 1, 2, 3, ...]
    left_indices = [i for i in range(pad_left - 1, -1, -1)]  # e.g. [3, 2, 1, 0]
    right_indices = [-(i + 1) for i in range(pad_right)]  # e.g. [-1, -2, -3]
    x_indices = torch.tensor(left_indices + _x_indices + right_indices, device=tensor.device)

    return tensor[:, :, x_indices]


def peak(tensor, edge: float = float('-inf')):  # HxW
    wp = F.pad(tensor, (1, 1), value=edge)
    to_right = torch.ge(tensor, wp[..., 2:])
    to_left = torch.ge(tensor, wp[..., :-2])

    hp = F.pad(tensor, (0, 0, 1, 1), value=edge)
    to_up = torch.ge(tensor, hp[..., :-2, :])
    to_down = torch.ge(tensor, hp[..., 2:, :])

    return torch.logical_and(torch.logical_and(to_left, to_right), torch.logical_and(to_up, to_down))


class OpenPosePredictor(nn.Module):
    def __init__(self, param_path: str, in_size: Tuple[int, int] = (512, 384)):
        super(OpenPosePredictor, self).__init__()

        # Prepare CNN
        model = bodypose_model()
        model_dict = transfer(model, torch.load(param_path))
        model.load_state_dict(model_dict)
        self.model = torch.jit.trace(model, torch.rand(1, 3, in_size[0], in_size[1]))

        radius = int(4 * 3 + 0.5)  # 12
        g_kernel = make_gaussian_kernel(radius=radius, sigma=3.).reshape(1, 1, -1)
        self.register_buffer('g_kernel', g_kernel)

        # inference params
        self.boxsize = 368
        self.stride = 8
        self.thre1 = 0.1
        self.thre2 = 0.05

        # find connection in the specified sequence, center 29 is in the position 15
        self.limbSeq = torch.asarray([[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                   [1, 16], [16, 18], [3, 17], [6, 18]]) - 1
        # the middle joints heatmap correspondence
        self.mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
                  [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
                  [55, 56], [37, 38], [45, 46]]

    def gaussian_filter(self, tensor: torch.Tensor, width: int, height: int):
        b, c, h, w = tensor.shape
        tensor = tensor.reshape(-1, 1, width)  # BCHx1xW
        tensor = F.conv1d(pad_symmetric(tensor, 12, 12), weight=self.g_kernel, bias=None)  # BCHx1xW
        tensor = tensor.reshape(-1, c, height, width).permute(0, 1, 3, 2).reshape(-1, 1, height)  # BCW,1,H
        tensor = F.conv1d(pad_symmetric(tensor, 12, 12), weight=self.g_kernel, bias=None)  # BCW,1,H
        tensor = tensor.reshape(-1, c, width, height).permute(0, 1, 3, 2)  # BxCxHxW
        return tensor
        
    def forward(self, img):
        assert img.ndim == 4
        assert img.shape[0] == 1, 'Inference only supports single-image input'
        assert img.dtype == torch.float  # we expect it to be normalized with [ /256 - 0.5]
        # resize input image to match trained size
        batch, c, height, width = img.shape
        scale = float(0.5 * self.boxsize / height)

        # resize image
        image_to_test = F.interpolate(img, scale_factor=scale, mode='bicubic')

        # pad zeros to botton-right
        padding_bottom = 0 if (height % self.stride == 0) else self.stride - (height % self.stride)  # down
        padding_right = 0 if (width % self.stride == 0) else self.stride - (width % self.stride)  # right
        image_to_test_padded = F.pad(image_to_test, (0, padding_right, 0, padding_bottom))

        # run convnet
        Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(image_to_test_padded)

        heatmap = F.interpolate(Mconv7_stage6_L2, scale_factor=float(self.stride), mode='bicubic')
        heatmap = heatmap[:, :, :image_to_test_padded.shape[2] - padding_bottom, :image_to_test_padded.shape[3] - padding_right]
        heatmap = F.interpolate(heatmap, size=(height, width), mode='bicubic')  # Bx19xHxW
        heatmap = heatmap[:, :-1, ...]  # drop last channel

        paf = F.interpolate(Mconv7_stage6_L1, scale_factor=float(self.stride), mode='bicubic')
        paf = paf[:, :, :image_to_test_padded.shape[2] - padding_bottom, :image_to_test_padded.shape[3] - padding_right]
        paf = F.interpolate(paf, size=(height, width), mode='bicubic')

        # Run gaussian filter on each heatmap
        heatmap_enh = self.gaussian_filter(heatmap, width, height)  # Bx19xHxW

        # find peaks - single image only
        peaks_binary = torch.logical_and(peak(heatmap_enh, 0.), torch.gt(heatmap_enh, self.thre1))
        peaks = peaks_binary.nonzero().narrow(dim=1, start=1, length=3)  # Nx3  C|Y|X
        peak_ids = torch.arange(peaks.shape[0], device=peaks.device)  # N
        peak_values = heatmap[peaks_binary > 0]  # N
        peak_with_data = torch.cat([peaks, peak_values.unsqueeze(1), peak_ids.unsqueeze(1)], dim=1)  # (X,Y,conf,id)

        #all_peaks = [peak_with_data[peaks[:, 0] == c][:, (2, 1, 3, 4)].tolist() for c in range(18)]
        all_peaks = [peak_with_data[peaks[:, 0] == c].index_select(dim=1, index=torch.LongTensor([2,1,3,4])) for c in range(18)]

        #return self.extract_poses(all_peaks, paf[0].permute(1, 2, 0).numpy(), height)
        return all_peaks, paf[0].permute(1, 2, 0), height

    def extract_poses(self, all_peaks, paf, height):
        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(self.mapIdx)):
            score_mid = paf[:, :, [x - 19 for x in self.mapIdx[k]]]
            candA = all_peaks[self.limbSeq[k][0]]
            candB = all_peaks[self.limbSeq[k][1]]
            nA = len(candA)
            nB = len(candB)

            if nA != 0 and nB != 0:
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        norm = max(0.001, norm)
                        vec = np.divide(vec, norm)

                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                            np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * height / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > self.thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if len(connection) >= min(nA, nB):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(self.mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = self.limbSeq[k].numpy()

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
        # candidate: x, y, score, id
        return candidate, subset


def torchscript_export(param_path, output_path):
    model = bodypose_model()
    model_dict = transfer(model, torch.load(param_path))
    model.load_state_dict(model_dict)
    model = torch.jit.trace(model, torch.rand(1, 3, in_size[0], in_size[1]))
    model.save(output_path)


if __name__ == '__main__':
    param_path = '/home/istvanfe/Work/vton-data-generator/models/openpose/body_pose_model.pth'
    test_image = '/home/istvanfe/Datasets/DressCode/upper_body/images_lowres/000000_0.jpg'
    in_size = 512, 384

    # DEBUG
    DEBUG = True
    DEBUG = False
    if DEBUG:
        img_vec = cv2.imread(test_image)
        model = Body(param_path)
        with torch.no_grad():
            print(model(img_vec)[1])
        quit()

    #torchscript_export(param_path, 'model/openpose18.pt'); quit()

    img_vec = cv2.imread(test_image)  # HxWxC (B,G,R)
    img_vec = torch.from_numpy(img_vec).to(torch.float32) / 256 - 0.5
    img_vec = img_vec.permute(2,0,1)

    predictor = OpenPosePredictor(param_path)

    #predictor = torch.jit.script(predictor, img_vec.unsqueeze(0))
    #predictor.save('model/openpose18.pt')

    with torch.no_grad():
        print(predictor(img_vec.unsqueeze(0))[1])

