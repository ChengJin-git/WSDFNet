# ------------------------------------------------------------------
# Copyright (c) 2021, Zi-Rong Jin, Tian-Jing Zhang, Cheng Jin, and 
# Liang-Jian Deng, All rights reserved.
#
# This work is licensed under GNU Affero General Public License
# v3.0 International To view a copy of this license, see the
# LICENSE file.
#
# This file is running on WorldView-3 dataset. For other dataset
# (i.e., QuickBird and GaoFen-2), please change the corresponding
# inputs.
# ------------------------------------------------------------------

import torch
from evaluate import compute_index
from scipy import io as sio
from model import WSDFNet


def load_set(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    lms = torch.from_numpy(
        data['lms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy((data['ms'] / 2047.0)
                          ).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy((data['pan'] / 2047.0))   # HxW = 256x256

    return lms, ms, pan


def load_gt_compared(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    test_gt = torch.from_numpy(data['gt'] / 2047.0)  # CxHxW = 8x256x256

    return test_gt


file_path = "test_data/WorldView-3_1.mat"
test_lms, test_ms, test_pan = load_set(file_path)
test_lms = test_lms.cuda().unsqueeze(dim=0).float()

# convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
test_ms = test_ms.cuda().unsqueeze(dim=0).float()
test_pan = test_pan.cuda().unsqueeze(dim=0).unsqueeze(
    dim=1).float()  # convert to tensor type: 1x1xHxW
test_gt = load_gt_compared(file_path)  # compared_result
test_gt = (test_gt * 2047).cuda().double()
model = WSDFNet().cuda()
model.load_state_dict(torch.load('./pretrained/WSDFNET_500.pth'))

model.eval()
with torch.no_grad():
    output3 = model(test_pan, test_lms)
    result_our = torch.squeeze(output3).permute(1, 2, 0)
    sr = torch.squeeze(output3).permute(
        1, 2, 0).cpu().detach().numpy()  # HxWxC
    result_our = result_our * 2047
    result_our = result_our.type(torch.DoubleTensor).cuda()

    sio.savemat('../results/WorldView-3_1_wsdfnet.mat', {'wsdfnet_output': sr})

    our_SAM, our_ERGAS = compute_index(test_gt, result_our, 4)
    # print loss for each epoch
    print('WSDFNet_output_SAM: {}'.format(our_SAM))
    print('WSDFNet_output_ERGAS: {}'.format(our_ERGAS))
