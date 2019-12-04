"""
@project: mobile_sr_evaluation
@author: sfzhou
@file: evaluation.py
@ide: PyCharm
@time: 2019/5/14 15:32

"""
import argparse
import torch
from utils.dataloader import TestDataset
from torch.utils.data import DataLoader
from utils.util import load_state_dict, sr_forward_psnr, sr_forward_time
#from torchprofile import profile
from utils.profile import profile
import time

from model.SRCNN import SRCNN
from model.FSRCNN import FSRCNN
from model.model import model
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_path', type=str, default='pre-train-model/srcnn.pth', help='path of checkpoint of baseline')
    parser.add_argument('--model_path', type=str, default='pre-train-model/model.pth', help='path of checkpoint of test model')
    parser.add_argument('--LR_path', type=str, default='../../teseting_data', help='path of the LR images')
    parser.add_argument('--HR_path', type=str, default='../../teseting_data', help='path of the HR images')
    parser.add_argument('--upscale', type=int, default=2, help='scale factor for up-sample LR image ')
    parser.add_argument('--cuda', type=bool, default=False, help='whether use cuda or not')
    parser.add_argument('--alpha', type=float, default=2, help='the weight of alpha')
    parser.add_argument('--beta', type=float, default=200, help='the weight of beta')
    parser.add_argument('--gamma', type=float, default=2, help='the weight of gamma')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of inference')
    parser.add_argument('--cycle_num', type=int, default=3, help='the number of repeat running model')



    return parser.parse_args()


def evaluation(opt):

    device = torch.device('cuda' if opt.cuda else 'cpu')

    alpha = opt.alpha
    beta = opt.beta
    gamma = opt.gamma
    cycle_num = opt.cycle_num

    #load test model
    #test_model = FSRCNN(opt.upscale)
    test_model = model(opt.upscale)
    state_dict = load_state_dict(opt.model_path)
    test_model.load_state_dict(state_dict)
    test_model = test_model.to(device)
    test_model.eval()

    #load baseline
    baseline_model = SRCNN(opt.upscale)
    baseline_dict = load_state_dict(opt.baseline_path)
    baseline_model.load_state_dict(baseline_dict)
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    #load dataset
    dataset = TestDataset(opt.HR_path, opt.LR_path)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    crop_boarder = opt.upscale
    print("dd")
    baseline_psnr, baseline_ssim = sr_forward_psnr(dataloader, baseline_model, device, crop_boarder)
    print("baseline_psnr:"+ str(baseline_psnr))
    print("baseline_ssim:"+ str(baseline_ssim))
    test_psnr, test_ssim = sr_forward_psnr(dataloader, test_model, device, crop_boarder)
    print("test_psnr:"+ str(test_psnr))
    print("test_ssim:"+ str(test_ssim))
    baseline_times = 0.0
    test_times = 0.0
    # test_ssim = 0.0
    # test_psnr = 0.0
    # baseline_ssim = 0.0
    # baseline_psnr = 0.0
    for index in range(cycle_num):
        baseline_time = sr_forward_time(dataloader, baseline_model, device)
        baseline_times += baseline_time
        print("baseline time"+str(baseline_times))
    for index in range(cycle_num):
        test_time = sr_forward_time(dataloader, test_model, device)
        test_times += test_time
        print("test time"+str(test_times))


    score = alpha * (test_psnr-baseline_psnr) + beta * (test_ssim-baseline_ssim) + gamma * min(baseline_times/test_times, 4)

    print('psnr: {:.4f}'.format(alpha * (test_psnr-baseline_psnr)))
    print('ssim: {:.4f}'.format(beta * (test_ssim-baseline_ssim)))
    print('time: {:.4f}'.format(gamma * min((baseline_times/test_times), 4)))
    print('score: {:.4f}'.format(score))

    print('avarage score: {:.4f}'.format(score))

    #calc FLOPs
    width = 360
    height = 240
    flops, params = profile(test_model, input_size=(1, 3, height, width))
    print('test_model{} x {}, flops: {:.4f} GFLOPs, params: {}'.format(height, width, flops/(1e9), params))
















if __name__ == '__main__':

    opt = parser_args()
    evaluation(opt)