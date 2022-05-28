from distutils.command.config import config
from tqdm import tqdm
import torch
import os
import shutil
from utils.metric import compute_iou
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ImageAug, DeformAug
from utils.image_process import ScaleAug, CutOut, ToTensor
from utils.loss import MySoftmaxCrossEntropyLoss
from model.deeplabv3plus import DeeplabV3Plus
from model.unet import ResNetUNet
from config import Config
from lr.cyclic_lr import CosineAnnealingLR_with_Restart


# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

device_list = [0]
train_net = 'deeplabv3p' # 'unet'
# nets['deeplabv3p']:DeeplabVePlus
nets = {'deeplabv3p': DeeplabV3Plus, 'unet': ResNetUNet}

def loss_func(predict, target, nbclasses, epoch):
    ''' can modify or add losses '''
    ce_loss = MySoftmaxCrossEntropyLoss(nbclasses=nbclasses)(predict, target)
    return ce_loss


def train_epoch(net, cycle_index, dataLoader, optimizer, trainF, config, lr):
    # 切换成训练模式
    net.train()

    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        # 梯度清零
        optimizer.zero_grad()
        # cbrp-cbrp-
        # 前向计算
        out = net(image)
        # 算loss
        mask_loss = loss_func(out, mask, config.NUM_CLASSES, cycle_index)
        total_mask_loss += mask_loss.item()
        # D(loss)/D(w)
        # 反向传播求梯度
        mask_loss.backward()
        # w = w - lr * delta_w
        # 更新权重
        optimizer.step()
        dataprocess.set_description_str("epoch:{}".format(cycle_index))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss.item()))


def test(net, cycle_index, dataLoader, testF, config, lr, max_iou, global_max_iou):
    # 切换eval模式
    net.eval()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    result = {"TP": {i:0 for i in range(8)}, "TA":{i:0 for i in range(8)}}
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        # 前向计算
        out = net(image)
        # 算loss
        mask_loss = loss_func(out, mask, config.NUM_CLASSES, cycle_index)
        total_mask_loss += mask_loss.detach().item()
        pred = torch.argmax(F.softmax(out, dim=1), dim=1)
        result = compute_iou(pred, mask, result)
        dataprocess.set_description_str("cycle_index:{}".format(cycle_index))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss))

    miou = 0
    for i in range(8):
        iou_i = result["TP"][i]/result["TA"][i]
        result_string = "{}: {:.4f} \n".format(i, iou_i)
        print(result_string)
        testF.write(result_string)
        miou += iou_i
    miou /= 8
    # 更新本轮循环和全局的最优权重
    if miou > max_iou:
        max_iou = miou
        save_pth = os.path.join(os.getcwd(), config.SAVE_PATH, "cycle_index_{}_max_iou.pth.tar".format(cycle_index))
        torch.save(net.state_dict(), save_pth)
        testF.write('save cycle ' + str(cycle_index) + ' max iou model: ' + str(max_iou) + '\n')
        testF.write("lr:{}, mask loss is {:.4f} \n".format(lr, total_mask_loss / len(dataLoader)))
        testF.flush()
    if miou > global_max_iou:
        global_max_iou = miou
        save_pth = os.path.join(os.getcwd(), config.SAVE_PATH, "global_max_iou.pth.tar")
        torch.save(net.state_dict(), save_pth)
        testF.write('save global max iou model, global_max_iou:' + str(global_max_iou)+ '\n')
        testF.write("lr:{}, mask loss is {:.4f} \n".format(lr, total_mask_loss / len(dataLoader)))
        testF.flush()
    return max_iou, global_max_iou


def main():
    lane_config = Config()
    if os.path.exists(lane_config.SAVE_PATH):
        shutil.rmtree(lane_config.SAVE_PATH)
    os.makedirs(lane_config.SAVE_PATH, exist_ok=True)
    trainF = open(os.path.join(lane_config.SAVE_PATH, "train_log.csv"), 'w')
    testF = open(os.path.join(lane_config.SAVE_PATH, "val_log.csv"), 'w')
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataset = LaneDataset("disk2/lane_segment/AdamW_cosine_annealing_lane_segmentation/data_list/train.csv", transform=transforms.Compose([ImageAug(), DeformAug(),
                                                                              ScaleAug(), CutOut(32, 0.5), ToTensor()]))

    train_data_batch = DataLoader(train_dataset, batch_size=8*len(device_list), shuffle=True, drop_last=True, **kwargs)
    val_dataset = LaneDataset("disk2/lane_segment/AdamW_cosine_annealing_lane_segmentation/data_list/val.csv", transform=transforms.Compose([ToTensor()]))

    val_data_batch = DataLoader(val_dataset, batch_size=4*len(device_list), shuffle=False, drop_last=False, **kwargs)
    net = nets[train_net](lane_config)
    if torch.cuda.is_available():
        net = net.cuda(device=device_list[0])
        net = torch.nn.DataParallel(net, device_ids=device_list)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lane_config.BASE_LR,
    #                             momentum=0.9, weight_decay=lane_config.WEIGHT_DECAY)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lane_config.BASE_LR, weight_decay=lane_config.WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lane_config.BASE_LR, betas=lane_config.BETA, eps = lane_config.EPS, weight_decay=lane_config.WEIGHT_DECAY, amsgrad=True)
    
    adamW_CosineAnneal = CosineAnnealingLR_with_Restart(optimizer,
                                          T_max=lane_config.CYCLE_INTER,
                                          T_mult=2,
                                          model=net,
                                          out_dir='cosine_annealing_snapshot',
                                          take_snapshot=False,
                                          eta_min=1e-5,
                                          eta_max=[1e-1])
    global_max_iou = 0
    for cycle_index in range(lane_config.CYCLE_NUM): # 循环次数:CYCLE_NUM
        print('cycle index: ' + str(cycle_index))
        max_iou = 0

        for epoch in range(0, lane_config.CYCLE_INTER): # 周期:CYCLE_INTER个epoch
            # 更新学习率
            adamW_CosineAnneal.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr : {:.4f}'.format(lr))

            # 梯度清零
            optimizer.zero_grad()

            # 训练过程
            train_epoch(net, epoch, train_data_batch, optimizer, trainF, lane_config, lr)

            # epoch过了周期的一半
            if epoch >= lane_config.CYCLE_INTER//2:
                max_iou, global_max_iou =  test(net, cycle_index, val_data_batch, testF, lane_config, lr, max_iou, global_max_iou)        
    trainF.close()
    testF.close()
       


if __name__ == "__main__":
    main()
