from re import I
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MySoftmaxCrossEntropyLoss(nn.Module):

    def __init__(self, nbclasses):
        super(MySoftmaxCrossEntropyLoss, self).__init__()
        self.nbclasses = nbclasses

    def forward(self, inputs, target):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, self.nbclasses)  # N,H*W,C => N*H*W,C
        target = target.view(-1) # N*H*W
        return nn.CrossEntropyLoss(reduction="mean")(inputs, target)

        # N*H*W,C
        # N*H*W

        # 2*3*3, 3
        # 2*3*3


        # bce
        # 2*3*3
        # 2*3*3

# class MyFocalLoss(nn.Module):
#     def __init__(self, nbclasses, alpha, gamma):
#         super(MyFocalLoss, self).__init__()
#         self.nbclasses = nbclasses
#         self.alpha = alpha
#         self.gamma = gamma
#     def one_hot(self, targets, nbclasses):
#         """Convert an iterable of indices to one-hot encoded labels."""
#         targets = targets.reshape(-1)
#         return torch.eye(nbclasses)[targets]
#     def forward(self, inputs, targets): # target long dtype
#         print("inputs.shape:{}, targets.shape:{}".format(inputs.shape, targets.shape)) # N 张图
#         # inputs.shape:torch.Size([8, 8, 384, 1024]), targets.shape:torch.Size([8, 384, 1024])  inputs: N,nbclasses,h,w   targets: N,h,w
#         loss = 0.0
#         if inputs.dim() > 2:
#             one_hot_targets = self.one_hot(targets, self.nbclasses) # one_hot_targets: N*h*w, nbclasses
#             print("one_hot_targets.shape:{}".format(one_hot_targets.shape))
#             # one_hot_targets.shape:torch.Size([3145728, 8])
           
#             inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.nbclasses).sigmoid().cuda()  # inputs n, c, h, w ==> n*h*w, c

#             # pt = torch.tensor([i_ if t_==1 else 1-i_ for (i, t) in zip(inputs, targets) for (i_, t_) in zip(i, t)], dtype=torch.double)
#             # pt = pt.reshape(-1, self.nbclasses) # n*h*w, c

#             pt = torch.where(one_hot_targets==1, inputs, 1-inputs).cuda()
#             loss = -self.alpha*(1-pt)**self.gamma*torch.log(pt)*one_hot_targets-(1-self.alpha)*(1-pt)**self.gamma*torch.log(pt)*(1-one_hot_targets)
#         return loss.sum()/pt.size(0) # scaler



    # def forward(self, inputs, targets): # target long dtype
    #     print("inputs.shape:{}, targets.shape:{}".format(inputs.shape, targets.shape)) # N 张图
    #     # inputs.shape:torch.Size([8, 8, 384, 1024]), targets.shape:torch.Size([8, 384, 1024])  inputs: N,nbclasses,h,w   targets: N,h,w
    #     loss = 0.0
    #     if inputs.dim() > 2:
    #         one_hot_targets = self.one_hot(targets, self.nbclasses) # one_hot_targets: N*h*w, nbclasses
    #         print("one_hot_targets.shape:{}".format(one_hot_targets.shape))
    #         # one_hot_targets.shape:torch.Size([3145728, 8])
    #         for input_, target in zip(inputs, one_hot_targets): # inputs/one_hot_targets: N,nbclasses,h,w   input_/targets: nbclasses,h,w   以下是对一张图片求loss
    #             # trans_input, trans_target = input_.transpose(1, 2, 0), target.transpose(1, 2, 0) # h,w,nbclasses
    #             print(input_.shape, target.shape) # torch.Size([8, 384, 1024]) torch.Size([8])
    #             trans_input, trans_target = input_.permute(1, 2, 0), target.permute(1, 2, 0) # h,w,nbclasses
    #             print(type(trans_input))
    #             for i in range(trans_input.size(0)):
    #                 for j in range(trans_input.size(1)):
    #                     pixel_input, pixel_target = trans_input[i][j], trans_target[i][j]
    #                     pt = torch.tensor([i if t==1 else 1-i for (i, t) in zip(pixel_input, pixel_target)], dtype=torch.int) 
    #                     # i/t: h,w
    #                     sub_loss = -self.alpha*(1-pt)**self.gamma*torch.log(torch.tensor(pt, dtype=torch.double))*pixel_target-(1-self.alpha)*(1-pt)**self.gamma*torch.log(torch.tensor(pt, dtype=torch.double))*(1-pixel_target)
    #                     # RuntimeError: The size of tensor a (8) must match the size of tensor b (1024) at non-singleton dimension 2
    #                     loss += sub_loss
    #     return loss/targets.size(0)*targets.size(1)*targets.size(2) # scaler



class MyFocalLoss2(nn.Module):
    def __init__(self, nbclasses, alpha=[381155699, 4830287, 861159, 440608, 37987, 4061447, 1235420, 593393], gamma=2):
        '''
        alpha: [num_class0, num_class1, ..., num_classn]
        '''
        super(MyFocalLoss2, self).__init__()
        self.nbclasses = nbclasses
        alpha = np.array(alpha, dtype=np.float64)
        alpha[:] = 1./(alpha[:]/alpha.sum())
        alpha = torch.from_numpy(alpha).cuda()
        self.alpha = alpha # nbclass
        self.gamma = gamma
    def one_hot(self, targets, nbclasses):
        """Convert an iterable of indices to one-hot encoded labels."""
        targets = targets.reshape(-1)
        return torch.eye(nbclasses)[targets]
    def forward(self, inputs, targets): # target long dtype
        # print("inputs.shape:{}, targets.shape:{}".format(inputs.shape, targets.shape)) # N 张图
        # inputs.shape:torch.Size([8, 8, 384, 1024]), targets.shape:torch.Size([8, 384, 1024])  inputs: N,nbclasses,h,w   targets: N,h,w
        loss = 0.0
        if inputs.dim() > 2:
            one_hot_targets = self.one_hot(targets, self.nbclasses).cuda() # one_hot_targets: N*h*w, nbclasses
            # print("one_hot_targets.shape:{}".format(one_hot_targets.shape))
            # one_hot_targets.shape:torch.Size([3145728, 8])
           
            inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.nbclasses).sigmoid().cuda()  # inputs n, c, h, w ==> n*h*w, c

            # pt = torch.tensor([i_ if t_==1 else 1-i_ for (i, t) in zip(inputs, targets) for (i_, t_) in zip(i, t)], dtype=torch.double)
            # pt = pt.reshape(-1, self.nbclasses) # n*h*w, c

            pt = torch.where(one_hot_targets==1, inputs, 1-inputs).cuda()

            loss = -self.alpha*(1-pt)**self.gamma*torch.log(pt)*one_hot_targets
        return loss.sum()/pt.size(0) # scaler

















































        

        

