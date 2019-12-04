from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

def make_model(args, parent=False):
    inpu = torch.randn(1, 3, 360, 240).cpu()
    flops, params = profile(RTCV(args).cpu(), inputs=(inpu,))
    print(params)
    print(flops)
    return RTCV(args)
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride) #, padding)
        
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
class one_conv(nn.Module):
    def __init__(self,inchanels,growth_rate,kernel_size = 3, relu = True):
        super(one_conv,self).__init__()
        wn = lambda x:torch.nn.utils.weight_norm(x)
        self.conv = nn.Conv2d(inchanels,growth_rate,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        self.flag = relu
        self.conv1 = nn.Conv2d(growth_rate,inchanels,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        if relu:
            self.relu = nn.PReLU(growth_rate)#nn.ReLU()
        self.weight1 = common.Scale(1)
        self.weight2 = common.Scale(1)
    def forward(self,x):
        if self.flag == False:
            output = self.weight1(x) + self.weight2(self.conv1(self.conv(x)))
        else:
            output = self.weight1(x) + self.weight2(self.conv1(self.relu(self.conv(x))))
        return output#torch.cat((x,output),1)
        
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, relu=True,
                 bn=False, bias=False, up_size=0,fan=False):
        super(BasicConv, self).__init__()
        wn = lambda x:torch.nn.utils.weight_norm(x)
        self.out_channels = out_planes
        self.in_channels = in_planes
        if fan:
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.PReLU(out_planes) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x

class one_module(nn.Module):
    def __init__(self, n_feats):
        super(one_module, self).__init__()
        self.layer1 = BasicConv(n_feats, 11,3,1,1)
        self.layer2 = one_conv(11, 6,3)
        self.layer3 = one_conv(11, 6,3)
        #self.layer5 = one_conv(12, 6,3)
        self.layer4 = BasicConv(33, 36,1,1,0)
        #self.layer4 = BasicConv(2*n_feats, n_feats, 3,1,1)
        #self.weight1 = common.Scale(1)
        #self.weight2 = common.Scale(1)
        #self.weight3 = common.Scale(1)
        self.weight4 = common.Scale(1)
        self.weight5 = common.Scale(1)
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        #x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        #x4 = self.layer5(x3)
        #x4 = self.layer4(x3)
        return self.weight4(x) + self.weight5(self.layer4(torch.cat([x1,x3,x2],1)))#self.weight4(x)+self.weight5(x4)


class Un(nn.Module):
    def __init__(self,n_feats, wn):
        super(Un, self).__init__()
        self.encoder1 = one_module(36)
        self.encoder2 = one_module(36)
        # #self.encoder3 = one_module(n_feats)
        # self.decoder1 = one_module(n_feats)
        # #self.decoder2 = one_module(n_feats)
        # #self.up1 = BasicConv(n_feats, n_feats,3,1,1)
        # #self.up2 = BasicConv(n_feats, n_feats,3,1,1)
        # self.down1 = nn.MaxPool2d(kernel_size=2)#BasicConv(n_feats, n_feats,3,2,1)
        # self.down2 = nn.MaxPool2d(kernel_size=2)#BasicConv(n_feats, n_feats,3,2,1)
        # self.up2 = BasicConv(2*n_feats, n_feats,1,1,0, groups=8)
        # self.up1 = BasicConv(2*n_feats, n_feats,1,1,0,groups = 8 )
        self.weight = common.Scale(1)
        self.weight6 = common.Scale(1)
        #self.weight7 = common.Scale(1)
        #self.weight8 = common.Scale(1)
    def forward(self,x):
        x1 = self.encoder1(x)
        #x1 = self.encoder2(x1)
        #x2 = F.interpolate(x1, scale_factor = 2, mode='bilinear', align_corners=True)
        # x2 = self.down1(x1)
        # x2_up = F.interpolate(x2, size = x1.size()[-2:], mode='bilinear', align_corners=True)
        # x2_high = x1-x2_up
        x1 = self.encoder2(x1)
        # #x3 = F.interpolate(x2, scale_factor = 2, mode='bilinear', align_corners=True)
        # #x3 = self.down2(x2)
        # #x3_up = F.interpolate(x3, size = x2.size()[-2:], mode='bilinear', align_corners=True)
        # #x3_high = x2-x3_up
        # #x3 = self.encoder3(x3)
        # x5 = F.interpolate(x2, size = x1.size()[-2:], mode='bilinear', align_corners=True)#self.down1(x3)
        # x5 = x5+x2_high
        # x5 = self.up1(torch.cat([x1,x5],1))
        # x5_1 = self.decoder1(x5)
        #x5 = self.weight7(x5)+self.weight8(x5_1)
        #x6 = F.interpolate(x5, size = x1.size()[-2:], mode='bilinear', align_corners=True)#self.down2(x5)#F.max_pool2d(x5, 3, 2, 1)
        #x6 = x6+x2_high
        #x6 = self.up1(torch.cat([x1,x6],1))
        #x6 = self.decoder2(x6)
        return self.weight(x) + self.weight6(x1)

class RTCV(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RTCV, self).__init__()
        wn = lambda x:torch.nn.utils.weight_norm(x)
        n_feats = 16
        n_blocks = 1
        kernel_size = 3
        scale = args.scale[0] #gaile
        act = nn.ReLU(True)
        #self.up_sample = F.interpolate(scale_factor=2, mode='nearest')
        self.n_blocks = n_blocks
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(1, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [nn.Conv2d(3, 36,5,padding=2),
                        nn.PReLU(36)]
                        #nn.Conv2d(24, 24,5,padding=2),]#[conv(args.n_colors, n_feats, kernel_size)]
        
        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                Un(n_feats=n_feats, wn = wn))

        # define tail module
        modules_tail = [
            #nn.Conv2d(n_feats * (self.n_blocks + 1), n_feats, 1, padding=0, stride=1),
            #conv(self.n_blocks*n_feats, n_feats, kernel_size),
            #BasicConv(self.n_blocks*n_feats, n_feats, kernel_size, groups=3),
            nn.Conv2d(36, 3*(scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale)]
            #common.Upsampler(conv, scale, n_feats, act=False),
            #conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(1, rgb_mean, rgb_std, 1)
        #self.shuff = BasicConv(n_blocks*n_feats, n_feats,3,1,1)
        #self.up = nn.Sequential(common.Upsampler(conv,scale,n_feats,act=False),
        #                  BasicConv(n_feats, 3,3,1,1))
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x1,x2 = None, test=False):
        x1 = self.sub_mean(x1)
        x1 = self.head(x1)
        res2 = x1
        #res2 = x2
        #MSRB_out = []
        for i in range(self.n_blocks):
            x1 = self.body[i](x1)
        #    MSRB_out.append(x1)
        #res1 = x1#torch.cat(MSRB_out,1)
        #res2 = self.body(x2)
        #res1 = self.body(x1)
        #res1 = res1 + x1
        #res2 = res2 + x2
        #MSRB_out.append(res)
        
        #res = torch.cat(MSRB_out,1)
        x1 = self.tail(x1 + res2)
        #x1 = self.up(res2) + x1
        x1 = self.add_mean(x1)
        #x2 = self.tail(res2)
        return x1

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
        #MSRB_out = []from model import common

