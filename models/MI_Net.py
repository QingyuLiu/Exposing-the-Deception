import torch
import torch.nn as nn
from models.resnet import resnet34
from torch.nn import init
from models.efficientnet import EfficientNet
from models.mobilenet import MobileNetV1
import logging
class MI_Net(nn.Module):
    def __init__(self, model='resnet34',num_regions=4,num_classes=2,freeze_fc=False,dropout=0.5)-> object:
        super(MI_Net, self).__init__()

        self.num_regions=num_regions
        logging.info(f'Now has {num_regions} region models')
        self.region_models = []
        for i in range(num_regions):
            if model == 'resnet':
                layer = resnet34(pretrained=True)
            elif model == 'mobilenet':
                layer = MobileNetV1()
            elif model == 'efficientnet':
                layer = EfficientNet(pretrained=True)
            else:
                logging.error("please choose the tpye of backbone in Local Information Block.")
            layer_name = 'region_model{}'.format(i + 1)
            self.add_module(layer_name, layer)
            self.region_models.append(layer_name)

        in_size = get_output_size(getattr(self, 'region_model1'))
        self.local_linears = []
        for i in range(num_regions):
            local_linear=nn.Linear(in_size * (num_regions - 1), num_classes)
            layer_name = 'local_linear{}'.format(i + 1)
            self.add_module(layer_name, local_linear)
            self.local_linears.append(layer_name)
        #

        self.bottleneck = ChannelCompress(in_ch=in_size*num_regions, out_ch=in_size)
        # self.global_model=ResNet_18()

        self.baseline_linear = nn.Linear(in_size*num_regions, num_classes)  # *(num_regions+1)

        self.linear = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_size, num_classes)
        )
        # self.linear = nn.Linear(in_size, num_classes)
        if freeze_fc:
            freeze(self.linear)

    def forward(self, x):
        features=[]
        for i, layer_name in enumerate(self.region_models):
            layer = getattr(self, layer_name)
            feature = layer(x)
            features.append(feature)

        feature = torch.cat(features, dim=1)

        p_y_given_f1_f2_f3_f4=self.baseline_linear(feature)

        global_feature_z = self.bottleneck(feature) # Fusion layer

        p_y_given_z = self.linear(global_feature_z)

        p_y_given_f1_fn_list=[]

        for i, layer_name in enumerate(self.local_linears):
            local_linear = getattr(self, layer_name)
            tmp=features.copy()
            tmp.pop(i)
            f1_fn_except_i = torch.cat(tmp, dim=1)
            p_y_given_f1_fn_except_i=local_linear(f1_fn_except_i)
            p_y_given_f1_fn_list.append(p_y_given_f1_fn_except_i)

        return {'p_y_given_z': p_y_given_z, 'p_y_given_f_all': p_y_given_f1_f2_f3_f4,
                'p_y_given_f1_fn_list': p_y_given_f1_fn_list}

    def initNetParams(self,net):
        '''Init net parameters.'''
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-3)
                if m.bias:
                    init.constant(m.bias, 0)


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False

class ChannelCompress(nn.Module):
    def __init__(self, in_ch=2048, out_ch=256,dropout=0.5):
        """
        reduce the amount of channels to prevent final embeddings overwhelming shallow feature maps
        out_ch could be 512, 256, 128
        """
        super(ChannelCompress, self).__init__()
        num_bottleneck = 1000
        add_block = []
        add_block += [nn.Linear(in_ch, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.ReLU()]
        add_block += [nn.Dropout(p=dropout)]
        add_block += [nn.Linear(num_bottleneck, 500)]
        add_block += [nn.BatchNorm1d(500)]
        add_block += [nn.ReLU()]
        add_block += [nn.Dropout(p=dropout)]
        add_block += [nn.Linear(500, out_ch)]

        # Extra BN layer, need to be removed
        #add_block += [nn.BatchNorm1d(out_ch)]

        add_block = nn.Sequential(*add_block)
        # add_block.apply(weights_init_kaiming)
        self.model = add_block

    def forward(self, x):
        x = self.model(x)
        return x

def get_output_size(net):
    input = torch.randn(1,3,224, 224)
    # torch.Size([250, 3, 224, 224])
    output=net(input)
    return output.size(1)