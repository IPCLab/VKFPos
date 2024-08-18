import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResNet import resnet_elu
from .att import AttentionBlock

class ResNetHead(nn.Module):
    def __init__(self):
        super(ResNetHead, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.elu = nn.ELU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.maxpool(x)

        return x

class ResModule(nn.Module):

    def __init__(self, inplanes, planes, blocks_n, stride, layer_idx,  block=resnet_elu.Bottleneck):
        super(ResModule, self).__init__()
        self.module_name = 'layer'+str(layer_idx)
        self.inplanes = inplanes
        self.planes = planes

        self.resModule = nn.ModuleDict({
            self.module_name:  self._make_layer(
                block, self.planes, blocks_n, stride)
        })

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resModule[self.module_name](x)
        return x    

class VKFPosOdom(nn.Module):
    _inplanes = 64
    
    def __init__(self, share_levels_n:int=0, dropout:float=0.2, pooling_size:int=1,
                 block=resnet_elu.Bottleneck):
        super(VKFPosOdom, self).__init__()
        layers = [3, 4, 6, 3]
        strides = [1, 2, 2, 2]
        self.block = block
        self.share_levels_n = share_levels_n
        self.dropout = dropout
        self.pooling_size = pooling_size
        
        self.odom_en_head = ResNetHead()
        
        _layers = []
        self.inplanes = self._inplanes
        for i in range(1, 4): 
            planes = 64*2**(i-1)
            _layers.append(
                ResModule(inplanes=self.inplanes, planes=planes,
                          blocks_n=layers[i-1], stride=strides[i-1], layer_idx=i)
            )
            self.inplanes = planes * block.expansion
        self.odom_encoder = nn.Sequential(*_layers)
        self.odom_final_res = ResModule(inplanes=self.inplanes*2,
                                        planes=64*2**(len(layers)-1),
                                        blocks_n=layers[len(layers)-1], stride=strides[len(layers)-1], layer_idx=len(layers))
        
        self.odom_avgpool = nn.AdaptiveAvgPool2d(pooling_size)
        self.odom_fc1 = nn.Linear(
            2048*pooling_size*pooling_size, 1024)
        self.odom_fcx = nn.Linear(1024, 3)
        self.odom_fcq = nn.Linear(1024, 3)
        
        self.odom_dropout = nn.Dropout(p=self.dropout)
        
    def forward(self, input):
        # input shape = Bxclipsx3xHxW
        # transpose to clipsxBx3xHxW
        input = input.permute(1, 0, 2, 3, 4)
        encode_feature = []
        for img_batch in input:
            out = self.odom_en_head(img_batch)
            out = self.odom_encoder(out)
            encode_feature.append(out)
        out2 = torch.cat(encode_feature, dim=1)
        
        out3 = self.odom_final_res(out2)
        out3 = self.odom_avgpool(out3)
        out3 = out3.view(out3.size(0), -1)
        
        out4 = self.odom_fc1(out3)
        out4 = F.elu(out4)
        out4 = self.odom_dropout(out4)
        
        # predict final part
        x_odom = self.odom_fcx(out4)
        q_odom = self.odom_fcq(out4)
        
        return torch.cat([x_odom, q_odom], dim=1)
 
class VKFPosBoth(nn.Module):
    _inplanes = 64
    
    def __init__(self, training:bool=True, share_levels_n:int=3, dropout:float=0.5, pooling_size:int=1,
                 block=resnet_elu.Bottleneck):
        super(VKFPosBoth, self).__init__()
        
        self.training = training
        self.block = block
        self.share_levels_n = share_levels_n
        self.pooling_size = pooling_size
        
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights='ResNet34_Weights.DEFAULT')
        self.resnet_odom = nn.Sequential(*list(resnet.children())[:-2])

        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights='ResNet34_Weights.DEFAULT')
        self.resnet_global = nn.Sequential(*list(resnet.children())[:-2])

        odom_feature_len = 2048
        self.odom_att = AttentionBlock(odom_feature_len*pooling_size*pooling_size)
        self.global_att = AttentionBlock(2048*pooling_size*pooling_size)
        self.odom_avgpool = nn.AdaptiveAvgPool2d(pooling_size)
        self.odom_expand_fc = nn.Linear(1024, odom_feature_len)
        self.global_expand_fc = nn.Linear(512, 2048)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
        self.odom_fcx = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(odom_feature_len, 3))
        self.odom_fcq = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(odom_feature_len, 3))
        self.odom_fcx_uncer = nn.Sequential(nn.Dropout(dropout),
                                            nn.Linear(odom_feature_len, 3))
        self.odom_fcq_uncer = nn.Sequential(nn.Dropout(dropout),
                                            nn.Linear(odom_feature_len, 3))
        
        global_feature_len = 2048
        
        self.global_avgpool = nn.AdaptiveAvgPool2d(pooling_size)
        predictor_count = 1

        self.global_fcxs = nn.ModuleList([nn.Sequential(
                                            nn.Dropout(dropout),
                                            nn.Linear(global_feature_len, 3)) 
                                            for i in range(predictor_count)])

        self.global_fcqs = nn.ModuleList([nn.Sequential(
                                            nn.Dropout(dropout),
                                            nn.Linear(global_feature_len, 3))  
                                            for i in range(predictor_count)])
        self.global_fcx_uncers = nn.ModuleList([nn.Sequential(
                                            nn.Dropout(dropout),
                                            nn.Linear(global_feature_len, 3)) 
                                            for i in range(predictor_count)])
        
        self.global_fcq_uncers = nn.ModuleList([nn.Sequential(
                                            nn.Dropout(dropout),
                                            nn.Linear(global_feature_len, 3))  
                                            for i in range(predictor_count)])
        
        
        self.odom_param_list = [self.resnet_odom.parameters(),
                                self.odom_avgpool.parameters(),
                                self.odom_att.parameters(),
                                self.odom_expand_fc.parameters(),
                                self.odom_fcx.parameters(),
                                self.odom_fcq.parameters(),
                                self.odom_fcx_uncer.parameters(),
                                self.odom_fcq_uncer.parameters()]
        
        self.global_param_list =[self.resnet_global.parameters(),
                                self.global_att.parameters(),
                                self.global_expand_fc.parameters(),
                                self.global_avgpool.parameters(),
                                self.global_fcxs.parameters(),
                                self.global_fcqs.parameters(),
                                self.global_fcx_uncers.parameters(),
                                self.global_fcq_uncers.parameters()]
        
        self.global_odom_param_list = [self.resnet_odom.parameters(),
                                        self.odom_att.parameters(),
                                        self.global_att.parameters(),
                                        self.odom_avgpool.parameters(),
                                        self.odom_expand_fc.parameters(),
                                        self.global_expand_fc.parameters(),
                                        self.odom_fcx.parameters(),
                                        self.odom_fcq.parameters(),
                                        self.global_avgpool.parameters(),
                                        self.global_fcxs.parameters(),
                                        self.global_fcqs.parameters()]
        
        self.uncertainty_param_list = [ self.odom_fcx_uncer.parameters() ,
                                        self.odom_fcq_uncer.parameters(),
                                        self.global_fcx_uncers.parameters(),
                                        self.global_fcq_uncers.parameters()]
        
        init_modules = [self.odom_avgpool,
                        self.odom_att,
                        self.odom_expand_fc,
                        self.odom_fcx,
                        self.odom_fcq,
                        self.odom_fcx_uncer,
                        self.odom_fcq_uncer,
                        self.global_att,
                        self.global_expand_fc,
                        self.global_avgpool,
                        self.global_fcxs,
                        self.global_fcqs,
                        self.global_fcx_uncers,
                        self.global_fcq_uncers]
        
        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        
    def forward(self, input):
        # input shape = Bxclipsx3xHxW
        # transpose to clipsxBx3xHxW
        input = input.permute(1, 0, 2, 3, 4)
        clips = input.shape[0]
        # for encode part
        odom_encode_feature = []
        global_encode_feature = []
        
        for (i, img_batch) in enumerate(input):
            # for odom
            if i >= (clips-2):
                odom_encode_feature.append(self.resnet_odom(img_batch)) # Bx512x8x11
            # for global
            global_encode_feature.append(self.resnet_global(img_batch)) # Bx512x8x11
           
        
        # # for odom
        odom_encode = torch.cat(odom_encode_feature, dim=1)
        odom_encode_2 = self.odom_avgpool(odom_encode) # Bx4096x1x1
        odom_encode_2 = odom_encode_2.view(odom_encode_2.size(0), -1) # Bx4096
        odom_encode_2 = self.odom_expand_fc(odom_encode_2) # 2048
        
    
        odom_encode_2 = self.relu(odom_encode_2)
        odom_encode_att = self.odom_att(odom_encode_2)
     
        x_odom = self.odom_fcx(odom_encode_att)
        q_odom = self.odom_fcq(odom_encode_att)

        # uncertainty prediction
        x_odom_uncer = self.odom_fcx_uncer(odom_encode_att)
        q_odom_uncer = self.odom_fcq_uncer(odom_encode_att)
        
        
        # for global
        if self.training:
            global_encode_1 = torch.concat([feature for feature in global_encode_feature[:-1]], axis=1)
            global_encode_1 = self.global_avgpool(global_encode_1)
            global_encode_1 = global_encode_1.view(global_encode_1.size(0), -1)
            global_encode_1 = self.global_expand_fc(global_encode_1)
            
            global_encode_1 = self.relu(global_encode_1)
            global_encode_att_1 = self.global_att(global_encode_1)
            x_global_p = torch.stack([global_fcx(global_encode_att_1)
                                      for global_fcx in self.global_fcxs], dim=1)
            q_global_p = torch.stack([global_fcq(global_encode_att_1)
                                      for global_fcq in self.global_fcqs], dim=1)
            
            x_global_uncer_p = torch.stack([global_fcx_uncer(global_encode_att_1)
                                      for global_fcx_uncer in self.global_fcx_uncers], dim=1)
            q_global_uncer_p = torch.stack([global_fcq_uncer(global_encode_att_1)
                                      for global_fcq_uncer in self.global_fcq_uncers], dim=1)

        global_encode_2 = torch.concat([feature for feature in global_encode_feature[1:]], axis=1)
        global_encode_2 = self.global_avgpool(global_encode_2)
        global_encode_2 = global_encode_2.view(global_encode_2.size(0), -1)
        global_encode_2 = self.global_expand_fc(global_encode_2)
        
        global_encode_2 = self.relu(global_encode_2)
        global_encode_att_2 = self.global_att(global_encode_2)

        x_global = torch.stack([global_fcx(global_encode_att_2)
                                      for global_fcx in self.global_fcxs], dim=1)
        q_global = torch.stack([global_fcq(global_encode_att_2)
                                      for global_fcq in self.global_fcqs], dim=1)

        # # uncertainty prediction
        x_global_uncer = torch.stack([global_fcx_uncer(global_encode_att_2)
                                      for global_fcx_uncer in self.global_fcx_uncers], dim=1)
        q_global_uncer = torch.stack([global_fcq_uncer(global_encode_att_2)
                                      for global_fcq_uncer in self.global_fcq_uncers], dim=1)
        
        
        if self.training:
            x_global = torch.cat([x_global_p.unsqueeze(2), x_global.unsqueeze(2)], dim=2)
            q_global = torch.cat([q_global_p.unsqueeze(2), q_global.unsqueeze(2)], dim=2)
            x_global_uncer = torch.cat([x_global_uncer.unsqueeze(2), x_global_uncer_p.unsqueeze(2)], dim=2)
            q_global_uncer = torch.cat([q_global_uncer.unsqueeze(2), q_global_uncer_p.unsqueeze(2)], dim=2)
            
        return torch.cat([x_odom, q_odom, x_odom_uncer, q_odom_uncer], dim=1), torch.cat([x_global, q_global, x_global_uncer, q_global_uncer], dim=-1)
    
    
if __name__ == '__main__':
    from torchview import draw_graph
    device = torch.device("cuda:0")
    input = torch.rand((8, 2, 3, 640, 480), device=device)
    tgt = torch.rand((8, 3, 6)) 
    model = VKFPosBoth().to(device)
    odom_output, global_output = model(input) # in training mode
    output = model(input)
    model_graph = draw_graph(model, input_size=(8, 2, 3, 640, 480), 
                             device="cuda:0", expand_nested=True)
    model_graph.visual_graph.render(format='png')

        
        