import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchsummaryX import summary
import visdom
import time

MAX_SUBNETS = 2
NUM_STAGE = [1, 2]
BN_MOMENTUM = 0.1
STEM_RES_CHANNEL = 64
STAGE_NUM_BLOCK = 4


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck,self).__init__()
        '''
        1x1 conv + BN + relu
        3x3 conv + BN + relu
        1x1 conv + BN
        + Residual connection + relu
        -> bottleneck form resnet, so use "add" instead concate
        original : 64 -> 64 -> 256 depth
        -> however, here uses 64->64->64
        '''
        if in_channels != out_channels:
            print("bottleneck layer is not for transition. use trans_bottleneck")
            assert(0)

        self.conv1 = nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = 64,
                            kernel_size = 1, 
                            stride = 1, 
                            padding= (0,0),
                            bias = False)
        self.conv2 = nn.Conv2d(
                            in_channels = 64, 
                            out_channels = 64, 
                            kernel_size = 3, 
                            stride = 1, 
                            padding= (1,1),
                            bias = False)
        self.conv3 = nn.Conv2d(
                            in_channels = 64, 
                            out_channels = out_channels, 
                            kernel_size = 1, 
                            stride = 1, 
                            padding= (0,0),
                            bias = False)

        self.BN1 = nn.BatchNorm2d(
                            num_features = 64,
                            momentum = BN_MOMENTUM)
        self.BN2 = nn.BatchNorm2d(
                            num_features = 64,
                            momentum = BN_MOMENTUM)
        self.BN3 = nn.BatchNorm2d(
                            num_features = 64,
                            momentum = BN_MOMENTUM)

    def forward(self,input):
        out = self.conv1(input)
        out = F.relu(self.BN1(out))
        out = self.conv2(out)
        out = F.relu(self.BN2(out))
        out = self.conv3(out)
        out = self.BN3(out)

        out = out + input
        out = F.relu(out)

        return out

# The detailed structure is from official code
class Img2channel(nn.Module):
    def __init__(self):
        super(Img2channel,self).__init__()
        self.conv1 = nn.Conv2d(
                            in_channels = 3,
                            out_channels = 64,
                            kernel_size = 3,
                            stride = 2,
                            padding = (1,1),
                            bias = False)
        self.conv2 = nn.Conv2d(
                            in_channels = 64,
                            out_channels = 64,
                            kernel_size = 3,
                            stride = 2,
                            padding = (1,1),
                            bias = False)

        self.BN1 = nn.BatchNorm2d(
                            num_features = 64,
                            momentum= BN_MOMENTUM)
        self.BN2 = nn.BatchNorm2d(
                            num_features = 64,
                            momentum= BN_MOMENTUM)

    def forward(self,input):
        out = F.relu(self.BN1(self.conv1(input)))
        out = F.relu(self.BN2(self.conv2(out)))

        return out

# it sets the bias as False!
class Base_Block(nn.Module):
    def __init__(self,num_channels):
        super(Base_Block, self).__init__()

        self.conv1 = nn.Conv2d(
                            in_channels = num_channels,
                            out_channels = num_channels,
                            kernel_size = 3,
                            stride = 1,
                            padding = (1,1),
                            bias = False)
        self.conv2 = nn.Conv2d(
                            in_channels = num_channels,
                            out_channels = num_channels,
                            kernel_size = 3,
                            stride = 1,
                            padding = (1,1),
                            bias = False)
        
        self.BN1 = nn.BatchNorm2d(
                            num_features = num_channels,
                            momentum = BN_MOMENTUM)
        self.BN2 = nn.BatchNorm2d(
                            num_features = num_channels,
                            momentum = BN_MOMENTUM)

    def forward(self,input):
        out = F.relu(self.BN1(self.conv1(input)))
        out = self.BN2(self.conv2(out))

        out = out + input
        out = F.relu(out)

        return out

class HRNet_Stage(nn.Module):
    def __init__(self, num_channels, num_subNets, device, is_expand = False):
        super(HRNet_Stage,self).__init__()
        self.is_expand = is_expand
        self.num_subNets = num_subNets
        self.device = device

        '''repeat residual blocks'''
        self.subNets = []
        for i in range(num_subNets):
            Blocks = []
            channels = num_channels * (2**i)
            for j in range(STAGE_NUM_BLOCK):
                Blocks.append(Base_Block(channels))
            self.subNets.append(nn.Sequential(*Blocks).to(device))
        
        '''exchange stage'''
        self.exchanges = []
        for order in range(num_subNets):
            self.exchanges.append(self._exchange_init_(num_channels,num_subNets,order))

        '''expand stage'''
        if is_expand : 
            expand = []
            expand.append(nn.Conv2d(
                                    in_channels = num_channels * (2**(num_subNets-1)),
                                    out_channels = num_channels * (2**num_subNets),
                                    kernel_size = 3,
                                    stride = 2,
                                    padding = (1,1),
                                    bias = False
            ))
            expand.append(nn.BatchNorm2d(
                                    num_features = num_channels * (2**num_subNets),
                                    momentum = BN_MOMENTUM
            ))
            expand.append(nn.ReLU())
            self.expand = nn.Sequential(*expand).to(self.device)

    def _exchange_init_(self,num_channels, num_subNets, current_order):
        if current_order >= num_subNets:
            print("wrong order {}/{}".format(current_order,num_subNets))
            assert(0)
        result = []
        num_up = current_order
        num_down = num_subNets - current_order - 1
        channels = num_channels * (2**current_order)

        if num_up>0:
            for i in range(num_up):
                module = []
                module.append(nn.Conv2d(
                                    in_channels = channels,
                                    out_channels = channels//(2**(num_up-i)),
                                    kernel_size = 1,
                                    stride = 1,
                                    padding = (0,0),
                                    bias = False
                ))
                module.append(nn.Upsample(
                                    scale_factor = 2**(num_up-i),
                                    mode = 'nearest'
                ))
                module.append(nn.BatchNorm2d(
                                    num_features = channels//(2**(num_up-i)),
                                    momentum = BN_MOMENTUM
                ))
                module.append(nn.ReLU())
                result.append(nn.Sequential(*module).to(self.device))
        
        result.append(nn.Identity().to(self.device))

        if num_down>0:
            for i in range(num_down):
                module = []
                channel = channels
                for j in range(i+1):
                    module.append(nn.Conv2d(
                                    in_channels = channel,
                                    out_channels = channel*2,
                                    kernel_size = 3,
                                    stride = 2,
                                    padding = (1,1),
                                    bias = False
                    ))
                    module.append(nn.BatchNorm2d(
                                    num_features = channel*2,
                                    momentum = BN_MOMENTUM
                    ))
                    module.append(nn.ReLU())
                    channel = channel*2
                result.append(nn.Sequential(*module).to(self.device))
        
        return result
        
    def forward(self,input):
        
        ''' lower order means higher resolution '''

        '''res_blocks'''
        inter_out = []
        
        for net_i in range(self.num_subNets):
            out = self.subNets[net_i](input[net_i])
            inter_out.append(out)

        '''exchange_stage'''
        final_out = []
        #final_out.retain_grad()

        for net_i in range(self.num_subNets):
            for i in range(self.num_subNets):
                if net_i == 0:
                    final_out.append(self.exchanges[net_i][i](inter_out[net_i]).clone())
                else:
                    final_out[i] += self.exchanges[net_i][i](inter_out[net_i])
        
        '''expand'''
        if self.is_expand:
            smallest = final_out[self.num_subNets-1]
            out = self.expand(smallest)
            final_out.append(out)
        
        return final_out

class HRNet_Final(nn.Module):
    def __init__(self,num_channels,num_subNets,device):
        super(HRNet_Final,self).__init__()
        self.num_subNets = num_subNets
        self.device = device

        '''repeat residual blocks'''
        self.subNets = []
        for i in range(num_subNets):
            Blocks = []
            channels = num_channels * (2**i)
            for j in range(STAGE_NUM_BLOCK):
                Blocks.append(Base_Block(channels))
            self.subNets.append(nn.Sequential(*Blocks).to(device))

        '''connect all blocks into one'''
        self.final = self._fusing_(num_channels,num_subNets)
    
    def _fusing_(self,num_channels,num_subNets):
        fuse = []
        for i in range(num_subNets-1):
            module = []
            module.append(nn.Conv2d(
                                in_channels = num_channels*(2**(i+1)),
                                out_channels = num_channels,
                                kernel_size = 1,
                                stride = 1,
                                padding = (0,0),
                                bias = False
            ))
            module.append(nn.Upsample(
                                scale_factor = 2**(i+1),
                                mode = 'nearest'
            ))
            module.append(nn.BatchNorm2d(
                                num_features = num_channels,
                                momentum = BN_MOMENTUM
            ))
            module.append(nn.ReLU())
            fuse.append(nn.Sequential(*module).to(self.device))
        return fuse
    
    def forward(self,input):
        '''res_blocks'''
        inter_out = []
        for net_i in range(self.num_subNets):
            out = self.subNets[net_i](input[net_i])
            inter_out.append(out)

        output = inter_out[0]
        for net_i in range(self.num_subNets-1):
            output += self.final[net_i](inter_out[net_i+1])
        
        return output

# It's simply the first stage
class back_HRNet(nn.Module):
    def __init__(self, device, is_debug, N_RESIDUAL, MAX_SUBNETS, N_STAGE, N_CHANNELS):
        super(back_HRNet,self).__init__()
        self.num_res = N_RESIDUAL
        self.num_stage = N_STAGE
        self.max_subnets = MAX_SUBNETS
        self.device = device

        self.is_debug = is_debug
        if is_debug:
            self.vis = visdom.Visdom(env="HRNet")
            self.env_list = ["first_res",
                             "second_res",
                             "third_res",
                             "fourth_res",
                             "input"]
            self.stage_list = ["first_stage",
                               "second_stage",
                               "thrid_stage",
                               "fourth_stage"]
            self.cnt = 0
            for i in range(len(self.env_list)):
                self.vis.delete_env(env= self.env_list[i])

        if len(N_STAGE) != MAX_SUBNETS:
            print("num stage : {}, max subnets : {}, different!".format(N_STAGE,
                                                                        MAX_SUBNETS))

        '''stem + first stage'''
        res_units = []
        res_units.append(Img2channel())
        for i in range(self.num_res):
            res_unit = Bottleneck(STEM_RES_CHANNEL,STEM_RES_CHANNEL)
            res_units.append(res_unit)
        self.first_stage = nn.Sequential(*res_units)
        self.first_trans = []
        for i in range(2):
            module = []
            module.append(nn.Conv2d(
                            in_channels = STEM_RES_CHANNEL,
                            out_channels = N_CHANNELS*(i+1),
                            kernel_size = 3,
                            stride = i+1,
                            padding = (1,1),
                            bias = False
            ))
            module.append(nn.BatchNorm2d(
                            num_features = N_CHANNELS*(i+1),
                            momentum = BN_MOMENTUM
            ))
            module.append(nn.ReLU())
            self.first_trans.append(nn.Sequential(*module).to(device))

        if self.num_stage[0] != 1:
            print("setting first stage longer than one isn't implemented yer")
            assert(0)


        '''from second to fourth stage'''
        self.stages = []
        for step in range(MAX_SUBNETS-1):
            self.stages.append(self._make_steps_(N_CHANNELS,step+2,N_STAGE[step+1]))

        

    def _make_steps_(self,N_CHANNELS,num_subNets,num_stage):
        stage = []
        for step in range(num_stage):
            if step != num_stage-1:
                stage.append(HRNet_Stage(N_CHANNELS,num_subNets,self.device))
            else:
                stage.append(HRNet_Stage(N_CHANNELS,num_subNets,self.device,True))
        return stage

    def forward(self,input):
        print_img = False
        if self.is_debug:
            self.cnt += 1
            if self.cnt == 10:
                self.cnt = 0
                print_img = True
            else:
                print_img = False
        
        if print_img:
            self.vis.image(input[0],
                           env=self.env_list[4],
                           opts=dict(title=self.env_list[4]))
        
        
        first_out = self.first_stage(input)
        output = []
        for i in range(2):
            output.append(self.first_trans[i](first_out))
            
            '''visdom'''
            if print_img:
                temp = output[i][0].detach()
                temp = temp.unsqueeze(1)/torch.max(temp)
                self.vis.images(
                        temp,
                        env=self.env_list[i],
                        opts=dict(title=self.stage_list[0]+"_"+self.env_list[i]))
            
            '''dead-relu counting'''
            if self.is_debug:
                temp = torch.sum(output[i][0] == 0)
                size = output[i][0].view(-1).size()[0]
                print("{}_{} dead relu:{}, dead rate:{}".format(self.stage_list[0],
                                                                  self.env_list[i],
                                                                  temp,
                                                                  temp/size*100))
                
    

        for i in range(self.max_subnets-1):
            for j in range(self.num_stage[i+1]):
                output = self.stages[i][j](output)
            if print_img:
                for k in range(i+2):
                    temp = output[k][0].detach()
                    temp = temp.unsqueeze(1)/torch.max(temp)
                    self.vis.images(
                        temp,
                        env=self.env_list[k],
                        opts=dict(title=self.stage_list[i+1]+"_"+self.env_list[k]))
            
            '''dead-relu counting'''
            if self.is_debug:
                for k in range(i+2):
                    temp = torch.sum(output[k][0] == 0)
                    size = output[k][0].view(-1).size()[0]
                    print("{}_{} dead relu:{}, dead rate:{}".format(self.stage_list[i+1],
                                                                    self.env_list[k],
                                                                    temp,
                                                                    temp/size*100))

        return output


class HRNet(nn.Module):
    def __init__(self, 
                 device, 
                 is_debug = False, 
                 N_RESIDUAL = STAGE_NUM_BLOCK, 
                 MAX_SUBNETS = MAX_SUBNETS, 
                 N_STAGE = NUM_STAGE, 
                 N_CHANNELS = 32,
                 Version = 1):
        
        super(HRNet,self).__init__()
        self.version = Version
        
        self.backbone = back_HRNet(device, 
                                   is_debug, 
                                   N_RESIDUAL, 
                                   MAX_SUBNETS,
                                   N_STAGE,
                                   N_CHANNELS)
        
        '''Final Module'''
        if Version==2:
            self.final = HRNet_Final(N_CHANNELS,MAX_SUBNETS,device)
        
        self.to_heat = nn.Conv2d(
                            in_channels = N_CHANNELS,
                            out_channels = 17,
                            kernel_size = 1,
                            stride = 1,
                            padding= (0,0)
        )
        
        init_weights(self)

    def forward(self,input):
        output = self.backbone(input)
        if self.version == 2:
            output = self.final(output)
        else:
            output = output[0]
        output = self.to_heat(output)

        return output


class Add_Module(nn.Module):
    def __init__(self,device, N_Module,N_out,N_CHANNEL,N_JOINTS=17):
        super(Add_Module,self).__init__()
        self.device = device
        self.N_Module = N_Module
        self.init_conv = nn.Conv2d(
                            in_channels = N_CHANNEL,
                            out_channels = N_JOINTS*2,
                            kernel_size = 1,
                            stride = 1,
                            padding = (0,0))
        
        '''stack the upconv modules '''
        for i in range(N_Module):
            if i == 0:
                deconv,res = self._upscale_(N_out,N_CHANNEL,N_JOINTS)
                self.deconv_modules = [deconv]
                self.res_modules = [res]
            else:
                deconv,res = self._upscale_(N_out,N_out,N_JOINTS)
                self.deconv_modules.append(deconv)
                self.res_modules.append(res)

    def _upscale_(self,N_out,N_CHANNEL,N_JOINTS):
        input_depth = N_CHANNEL+N_JOINTS
        deconv_blocks = []
        deconv_blocks.append(nn.ConvTranspose2d(
                            in_channels = input_depth,
                            out_channels = N_out,
                            kernel_size = 4,
                            stride = 2,
                            padding = 1,
                            output_padding = 0,
                            bias = False))
        deconv_blocks.append(nn.BatchNorm2d(
                            num_features=N_out,
                            momentum=BN_MOMENTUM))
        deconv_blocks.append(nn.ReLU())
        deconv_block = nn.Sequential(*deconv_blocks).to(self.device)

        res_blocks = []
        for i in range(4):
            res_blocks.append(Base_Block(N_out))
        res_blocks.append(nn.Conv2d(
                            in_channels = N_out,
                            out_channels = N_JOINTS*2,
                            kernel_size = 1,
                            stride = 1,
                            padding = (0,0)))
        res_block = nn.Sequential(*res_blocks).to(self.device)
        
        return deconv_block, res_block

    def forward(self, input):
        feature_out = []
        pose_out = []
        feature_out.append(input)
        pose_out.append(self.init_conv(input))

        for i in range(self.N_Module):
            deconv_input = torch.cat((feature_out[i],pose_out[i]),dim = 1)
            deconv_output = self.deconv_modules[i](deconv_input)
            feature_out.append(deconv_output)
            pose_out.append(self.res_modules[i](deconv_output))
        
        return pose_out

class HigherHRNet(nn.Module):
    def __init__(self, 
                 device, 
                 N_Module = 1, 
                 N_out = 48,
                 N_RESIDUAL = STAGE_NUM_BLOCK,
                 MAX_SUBNETS = MAX_SUBNETS, 
                 N_STAGE = NUM_STAGE,
                 N_CHANNELS = 32):
        
        super(HigherHRNet,self).__init__()
        self.NORMALIZE = True
        self.N_Module = N_Module
        '''Add some code for pre-trained HRNet loader'''
        self.backbone = back_HRNet(device, N_RESIDUAL, MAX_SUBNETS, N_STAGE, N_CHANNELS)
        self.add_mod = Add_Module(device,N_Module, N_out, N_CHANNELS)

    def forward(self,input):
        output = self.backbone(input)
        outputs = self.add_mod(output)

        return outputs
    
    def predict(self,input):
        outputs = self.forward(input)

        pred = outputs[self.N_Module]
        for i in range(self.N_Module):
            temp = torch.nn.UpsamplingNearest2d(scale_factor=2**(self.N_Module-i))
            pred += temp(outputs[i])
        pred = torch.div(pred,self.N_Module+1)
        
        # Normalize option
        if self.NORMALIZE :
            pred = torch.div(pred,torch.max(pred,dim=(1,2)))
        
        return pred


def init_weights(model):
    for m in model.modules():
        
        if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.normal_(m.weight, std=0.001)
            for name, _ in m.named_parameters():
                if name in ['bias']:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)
            for name, _ in m.named_parameters():
                if name in ['bias']:
                    nn.init.constant_(m.bias, 0)

'''For debugging'''
if __name__ == '__main__':
    bottle = HRNet(torch.device("cpu"))
    
    '''
    with torch.autograd.detect_anomaly():
        hrnet = HRNet()
        test_Net = HigherHRNet()
        rand_input = torch.randn(10,3,256,192)
        #print(test_Net)
        #summary(hrnet,rand_input)

        rand_input = torch.autograd.Variable(rand_input)

        output = test_Net(rand_input)
        print(1)

        loss = []
        for i in range(2):
            loss.append(torch.sum(output[i]))
            

        print(loss)
        for i in range(2):
            loss[i].backward(retain_graph=True)
    '''

