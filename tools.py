import torch 
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

class CA_Block(nn.Module):
    def __init__(self,num_filterss,reduction=16):
        super(CA_Block, self).__init__()
        self.se_module=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_filterss,num_filterss//reduction,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filterss//reduction,num_filterss,kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=x*self.se_module(x)
        return x

class SA_Block(nn.Module):
    def __init__(self):
        super(SA_Block, self).__init__()
        self.fus_module = nn.Sequential(
            nn.Conv2d(2, 1, (7,1), padding=(3,1)),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # 1 is channel
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        fus_out = torch.cat([avg_out, max_out], dim=1)
        x = x * self.fus_module(fus_out)
        return x

class Mul_Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size) -> None:
        super(Mul_Conv, self).__init__() # [8,16] kernel_size = [8,16]
        # if kernel_size[0] != kernel_size[1]:
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (kernel_size[0]*3,kernel_size[1]*3),stride=(kernel_size[0],kernel_size[1]),padding=(kernel_size[0],kernel_size[1])),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (kernel_size[0]*5,kernel_size[1]*5),stride=(kernel_size[0],kernel_size[1]),padding=(kernel_size[0]*2,kernel_size[1]*2)),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (kernel_size[0]*7,kernel_size[1]*7),stride=(kernel_size[0],kernel_size[1]),padding=(kernel_size[0]*3,kernel_size[1]*3)),
            nn.ReLU(inplace=True)
        ) 
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (kernel_size[0],kernel_size[1]),stride=(kernel_size[0],kernel_size[1])),
            nn.ReLU(inplace=True)
        )
        # else:
        #     self.conv1 = nn.Sequential(
        #         nn.Conv2d(in_channel, out_channel, (kernel_size[0]*3,kernel_size[1]*3), padding=(kernel_size[0]-1, kernel_size[0])),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.conv2 = nn.Sequential(
        #         nn.Conv2d(in_channel, out_channel, (kernel_size[0]*5,kernel_size[1]*5), padding=(kernel_size[0], kernel_size[0]+1)),
        #         nn.ReLU(inplace=True)
        #     )
        #     self.conv3 = nn.Sequential(
        #         nn.Conv2d(in_channel, out_channel, (kernel_size[0]*7,kernel_size[1]*7), padding=(kernel_size[0]+1, kernel_size[0]+2)),
        #         nn.ReLU(inplace=True)
        #     )
        
        # self.fusio = nn.Sequential(
        #     nn.Conv2d(out_channel*3,out_channel, (1,1)),
        #     nn.ReLU(inplace=True)
        # )
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self,x):
        branch1 = self.pool(self.conv1(x)) 
        branch2 = self.pool(self.conv2(x)) 
        branch3 = self.pool(self.conv3(x)) 
        branch4 = self.pool(self.conv4(x))
        concat = torch.cat([branch1,branch2,branch3,branch4],1)
        # fusion = self.fusio(fusion)

        return concat

if __name__ == "__main__":
    x = torch.rand(32,32,16,8)
    convs =  Mul_Conv(in_channel=32,out_channel=32,kernel_size=[1,1])
    x = convs(x)
    print(x.shape)

        