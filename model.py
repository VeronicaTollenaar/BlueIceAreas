# define CNN model, script based on U-Net implementaion on https://github.com/aladdinpersson/Machine-Learning-Collection
# import packages
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# define double convolution
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True):
        super(DoubleConv, self).__init__()
        # define convolutions if batchnorm is used
        if batchnorm:
            self.conv = nn.Sequential(
                # first convolution
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                # second convolution
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),      
            )
        # define convolutions if batchnorm is not used
        else:
            self.conv = nn.Sequential(
                # first convolution
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                # second convolution
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),      
            )
    def forward(self, x):
        return self.conv(x)

# define UNET
class UNET(nn.Module):
    # define parameters (standard values that can be adjusted (e.g., in_channels=10 in final model))
    # features represents number of channels at end of each double conv within different depths of the network
    def __init__(
            self, in_channels=4, out_channels=1, features=[64, 128, 256, 512],
            batchnorm=True, p_dropout=0.,
            ):
                super(UNET, self).__init__()
                self.ups = nn.ModuleList()
                self.downs = nn.ModuleList()
                # define max pooling
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

                # Down part of UNET
                for feature in features[:-1]:
                    self.downs.append(DoubleConv(in_channels, feature, batchnorm))
                    in_channels = feature
                
                # Up part of UNET
                for feature in reversed(features):
                    self.ups.append(
                        nn.ConvTranspose2d(
                            feature*2, feature, kernel_size=2, stride=2,
                        )
                    )
                    self.ups.append(DoubleConv(feature*2, feature, batchnorm))
                
                # Bottleneck of UNET
                self.prebottleneck = DoubleConv(features[-2], features[-1], batchnorm)
                self.bottleneck = DoubleConv(features[-1], features[-1]*2, batchnorm)
                self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
                self.sigmoid = nn.Sigmoid()
                self.dropout = nn.Dropout(p_dropout)
    # apply forward pass
    def forward(self, x):
        skip_connections = []
        # down 0,1,2
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        # down 3
        x = self.prebottleneck(x)
        skip_connections.append(x)
        x = self.dropout(x)
        x = self.pool(x)
        # down 4
        x = self.bottleneck(x)
        x = self.dropout(x)
        skip_connections = skip_connections[::-1]
        # up 3,2,1,0
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        # final conv
        x = self.final_conv(x)
        
        return x

def test():
    x = torch.randn((1, 3, 160, 160))
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    # print(preds.shape)
    # print(x.shape)
    # assert preds.shape == x.shape

if __name__ == "__main__":
    test()