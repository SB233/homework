the network structure:


MINSTAdder (
  (net_pool): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  (bc1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True)
  (bc2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True)
  (net1_conv1): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (net1_conv2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (net2_conv1): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (net2_conv2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (fc1): Linear (6272 -> 1024)
  (fc2): Linear (1024 -> 32)
  (fc3): Linear (32 -> 1)
)
