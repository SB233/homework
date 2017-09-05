

### second network structure:

```
MINSTAdder (
  (net_pool): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  (bc1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True)
  (bc2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True)
  (net1_conv1): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (net1_conv2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (net1_conv3): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (net2_conv1): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (net2_conv2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (net2_conv3): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (fc1): Linear (1152 -> 1024)
  (fc2): Linear (1024 -> 256)
  (fc3): Linear (256 -> 1)
)
```

acc: 96.86%

lesson learned: when the training loss is stable,
we can increase the neural units in the network to have more parameters to tune,
which can increase the accuracy.

### improvement over source template:
1. add cuda()
2. epoch --> 30
3. use Adam optimizer instead of SGD
4. add batch normalization and relu activation function
5. add evaluation and visualization module

### first network structure trial:

2 conv2d layers

```
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
```

acc: 75.46%
