
import time
import torch
import torch.nn.functional as F
torch.manual_seed(0)


x = torch.rand((1,256,29,29))
kernel = torch.rand((256,1,5,5))
start = time.time()

kernel_size = 5
kernel_stride = 5
# x = x.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
# a = a.contiguous().view(a.size(0), a.size(1), -1, a.size(4), a.size(5))
out = F.conv2d(x, kernel, groups=256)
print("The time for conv operation: ", time.time() - start)
start = time.time()
output = torch.rand(1,256,25,25)
# import pdb; pdb.set_trace()
for i in range(256):
    for j in range(25):
        for t in range(25):
            # print(x[0][1].shape)
            # t = x[0][1].unfold(0, kernel_size, 1).unfold(1, kernel_size, 1)
            # torch.matmul(k.view(1,256,5*5), m.contiguous().view(5*5,25,25))
            # output[i, j] = t
            output[0][i][j][t] = torch.sum(torch.mul(x[:,i,j:j+5,t:t+5], kernel[i,:,:,:]))
# torch.mul()

print("The time for conv operation: ", time.time() - start)



