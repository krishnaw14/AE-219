import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_cuda(x):
  return x.to(device)

def to_onehot(x, num_classes=10):
    assert isinstance(x, int) or isinstance(x, (torch.LongTensor, torch.cuda.LongTensor))
    if isinstance(x, int):
        c = torch.zeros(1, num_classes).long()
        c[0][x] = 1
    else:
        x = x.cpu()
        c = torch.LongTensor(x.size(0), num_classes)
        c.zero_()
        c.scatter_(1, x, 1) 
    return c


def sample_noise(batch_size, n_noise, n_c_discrete, n_c_continuous, label=None, supervised=False):
    z = to_cuda(torch.randn(batch_size, n_noise))
    if supervised:
        c_discrete = to_cuda(to_onehot(label)) 
    else:
        c_discrete = to_cuda(to_onehot(torch.LongTensor(batch_size, 1).random_(0, n_c_discrete))) 
    c_continuous = to_cuda(torch.zeros(batch_size, n_c_continuous).uniform_(-1, 1)) 
    c = torch.cat((c_discrete.float(), c_continuous), 1)
    return z, c

def log_gaussian(c, mu, var):

    return -((c - mu)**2)/(2*var+1e-8) - 0.5*torch.log(2*np.pi*var+1e-8)


def get_sample_image(n_noise, n_c_continuous, G):
    """
        save sample 100 images
    """
    images = []
    # continuous code
    for cc_type in range(2):
        for num in range(10):
            fix_z = torch.randn(1, 62)
            z = to_cuda(fix_z)
            cc = -1
            for i in range(10):
                cc += 0.2
                c_discrete = to_cuda(to_onehot(num)) # (B,10)
                c_continuous = to_cuda(torch.zeros(1, n_c_continuous))
                c_continuous.data[:,cc_type].add_(cc)
                c = torch.cat((c_discrete.float(), c_continuous), 1)
                y_hat = G(z, c)
                line_img = torch.cat((line_img, y_hat.view(28, 28)), dim=1) if i > 0 else y_hat.view(28, 28)
            all_img = torch.cat((all_img, line_img), dim=0) if num > 0 else line_img
        img = all_img.cpu().data.numpy()
        images.append(img)
    # discrete code
    for num in range(10):
        c_discrete = to_cuda(to_onehot(num)) # (B,10)
        for i in range(10):
            z = to_cuda(torch.randn(1, n_noise))
            c_continuous = to_cuda(torch.zeros(1, n_c_continuous))
            c = torch.cat((c_discrete.float(), c_continuous), 1)
            y_hat = G(z, c)
            line_img = torch.cat((line_img, y_hat.view(28, 28)), dim=1) if i > 0 else y_hat.view(28, 28)
        all_img = torch.cat((all_img, line_img), dim=0) if num > 0 else line_img
    img = all_img.cpu().data.numpy()
    images.append(img)
    return images[0], images[1], images[2]



