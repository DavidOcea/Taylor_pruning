import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import io
import cv2
from PIL import Image
import torchvision.transforms as transforms
try:
    import mc
except ImportError:
    pass

import pdb

def bin_loader(path):
    '''load verification img array and label from bin file
    '''
    with open(path, 'rb') as f:
        if sys.version_info[0] == 2:
            data = pickle.load(open(path, 'rb'))
        elif sys.version_info[0] == 3:
            data = pickle.load(open(path, 'rb'), encoding='bytes')
        else:
            raise EnvironmentError('Only support python 2 or 3')
    bins, lbs = data
    assert len(bins) == 2*len(lbs)
    imgs = [pil_loader(b) for b in bins]
    return imgs, lbs

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img

def cv2_loader(path):
    img =cv2.imread(path)
    return img

class GivenSizeSampler(Sampler):
    '''
    Sampler with given total size
    '''
    def __init__(self, dataset, total_size=None, rand_seed=None, sequential=False, silent=False):
        self.rand_seed = rand_seed if rand_seed is not None else 0
        self.dataset = dataset
        self.epoch = 0
        self.sequential = sequential
        self.silent = silent
        self.total_size = total_size if total_size is not None else len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        if not self.sequential:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.rand_seed)
            origin_indices = list(torch.randperm(len(self.dataset), generator=g))
        else:
            origin_indices = list(range(len(self.dataset)))
        indices = origin_indices[:]

        # add extra samples to meet self.total_size
        extra = self.total_size - len(origin_indices)
        if not self.silent:
            print('Origin Size: {}\tAligned Size: {}'.format(len(origin_indices), self.total_size))
        if extra < 0:
            indices = indices[:self.total_size]
        while extra > 0:
            intake = min(len(origin_indices), extra)
            indices += origin_indices[:intake]
            extra -= intake
        assert len(indices) == self.total_size, "{} vs {}".format(len(indices), self.total_size)

        return iter(indices)

    def __len__(self):
        return self.total_size

    def set_epoch(self, epoch):
        self.epoch = epoch


class BinDataset(Dataset):
    def __init__(self, bin_file, transform=None):
        self.img_lst, _ = bin_loader(bin_file)
        self.num = len(self.img_lst)
        self.transform = transform

    def __len__(self):
        return self.num

    def _read(self, idx=None):
        if idx == None:
            idx = np.random.randint(self.num)
        try:
            img = self.img_lst[idx]
            return img
        except Exception as err:
            print('Read image[{}] failed ({})'.format(idx, err))
            return self._read()

    def __getitem__(self, idx):
        img = self._read(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img


def build_labeled_dataset(filelist, prefix):
    img_lst = []
    lb_lst = []
    with open(filelist) as f:
        for x in f.readlines():
            n = x.split(' ')[0]
            lb = x.split(' ')[1]
            try:
                lb = int(lb)
            except ValueError:
                import pdb
                pdb.set_trace()
            img_lst.append('{}/{}'.format(prefix,n))
            lb_lst.append(lb)
    assert len(img_lst) == len(lb_lst)
    return img_lst, lb_lst

def build_unlabeled_dataset(filelist, prefix):
    img_lst = []
    with open(filelist) as f:
        for x in f.readlines():
            img_lst.append(os.path.join(prefix, x.strip().split(' ')[0]))
    return img_lst


class FileListLabeledDataset(Dataset):
    def __init__(self, filelist, prefix, transform=None, memcached=False, memcached_client=''):
        self.img_lst, self.lb_lst = build_labeled_dataset(filelist, prefix)
        self.num = len(self.img_lst)
        self.transform = transform
        self.num_class = max(self.lb_lst) + 1
        self.initialized = False
        self.memcached = memcached
        self.memcached_client = memcached_client

    def __len__(self):
        return self.num

    def __init_memcached(self):
        if not self.initialized:
            server_list_config_file = "{}/server_list.conf".format(self.memcached_client)
            client_config_file = "{}/client.conf".format(self.memcached_client)
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _read(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num)
        fn = self.img_lst[idx]
        lb = self.lb_lst[idx]
        try:
            if self.memcached:
                value = mc.pyvector()
                self.mclient.Get(fn, value)
                value_str = mc.ConvertBuffer(value)
                img = cv2_loader(value_str)
            else:
                img = cv2_loader(fn)
            return img, lb
        except Exception as err:
            print('Read image[{}, {}] failed ({})'.format(idx, fn, err))
            return self._read()

    def __getitem__(self, idx):
        if self.memcached:
            self.__init_memcached()
        img, lb = self._read(idx)
        if self.transform is not None:
            try:
                img = self.transform(img)
            except AttributeError:
                import pdb
                pdb.set_trace()
                print('idx:',idx)
        return img, lb

class FileListDataset(Dataset):
    def __init__(self, filelist, prefix, transform=None, memcached=False, memcached_client=''):
        self.img_lst = build_unlabeled_dataset(filelist, prefix)
        self.num = len(self.img_lst)
        self.transform = transform
        self.initialized = False
        self.memcached = memcached
        self.memcached_client = memcached_client

    def __len__(self):
        return self.num

    def __init_memcached(self):
        if not self.initialized:
            server_list_config_file = "{}/server_list.conf".format(self.memcached_client)
            client_config_file = "{}/client.conf".format(self.memcached_client)
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _read(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num)
        fn = self.img_lst[idx]
        try:
            #img = pil_loader(open(fn, 'rb').read())
            if self.memcached:
                value = mc.pyvector()
                self.mclient.Get(fn, value)
                value_str = mc.ConvertBuffer(value)
                img = pil_loader(value_str)
            else:
                img = pil_loader(open(fn, 'rb').read())
            return img
        except Exception as err:
            print('Read image[{}, {}] failed ({})'.format(idx, fn, err))
            return self._read()

    def __getitem__(self, idx):
        if self.memcached:
            self.__init_memcached()
        img = self._read(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img


#-----------test----
def iter_f(train_loader):
    import os
    import time
    start = time.time()
    for i, (input, target) in enumerate(train_loader):
        print(str(i) + ' in ' + str(os.getpid()))
        end = time.time()
        print("time last = ", end - start)
        start = end    
    
def main():
    class imshowCollate(object):

        def __init__(self):
            pass

        def __call__(self, batch):
            images, labels = zip(*batch)
            idx = 0
            for img in images:
                img = img.cpu().numpy().transpose((1, 2, 0))*255   #totensor
                cv2.imwrite('datatest/sev_img/img' + str(idx) + '——' +str(labels[idx]) +'.jpg', img)
                # print(img.shape)
                idx += 1
            return images, labels

    from transforms import  Compose, Normalize, RandomResizedCrop, RandomHorizontalFlip, \
        ColorJitter, ToTensor,Lighting

    batch_size = 16
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

    dataset = FileListLabeledDataset(
			'/workspace/mnt/group/algo/yangdecheng/work/multi_task/pytorch-train/datatest/test.txt','/workspace/mnt/group/algo/yangdecheng/work/multi_task/pytorch-train/datatest/pic',
			Compose([
				RandomResizedCrop((112), scale=(0.7, 1.2), ratio=(1. / 1., 4. / 1.)),
				RandomHorizontalFlip(),
				ColorJitter(brightness=[0.5,1.5], contrast=[0.5,1.5], saturation=[0.5,1.5], hue= 0),
				ToTensor(),
				Lighting(1, [0.2175, 0.0188, 0.0045], [[-0.5675,  0.7192,  0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948,  0.4203]]), #0.1
# 				normalize,
			]))

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True,
        num_workers=10, pin_memory=True, sampler=None,
               collate_fn= imshowCollate()
    )

    from multiprocessing import Process
    p_list =[]
    for i in range(1):
        p_list.append(Process(target=iter_f, args=(train_loader,)))
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()

if __name__ == '__main__':
    main()
