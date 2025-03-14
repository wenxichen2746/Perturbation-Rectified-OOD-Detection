import os

path = './data/images_classic/cifar100c'
save_path = './data/benchmark_imglist/cifar100/test_cifar100c.txt'
prefix = 'cifar100c/'
files = os.listdir(path)
with open(save_path, 'a') as f:
    for file in files:
        splits = file.split('_')
        label = (splits[1].split('.'))[0]
        line = prefix + file + ' ' + label + '\n'
        f.write(line)
    f.close()