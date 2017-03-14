from os.path import isdir, isfile
import tarfile
from urllib.request import urlretrieve

cifar10_dataset_download_path = \
    'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
cifar10_dataset_folder_path = 'cifar-10-batches-py'
cifar10_dataset_zip_name = 'cifar-10-python.tar.gz'

if not isdir(cifar10_dataset_folder_path):
    urlretrieve(cifar10_dataset_download_path, cifar10_dataset_zip_name)

    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()
