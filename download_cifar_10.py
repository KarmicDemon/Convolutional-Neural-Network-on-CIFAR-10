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
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar)
        tar.close()
