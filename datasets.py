import os, errno
from scipy import misc
import numpy as np


class Dataset():
    """
    Contains the dataset to be used in meta-learning
    self.all_items contains tuples (filename,category). 
    self.idx_classes contains a dictionary of class_name: class_index elements
    """
    def __init__(self, name,  items_path, find_items, parent):
        self.parent = parent
        self.name = name
        self.items, self.idx_classes = find_items(items_path)

    def __getitem__(self, item):
        return self.items[item]

    def n_classes(self):
        return len(self.idx_classes)

    def classes(self):
        """returns a list containing all the classes names"""
        return list(self.idx_classes.keys())

    def get_data(self, image_path):
        return self.parent.get_data(image_path)



class Omniglot():
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self, root, download=False, rotations=False):
        self.root = root

        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.train = Dataset('train', os.path.join(self.root, self.processed_folder, 'images_background'), self.find_items, self)
        self.test = Dataset('test', os.path.join(self.root, self.processed_folder, 'images_evaluation'), self.find_items, self)

    def download(self):
        """
        download files from url into raw folder and then decompress into processed folder.
        :return:
        """
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('>>Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.processed_folder)
            print(">>Unzip from " + file_path + " to " + file_processed)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("<<Download finished.")

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, "images_background"))

    @staticmethod
    def find_items(root_dir):
        idx_classes = {}
        items = {}

        for (root, dirs, files) in os.walk(root_dir):
            for f in files:
                if f.endswith("png"):
                    path_array = root.split('/')
                    class_name = path_array[-2] + "/" + path_array[-1]
                    if class_name not in idx_classes.keys():
                        idx_classes[class_name] = len(idx_classes)
                        items[idx_classes[class_name]] = []

                    items[idx_classes[class_name]].append((os.path.join(root, f), idx_classes[class_name]))

        print("== Found %d items in %d classes " % (len(items), len(idx_classes)))
        return items, idx_classes

    @staticmethod
    def get_data(image_path):
        img = np.array(misc.imread(image_path))
        return img




if __name__ == '__main__':
    o = Omniglot(root='omniglot', download=True)