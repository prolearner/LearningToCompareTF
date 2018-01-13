import os, errno
from scipy import misc
import numpy as np


class Dataset():
    """
    Contains the dataset to be used in meta-learning
    self._items contains tuples (filename,category, rotation).
    self.idx_classes contains a dictionary of class_name: class_index elements
    """
    def __init__(self, name,  items, idx_classes, parent):
        self.parent = parent
        self.name = name
        self.items = items
        self.idx_classes = idx_classes

    def __getitem__(self, item):
        return self.items[item]

    def n_classes(self):
        return len(self.idx_classes)

    def classes(self):
        """returns a list containing all the classes names"""
        return list(self.idx_classes.keys())

    def get_data(self, item):
        return self.parent.get_data(item)

    @staticmethod
    def union(d1, d2, u_name):
        if not isinstance(d1, Dataset) or not isinstance(d2, Dataset):
            raise TypeError('d1 and d2 must be both Datasets')
        if not d1.parent == d2.parent:
            raise ValueError('d1 and d2 must have the same parent')

        u_parent = d1.parent
        u_items = d1.items.copy()
        u_idx_classes = d1.idx_classes.copy()

        for class_d2, items_d2 in d2.items.items():
            if class_d2 not in u_idx_classes.keys():
                u_idx_classes[class_d2] = len(u_idx_classes)
                u_items[u_idx_classes[class_d2]] = items_d2
            else:
                for item in items_d2:
                    if item not in u_items[u_idx_classes[class_d2]]:
                        u_items[u_idx_classes[class_d2]].append(item)

        u_dataset = Dataset(u_name, '', lambda x: x, u_parent )
        u_dataset.items = u_items
        u_dataset.idx_classes = u_idx_classes

        return u_dataset


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
    - download: need to download the dataset
    - rotations: array of rotation [0, 1, 2, 3] contains rotations of 0, 90, 180, 270 degrees
    - split: value of training classes without considering rotations, if none folder splitting will be used
    '''

    def __init__(self, root, download=False, rotations=None, split=None):
        print('Loading Omniglot with rotations: {}, split: {}'.format(rotations, split))

        self.root = root

        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.split = split
        self.rotations = rotations
        if self.rotations is None:
            self.rotations = [0]

        train_item, train_idx, test_item, test_idx = self.find_items_and_split(os.path.join(self.root,
                                                                                                self.processed_folder))

        self.train = Dataset('train', train_item, train_idx, self)
        self.test = Dataset('test', test_item, test_idx, self)

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

    def find_items_and_split(self, root_dir):
        train_items = {}
        train_idx_classes = {}
        test_items = {}
        test_idx_classes = {}

        idx_classes = train_idx_classes
        items = train_items
        cur_split = (self.split if self.split is not None else 964) * len(self.rotations)
        for (root, dirs, files) in os.walk(root_dir):
            for f in files:
                if f.endswith("png"):
                    path_array = root.split('/')
                    class_name = path_array[-2] + "/" + path_array[-1]
                    if not self._check_class(class_name, idx_classes):
                        if len(idx_classes) > cur_split - 1:
                            idx_classes = test_idx_classes
                            items = test_items
                            cur_split = float('Inf')
                        self._add_class(idx_classes, items, class_name)

                    self._add_item(idx_classes, items, class_name, os.path.join(root, f))

        print("Classes Found: [%d, %d] ([train, test]) " % (len(train_idx_classes), len(test_idx_classes)))
        return train_items, train_idx_classes, test_items, test_idx_classes

    def _check_class(self, class_name, idx_classes):
        r_class_name = class_name + str(self.rotations[0])
        if r_class_name in idx_classes.keys():
            return True
        return False

    def _add_class(self, idx_classes, items, class_name):
        for r in self.rotations:
            r_class_name = class_name + str(r)
            idx_classes[r_class_name] = len(idx_classes)
            items[idx_classes[r_class_name]] = []

    def _add_item(self, idx_classes, items, class_name, item_path):
        for r in self.rotations:
            r_class_name = class_name + str(r)
            items[idx_classes[r_class_name]].append((item_path, idx_classes[r_class_name], r))

    @staticmethod
    def get_data(item):
        img = np.array(misc.imread(item[0]))
        img = np.rot90(img, int(item[2]))
        return img


if __name__ == '__main__':
    o = Omniglot(root='omniglot', download=True)