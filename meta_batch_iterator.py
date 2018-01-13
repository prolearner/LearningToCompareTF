import numpy as np
import datasets
from skimage import transform
import tensorflow as tf


class MetaBatchIterator():
    def __init__(self, dataset, batch_size=100, classes_per_set=5, samples_per_class=1, query_per_cls=19):
        self.dataset = dataset

        self.batch_size = batch_size
        self.classes_per_set = classes_per_set  # n-way
        self.samples_per_class = samples_per_class  # k-shot
        self.samples_per_class_eval = query_per_cls  # for evaluation

        print('MBI: (total sets:%d, c-way:%d, k-shot:%d, n_query_per_cls:%d)' % (
            batch_size, classes_per_set, samples_per_class, self.samples_per_class_eval))

        self.support_set_batch = []  # input for support set
        self.target_batch = []  # query for support set

        self.n_samples = self.samples_per_class * self.classes_per_set  # num of samples per set
        self.n_samples_eval = self.samples_per_class_eval * self.classes_per_set  # number of samples per set for evaluation

        # Transformations to the image
        self.single_example_size = dataset.single_example_size

    def transform(self, img):
        img = img * (1. / 255) - 0.5
        img = transform.resize(img, self.single_example_size)
        return img

    def get_episode(self):
        """
        returns batch of episodes for meta-learning.
        """
        n_classes = self.dataset.n_classes()

        # select n classes_per_set randomly
        selected_classes = np.random.choice(n_classes, self.classes_per_set, False)  # no duplicate
        support_set = []
        target_set = []
        for c in selected_classes:
            selected_samples = np.random.choice(len(self.dataset[c]),
                                                # number of images in current class
                                                self.samples_per_class + self.samples_per_class_eval,
                                                False)  # select k + n_query per class
            idxs_train = np.array(selected_samples[:self.samples_per_class])  # idx for train episode
            idxs_test = np.array(selected_samples[self.samples_per_class:])  # idx for test episode

            # get all images filename for current train episode
            support_set.append(np.array(self.dataset[c])[idxs_train].tolist())

            # get all images filename for current test episode
            target_set += np.array(self.dataset[c])[idxs_test].tolist()

        return support_set, target_set

    def get_batch(self):
        support_sets = []
        target_sets = []
        for i in range(self.batch_size):
            support_set, target_set = self.get_episode()
            support_sets.append(support_set)
            target_sets.append(target_set)
        return support_sets, target_sets

    def get_inputs(self):
        """retur"""
        support_sets, target_sets = self.get_batch()

        #TODO: add shuffle?
        c_way = self.classes_per_set
        k_shot = self.samples_per_class
        n_query = self.samples_per_class_eval*c_way

        y = np.zeros((self.batch_size, n_query, c_way, k_shot, 1))
        x1 = np.zeros((self.batch_size, n_query, c_way,  k_shot, *self.single_example_size))
        x2 = np.zeros((self.batch_size, n_query, c_way, k_shot, *self.single_example_size))

        for b, (target_set, support_set) in enumerate(zip(target_sets, support_sets)):
            for t, t_item in enumerate(target_set):
                for c, c_item in enumerate(support_set):
                    for s, s_item in enumerate(c_item):
                        if s_item[1] == t_item[1]:
                            y[b, t, c, s] = 1
                        x1[b, t, c, s] = self.transform(self.dataset.get_data(t_item))
                        x2[b, t, c, s] = self.transform(self.dataset.get_data(s_item))

        return x1, x2, y

    def get_placeholders(self):

        y_p = tf.placeholder(tf.float32, [None, None, None, None, 1])
        x1_p = tf.placeholder(tf.float32, [None, None, None, None, *self.single_example_size])
        x2_p = tf.placeholder(tf.float32, [None, None, None, None, *self.single_example_size])

        return x1_p, x2_p, y_p


if __name__ == '__main__':
    import matplotlib.pyplot as pyplot

    input_size = 28

    omniglot = datasets.Omniglot(root='omniglot', download=True, rotations=[0, 1, 2, 3],
                                 split=1200, example_size=(input_size, input_size, 1))

    train_batch_iterator = MetaBatchIterator(omniglot.train, batch_size=1)

    for j in range(100):

        x1, x2, y = train_batch_iterator.get_inputs()
        x1_f = np.reshape(x1, (-1, input_size, input_size, 1))
        x2_f = np.reshape(x2, (-1, input_size, input_size, 1))
        y_f = np.reshape(y, (-1, 1))

        for i in range(x1_f.shape[0]):
            figsz = (12, 4)
            fig, ax = pyplot.subplots(1, 3, figsize=figsz, )

            ax[0].imshow(x1_f[i, :, :, 0], cmap='gray')
            ax[0].set_xlabel('target image')
            ax[1].imshow(x2_f[i, :, :, 0], cmap='gray')
            ax[1].set_xlabel('support image')
            ax[2].scatter(0, y_f[i], label='is_equal', )
            ax[2].set_xlabel('same class')
            pyplot.show()

            # print("{}".format(j*x1.shape[0] + i))






