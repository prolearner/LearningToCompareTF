from utils import load_obj
import os
import numpy as np
from matplotlib import pyplot

RESULTSROOT = 'results/omniglot/'

def plot_results(results_filename, root):
    results_filepath = os.path.join(root, results_filename)
    results = load_obj(results_filepath)

    print(results)

    FLAGS = results['flags']
    batch_size = FLAGS.batch_size

    episodes = np.array([s[0]*batch_size/1000 for s in results['step_time']])
    time = np.array([s[1] for s in results['step_time']])
    test_accuracy = np.array([s[0] for s in results['test_accuracy']])
    test_accuracy_std = np.array([s[1] for s in results['test_accuracy']])

    test_loss = np.array([s[0] for s in results['test_loss']])

    train_accuracy = results['train_accuracy']


    figsz = (12, 4)
    fig, ax = pyplot.subplots(1, 2, figsize=figsz,)
    fig.suptitle('LearningToCompare Omniglot (NaiveRN)')


    cur_ax = ax[0]
    cur_ax.plot(episodes, test_accuracy, label='test accuracy (mean over 1000 ep)')
    #cur_ax.plot(episodes, test_loss, label='test loss (mean over 1000 ep)')
    # cur_ax.fill_between(episodes, test_accuracy - test_accuracy_std, test_accuracy + test_accuracy_std, alpha=0.2)
    # cur_ax.plot(episodes, train_accuracy, label='train accuracy')
    cur_ax.set_ylim([0.85, 1])
    cur_ax.set_xlim([0, episodes[-1]])

    cur_ax.set_xlabel('episodes / 1000')
    cur_ax.legend(loc=4)

    cur_ax = ax[1]
    #cur_ax.plot(episodes, test_accuracy, label='test accuracy (over 1000 ep)')
    cur_ax.plot(episodes, test_loss, label='test loss (mean over 1000 ep)')
    # cur_ax.fill_between(episodes, test_accuracy - test_accuracy_std, test_accuracy + test_accuracy_std, alpha=0.2)
    # cur_ax.plot(episodes, train_accuracy, label='train accuracy')
    # cur_ax.set_ylim([0.0, 1])
    cur_ax.set_xlim([0, episodes[-1]])

    cur_ax.set_xlabel('episodes / 1000')
    cur_ax.legend(loc=2)

    pyplot.show()



plot_results('r0_omniglot_similar2.pickle', RESULTSROOT)


