# LearningToCompareTF
implementation of the  [learning to compare article](https://arxiv.org/abs/1711.06025) using tensorflow

To start the omniglot experiment run
```
omniglot_test.py
```

omniglot dataset will be downloaded in the ```omniglot``` while results will be stored in pickle files in the
```results/omniglot/``` directory.

check the bottom of the file to see and modify the arguments that you can pass

## Results
Learning curves for a model similar to NaiveRN for the omniglot dataset augmented as in the paper:

![accuracy and loss on the test set over training episodes](https://github.com/prolearner/LearningToCompareTF/blob/master/plots/omniglot_plots.png)
