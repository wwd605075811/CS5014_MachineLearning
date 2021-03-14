import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.learning_curve import learning_curve  fix learning_curve DeprecationWarning
from sklearn.model_selection import learning_curve

# delete the line contains "?"
def delete_line(filename):
    x = []
    i = 0
    file = open(filename)
    for line in file.readlines():
        line = line.strip().split(",")
        if line.__contains__('?'):
            x.append(i - 1)
        i = i + 1
    file.close()
    return x;

def regularized_data(new_df):
    dummies_A13 = pd.get_dummies(new_df['A13'], prefix='A13')
    dummies_A12 = pd.get_dummies(new_df['A12'], prefix='A12')
    dummies_A10 = pd.get_dummies(new_df['A10'], prefix='A10')
    dummies_A9 = pd.get_dummies(new_df['A9'], prefix='A9')
    dummies_A7 = pd.get_dummies(new_df['A7'], prefix='A7')
    dummies_A6 = pd.get_dummies(new_df['A6'], prefix='A6')
    dummies_A5 = pd.get_dummies(new_df['A5'], prefix='A5')
    dummies_A4 = pd.get_dummies(new_df['A4'], prefix='A4')
    dummies_A1 = pd.get_dummies(new_df['A1'], prefix='A1')

    df = pd.concat(
        [new_df, dummies_A1, dummies_A4, dummies_A5, dummies_A6, dummies_A7, dummies_A9, dummies_A10, dummies_A12,
         dummies_A13], axis=1)
    df.drop(['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13'], axis=1, inplace=True)
    return df


# Use 'learning_curve' in sklearn to get training_score and cv_score
# Then use matplotlib to draw the 'learning curve'
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    Draw the learning curve of data on a model.
    Args:
    ----------
    estimator : Logistic regression classifier
    title : the title of this table
    X : feature，which type is numpy
    y : target vector, which type is numpy
    ylim : tuple(ymin, ymax), Set the lowest point and highest point of the ordinate in the image
    cv : When doing cross-validation, the data is divided into the number of copies, one of which
         is used as the cv set, and the remaining n-1 copies are used as training (default is 3 copies)
    n_jobs : Number of parallel tasks(default = 1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("train_set_number")
        plt.ylabel("score")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="training_socre")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="cv_socre")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff