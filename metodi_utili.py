import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import numpy as np
import itertools
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras import layers
from keras import activations

#Method used to plot the histogram of the target features
def histogram(x, title):
    legend = ['Attack distribution']
    n_bins = 20
    # Creating histogram
    fig, axs = plt.subplots(1, 1,
                            figsize=(10, 7),
                            tight_layout=True)

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        axs.spines[s].set_visible(False)

    # Remove x, y ticks
    axs.xaxis.set_ticks_position('none')
    axs.yaxis.set_ticks_position('none')

    plt.xticks(rotation=30, ha='right')

    # Add padding between axes and labels
    axs.xaxis.set_tick_params(pad=5)
    axs.yaxis.set_tick_params(pad=10)

    # Add x, y gridlines
    axs.grid(b=True, color='grey',
             linestyle='-.', linewidth=0.5,
             alpha=0.6)


    # Creating histogram
    N, bins, patches = axs.hist(x, bins=n_bins)

    # Setting color
    fracs = ((N ** (1 / 2)) / N.max())
    norm = colors.Normalize(fracs.min(), fracs.max())

    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    # Adding extra features
    plt.xlabel("Attack")
    plt.ylabel("Count")
    plt.legend(legend)
    plt.title(title)

    # Show plot
    plt.show()


def LSTM_model(data_X):
    #batch_size = 32
    N_features = len(data_X.columns)
    model = keras.Sequential()

    model.add(layers.Conv1D(64, kernel_size=64,
                            input_shape=(N_features, 1), padding='same', activation='relu'))

    model.add(layers.MaxPool1D(pool_size=10))
    model.add(layers.BatchNormalization())
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=False)))

    model.add(layers.Reshape((128, 1), input_shape=(128,)))

    model.add(layers.MaxPool1D(pool_size=5))
    model.add(layers.BatchNormalization())
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=False)))

    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(10))
    model.add(layers.Activation(activations.softmax))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def LSTM_application(model, kfold, ros, data, data_X, y_train):
    '''
    #####################################################################
    Resamplig with RandomOverSampler class, splitting the dataset in Train and test
    using k-fold, and applying the model to the splitted dataset k times.

    After that, the accuracy can be computed
    #####################################################################
    '''

    oos_pred = []
    for train_index, test_index in kfold.split(data_X, y_train):
        train_X, test_X = data_X.iloc[train_index], data_X.iloc[test_index]
        train_y, test_y = y_train.iloc[train_index], y_train.iloc[test_index]

        print("train index:", train_index)
        print("test index:", test_index)
        print(train_y.value_counts())

        train_X_over, train_y_over = ros.fit_resample(train_X, train_y)
        print(train_y_over.value_counts())

        x_columns_train = data.columns.drop('Class')
        x_train_array = train_X_over[x_columns_train].values
        x_train_1 = np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))

        dummies = pd.get_dummies(train_y_over)  # Classification
        #outcomes = dummies.columns
        #num_classes = len(outcomes)
        y_train_1 = dummies.values

        x_columns_test = data.columns.drop('Class')
        x_test_array = test_X[x_columns_test].values
        x_test_2 = np.reshape(x_test_array, (x_test_array.shape[0], x_test_array.shape[1], 1))

        dummies_test = pd.get_dummies(test_y)  # Classification
        #outcomes_test = dummies_test.columns
        #num_classes = len(outcomes_test)
        y_test_2 = dummies_test.values

        model.fit(x_train_1, y_train_1, validation_data=(x_test_2, y_test_2), epochs=1)  # MOD: was 9 epochs

        pred = model.predict(x_test_2)
        pred = np.argmax(pred, axis=1)
        y_eval = np.argmax(y_test_2, axis=1)
        score = metrics.accuracy_score(y_eval, pred)
        oos_pred.append(score)
        print("Validation score: {}".format(score))

    return oos_pred,y_eval,pred

def RandomForest_Model(clf,kfold, ros, data, data_X, y_train):
    oos_pred=[]
    for train_index, test_index in kfold.split(data_X, y_train):
        # Create a Gaussian Classifier


        train_X, test_X = data_X.iloc[train_index], data_X.iloc[test_index]
        train_y, test_y = y_train.iloc[train_index], y_train.iloc[test_index]

        print("train index:", train_index)
        print("test index:", test_index)
        print(train_y.value_counts())

        train_X_over, train_y_over = ros.fit_resample(train_X, train_y)
        print(train_y_over.value_counts())

        x_columns_train = data.columns.drop('Class')
        x_train_array = train_X_over[x_columns_train].values

        dummies = pd.get_dummies(train_y_over)  # Classification
        y_train_1 = dummies.values

        x_columns_test = data.columns.drop('Class')
        x_test_array = test_X[x_columns_test].values

        dummies_test = pd.get_dummies(test_y)  # Classification
        y_test_2 = dummies_test.values

        # Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(x_train_array, y_train_1)

        pred = clf.predict(x_test_array)
        pred = np.argmax(pred, axis=1)
        y_eval = np.argmax(y_test_2, axis=1)
        score = metrics.accuracy_score(y_eval, pred)

        oos_pred.append(score)
        print("Validation score: {}".format(score))

    return oos_pred,y_eval, pred


#Method used for the plot of confusion matrix
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()


def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()


def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows


def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns

'''
##############################################################
Generating the confusion matrix
##############################################################
'''
def show_confusion_matrix(y_eval,pred, title):
    confussion_matrix = confusion_matrix(y_eval, pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plot_confusion_matrix(cm=confussion_matrix,
                                       normalize=False,
                                       target_names=['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic',
                                                     'Normal', 'Reconnaissance', 'Shellcode', 'Worms'],
                                       title=title)

    print("\nLabel Precision Recall")
    for label in range(10):
        print(f"{label:5d} {precision(label, confussion_matrix):9.3f} {recall(label, confussion_matrix):6.3f}")