import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import io as sio
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

data_path = 'C:/Users/Theo/PycharmProjects/AADA/Projet_IMU/data'


def load_data(path):

    list_files = os.listdir(path)[1:]
    col_names = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'subject', 'experience', 'action']

    data_array = np.empty(shape=(1, 9))
    for name_file in list_files:

        name_variable = 'd_iner'
        mat_contents = sio.loadmat(os.path.join(path, name_file))
        mat_data = mat_contents[name_variable]

        action = np.full((mat_data.shape[0], 1), int(name_file.split('_')[0][1:]))
        subject = np.full((mat_data.shape[0], 1), int(name_file.split('_')[1][1:]))
        experience = np.full((mat_data.shape[0], 1), int(name_file.split('_')[2][1:]))

        tmp_array = np.concatenate([mat_data, subject, experience, action], axis=1)
        data_array = np.concatenate((data_array, tmp_array), axis=0)

    # deletes the first column
    data_array = np.delete(data_array, 0, 0)
    df = pd.DataFrame(data_array,  columns=col_names)

    return df


def tracer_signal(df, capteur, action, sujet, experience):
    capteur_list = ['acc', 'gyr']
    capteur = capteur - 1

    signal_x = df.loc[(df["subject"] == sujet) &
                      (df["experience"] == experience) &
                      (df["action"] == action),
                      f"{capteur_list[capteur]}_x"].values

    signal_y = df.loc[(df["subject"] == sujet) &
                      (df["experience"] == experience) &
                      (df["action"] == action),
                      f"{capteur_list[capteur]}_y"].values

    signal_z = df.loc[(df["subject"] == sujet) &
                      (df["experience"] == experience) &
                      (df["action"] == action),
                      f"{capteur_list[capteur]}_z"].values

    nb_point = len(signal_x)
    time_axis = [x * 0.02 for x in range(0, nb_point)]

    plt.title(f'capteur: {capteur_list[capteur]}, action: {str(action)}, '
              f'sujet: {str(sujet)}, essai: {str(experience)}')
    plt.plot(time_axis, signal_x, 'r')
    plt.plot(time_axis, signal_y, 'g')
    plt.plot(time_axis, signal_z, 'b')
    plt.xlabel('Time (s)')
    plt.show()


def features_extraction(df, capteur_lst):
    actions_features = {}
    capteur_features = {}

    actions = [i for i in range(1, 28)]
    subjects = [i for i in range(1, 9)]
    experiences = [i for i in range(1, 5)]

    for subject in subjects:
        for experience in experiences:
            for action in actions:
                for capteur in capteur_lst:
                    vector = df.loc[(df['action'] == action) &
                                    (df['subject'] == subject) &
                                    (df['experience'] == experience),
                                    [capteur]]

                    capteur_features[f"mean_{capteur}"] = float(vector.mean())
                    capteur_features[f"std_{capteur}"] = float(vector.std())
                    capteur_features[f"median_{capteur}"] = float(vector.median())
                    capteur_features["action"] = action

                actions_features[(subject, experience,  action)] = capteur_features
                capteur_features = {}

    feature_df = pd.DataFrame.from_dict(actions_features).dropna(axis=1)
    return feature_df


def train_test_split(df, subject_train, subject_test):

    normalized_df = {}
    features_name = list(df.columns)
    features_name.remove('action')

    for name, col in df[features_name].loc[subject_train].items():
        mean = col.mean()
        std = col.std()

        vector = np.array((df[name] - mean) / std)

        normalized_df[name] = vector

    normalized_df['action'] = df['action']
    normalized_df = pd.DataFrame.from_dict(normalized_df)

    train_df = normalized_df.loc[subject_train]
    test_df = normalized_df.loc[subject_test]

    return train_df, test_df


def shuffle(df):
    return df.sample(frac=1)


def train_classifier(classifiers, df_train, df_test):

    scores = {}

    features_name = list(df_train.columns)
    features_name.remove('action')

    # TODO : use gridsearch instead
    for clf in classifiers:

        clf_name = str(clf).split('(')[0]
        clf.fit(df_train[features_name], df_train['action'])
        pred = clf.predict(df_test[features_name])

        scores[clf_name] = classification_report(df_test['action'], pred)
        print(clf_name)
    return scores


data = load_data(data_path)
tracer_signal(data, 1, 1, 1, 3)
tracer_signal(data, 2, 1, 1, 3)

features = features_extraction(data, data.columns[0:6])

train_subject_lst = [1, 3, 5, 7]
test_subject_lst = [2, 4, 6, 8]
train, test = train_test_split(features.transpose(), train_subject_lst, test_subject_lst)

train = shuffle(train)
test = shuffle(test)

clf_lst = [DecisionTreeClassifier(),
           KNeighborsClassifier(n_neighbors=5),
           MLPClassifier(hidden_layer_sizes=(150, ), max_iter=1000,
                         alpha=1e-4, random_state=1),
           SVC(),
           RandomForestClassifier()]

scores = train_classifier(clf_lst, train, test)

for name, score in scores.items():
    print(name, score)