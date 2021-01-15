import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import io as sio


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

    for action in actions:
        for subject in subjects:
            for experience in experiences:
                for capteur in capteur_lst:
                    vector = df.loc[(df['action'] == action) &
                                    (df['subject'] == subject) &
                                    (df['experience'] == experience),
                                    [capteur]]

                    capteur_features[f"mean_{capteur}"] = float(vector.mean())
                    capteur_features[f"std_{capteur}"] = float(vector.std())

                actions_features[(subject, experience, action)] = capteur_features
                capteur_features = {}

    feature_df = pd.DataFrame.from_dict(actions_features).dropna(axis=1)
    return feature_df


data = load_data(data_path)
tracer_signal(data, 1, 1, 1, 3)
tracer_signal(data, 2, 1, 1, 3)
features = features_extraction(data, data.columns[0:6])
