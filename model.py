import csv
import cv2
import numpy as np
import os
from random import shuffle, random
import sklearn

def CollectSampleFromCsv(folder_path):
    print("Collecting data samples from " + folder_path)
    samples = []
    with open(folder_path + "/driving_log.csv") as csvfile:
        for line in csv.reader(csvfile):
            center_steer_angle = float(line[3])
            left_steer_angle = center_steer_angle + 0.25
            right_steer_angle = center_steer_angle - 0.25
            steer_angles = [center_steer_angle, left_steer_angle, right_steer_angle]
            for i in range(3):
                img_name = line[i].split('IMG/')[-1]
                rel_img_path = folder_path + '/IMG/' + img_name
                steer_angle = steer_angles[i]
                sample = (rel_img_path, steer_angle)
                samples.append(sample)
    return samples

def CollectSamples(whitelist):
    all_samples = []
    datafolder = "./driving_data"
    for subfolder in os.listdir(datafolder):
        if '.' not in subfolder and subfolder in whitelist:
            subfolder_path = datafolder + "/" + subfolder
            all_samples.extend(CollectSampleFromCsv(subfolder_path))
    return all_samples

def generator(samples, batch_size=34):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steer_angles = []
            i = 0
            len_batch_sample = len(batch_samples)
            while len(images) < len_batch_sample:
                #for batch_sample in batch_samples:
                batch_sample = batch_samples[i%len_batch_sample]
                i = i+1
                steer_angle = batch_sample[1]

                #skip 50% of the when driving straight
                if abs(steer_angle < 0.1):
                    if np.random.randint(2):
                        continue

                name = batch_sample[0]
                img = cv2.imread(name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # randomily flipping 50% of the images taht are not driving straight
                if np.random.randint(2) and abs(steer_angle) > 0.1:
                    img = cv2.flip(img, 1)
                    steer_angle = -steer_angle

                images.append(img)
                steer_angles.append(steer_angle)

            X_train = np.array(images)
            y_train = np.array(steer_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

import cnn
model = cnn.nvidia()
model.summary()

model.compile(loss='mse', optimizer='adam')

# Training
data_folders = ['udacity_data', 'track1', 'track1_curves']
all_samples = CollectSamples(data_folders)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(all_samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=34)
validation_generator = generator(validation_samples, batch_size=34)
train_history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=8)

import pickle
with open( "train_history.p", "wb" ) as save:
    pickle.dump(train_history.history, save)

model.save('model.h5')
