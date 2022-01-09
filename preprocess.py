import pandas as pd
import numpy as np
import pathlib
import os
from shutil import copyfile
import csv


NEW_FOLDER = "data/"

TRAIN_FOLDER = "data/train/"
TEST_FOLDER = "data/test/"


data_dir = pathlib.Path(NEW_FOLDER)
if not data_dir.exists():
	os.mkdir(NEW_FOLDER)
	
data_dir = pathlib.Path(TRAIN_FOLDER)
if not data_dir.exists():
	os.mkdir(TRAIN_FOLDER)
	
data_dir = pathlib.Path(TEST_FOLDER)
if not data_dir.exists():
	os.mkdir(TEST_FOLDER)

PATH = 'cv-corpus-7.0-2021-07-21/ru/'

TRAIN_FILE = 'train.tsv'
TEST_FILE = 'test.tsv'

TRAIN_SIZE = 2000
TEST_SIZE = 600

a = pd.read_csv(PATH + TRAIN_FILE, sep='\t')

male = a[['path', 'gender']]
male = male[male['gender'] == 'male']

female = a[['path', 'gender']]
female = female[female['gender'] == 'female']

rng = np.random.default_rng()
male_train = rng.choice(np.array(male), size=TRAIN_SIZE, axis=0)
female_train = rng.choice(np.array(female), size=TRAIN_SIZE, axis=0)
train = np.concatenate((male_train, female_train), axis=0)


b = pd.read_csv(PATH + TEST_FILE, sep='\t')

male = b[['path', 'gender']]
male = male[male['gender'] == 'male']

female = b[['path', 'gender']]
female = female[female['gender'] == 'female']

rng = np.random.default_rng()
male_test = rng.choice(np.array(male), size=TEST_SIZE, axis=0)
female_test = rng.choice(np.array(female), size=TEST_SIZE, axis=0)
test = np.concatenate((male_test, female_test), axis=0)

PREFIX = 'common_voice_ru_'

for i in train:
	i[0] = i[0][len(PREFIX):]

for i in test:
	i[0] = i[0][len(PREFIX):]

myFile = open(NEW_FOLDER + 'train.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(train)

myFile = open(NEW_FOLDER + 'test.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(test)



for i in train:
	copyfile(PATH + 'clips/' + PREFIX + i[0], TRAIN_FOLDER + i[0])

for i in test:
	copyfile(PATH + 'clips/' + PREFIX + i[0], TEST_FOLDER + i[0])
