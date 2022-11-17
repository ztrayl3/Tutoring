import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle

# Load our database of subjects
study = "emotion"
source = open(study + "_epochs.pkl", "rb")
epochs = pickle.load(source)
source.close()

# list our available videos, assign them random valence scores from -1 to 1
all_vids = {"video_element_v11": 1,
            "video_element_v12": -1,
            "video_element_v13": 0,
            "video_element_v14": 1,
            "video_element_v15": -1,
            "video_element_v16": 0,
            "video_element_v17": 1,
            "video_element_v18": -1,
            "video_element_v19": 0,
            "video_element_v20": 1,
            "video_element_v21": -1}

# sliding window to increase our sample size from 12 per class


# create x and y lists. For classifiers, X contains all the data and Y contains labels. Each epoch (x[0]) needs a label
X = epochs.get_data()
Y = np.zeros((X.shape[0],))
for i in range(X.shape[0]):  # for each epoch we have...
    label = all_vids[list(epochs[i].event_id.keys())[0]]  # reference our dictionary above
    Y[i] = label  # add label to the Y list

# make an SVM classifier with a linear kernel
classifier = SVC(C=1, kernel='linear')
# make our data 2D by collapsing all epochs into one
X_2d = X.reshape(len(X), -1)
X_2d = X_2d / np.std(X_2d)
# train on 80%, test on 20%
X_train, X_test, Y_train, Y_test = train_test_split(X_2d, Y, test_size=0.2, random_state=42)
clf = SVC(kernel='linear', C=1).fit(X_train, Y_train)
# print our average classification accuracy
print(clf.score(X_test, Y_test))


"""  Test with a bunch of predictions
for i in range(len(X_test)):
    sample = X_test.copy()[i].reshape(1, -1)
    print(clf.predict(sample))
"""