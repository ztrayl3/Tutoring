import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
import pickle

# #############################################################################
# # Set parameters and read data

# Load our database of subjects
study = "emotion"
source = open(study + "_epochs.pkl", "rb")
epochs = pickle.load(source)
source.close()

# list our available videos
all_vids = ["video_element_v11",
            "video_element_v12",
            "video_element_v13",
            "video_element_v14",
            "video_element_v15",
            "video_element_v16",
            "video_element_v17",
            "video_element_v18",
            "video_element_v19",
            "video_element_v20",
            "video_element_v21"]

# Choose any two videos (by index) to compare with the CSP/LDA combination (0-10)
epochs = epochs[all_vids[0], all_vids[1]]  # choose two for our 2 class LDA classifier
epochs_train = epochs[0:4]  # split half-and-half
epochs_test = epochs[4:8]  # note: this should really be done randomly, I split manually here for simplicity

labels = epochs_train.events[:, -1] - 15  # get labels into 0 or 1 format

# Define a monte-carlo cross-validation generator (reduce variance):
scores = []
epochs_data = epochs_test.get_data()
epochs_data_train = epochs_train.get_data()
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)

csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
