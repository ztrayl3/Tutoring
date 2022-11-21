from sklearn.model_selection import cross_val_score, KFold
from pyriemann.estimation import Covariances
from pyriemann.classification import TSclassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib
import mne


def sliding_window(window_size, step_size, epochs, end):
    # raw data is in format (epochs, channels, timepoints), ex: (13, 19, 15101)
    step = window_size * step_size  # step size, as a portion of the window size
    global sfreq  # epoch's sampling rate
    start = 0

    result = []
    for epoch in range(len(epochs)):
        while start < end:
            s = int(start / (1 / sfreq))  # convert seconds to samples
            e = int((start + window_size) / (1 / sfreq))  # convert seconds to samples
            temp = epochs[epoch, :, s:e]  # grab samples from s to e
            result.append(temp)

            start = start + step  # slide window over
    return np.stack(result[:-1])  # drop the last because it will rarely be the right size


# read raw EEG file
files = ["../Data/emotion/raw_exp1.edf", "../Data/emotion/raw_exp2.edf", "../Data/emotion/raw_exp3.edf",
         "../Data/emotion/raw_exp4.edf", "../Data/emotion/raw_exp5.edf", "../Data/emotion/raw_exp6.edf"]
raws = []
output = "openvibe"  # either "sklearn" or "openvibe" to detail which classifier format we're using

for eeg_path in files:
    raw = mne.io.read_raw_edf(eeg_path, preload=True)
    montage = mne.channels.make_standard_montage('standard_1020')  # load the standard 10-20
    raw.set_montage(montage, on_missing="ignore")  # apply the channel montage
    raws.append(raw)

# combine all raw files into one large one
all_raws = mne.concatenate_raws(raws)
events, event_dict = mne.events_from_annotations(all_raws)
sfreq = all_raws.info["sfreq"]

# Pre-Processing
artifact_removal = all_raws.copy()
artifact_removal.filter(l_freq=1.0, h_freq=None, n_jobs=-1)  # high-pass filter at 1Hz

# ICA artifact removal
ica = mne.preprocessing.ICA(n_components=0.95, random_state=97, max_iter="auto")
ica.fit(artifact_removal)  # fit the ICA with EEG and EOG information

# Visually inspect the data
N = ica.n_components_
ica.plot_properties(all_raws, picks=list(range(0, N)), psd_args={"fmin": 1.0, "fmax": 80.0})  # further analyze the channels
matplotlib.pyplot.show(block=True)  # wait until all figures are closed

# last chance to un-bad components...
response = input("Type any bad components (0-{}) that should be marked for exclusion (seperated by spaces): ".format(N))
ica.exclude = [int(x) for x in response.split(" ")]  # mark bad components for removal

# See the changes we've made
ica.plot_overlay(all_raws, exclude=ica.exclude, picks='eeg')
matplotlib.pyplot.show(block=True)  # wait until all figures are closed

ica.apply(all_raws)  # apply ICA to data, removing the artifacts

if output == "sklearn":
    tmax = 30
    epochs = mne.Epochs(all_raws, events, event_id=event_dict, tmin=-0.2, tmax=tmax,
                        preload=False, event_repeated="drop", reject=None, reject_by_annotation=None, proj=False)

    relevant = dict(valence=["OVTK_StimulationId_Label_00", "OVTK_StimulationId_Label_01"],
                    arousal=["OVTK_StimulationId_Label_10", "OVTK_StimulationId_Label_11"])
    feature = "raw"  # feature can either be "raw", "covariance", or "frequency"
    for emotion in relevant:  # either valence or arousal
        print("Classifying {}".format(emotion))
        data = []
        labels = []
        for ID in relevant[emotion]:  # for high/low
            one_class = sliding_window(1, 1/2, epochs[ID].get_data(), tmax)  # slide window to augment sample size
            data.append(one_class)  # add one class worth of data to pile
            labels.append([ID[-1:]] * len(one_class))  # add labels (either 0 or 1) for each sample
        data = np.concatenate((data[0], data[1]))
        labels = np.array(sum(labels, []))  # join the data and labels each into one object

        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        if feature == "raw":
            data_2d = data.reshape(len(data), -1)
            data_2d = data_2d / np.std(data_2d)
            clf = SVC(C=1, kernel='linear')
            scores = cross_val_score(clf, data_2d, labels, cv=cv, n_jobs=-1)
        elif feature == "covariance":
            cov_data_train = Covariances('oas').transform(data)  # NOTE: covariances needs regularization b/c of ICA
            clf = TSclassifier()
            scores = cross_val_score(clf, cov_data_train, labels, cv=cv, n_jobs=-1)
        elif feature == "frequency":
            data = []
            labels = []
            for ID in relevant[emotion]:  # for high/low
                eeg = epochs[ID]
                size = 1
                step = 0.5
                length = eeg.last / sfreq  # the latest (in seconds) timepoint
                start = 0
                stop = start + size
                temp = []

                while stop < length:
                    kwargs = dict(fmin=8, fmax=80,
                                  tmin=start, tmax=stop,
                                  n_jobs=-1)
                    psds, freqs = eeg.compute_psd(**kwargs).get_data(return_freqs=True)

                    # Convert power to dB scale.
                    psds = 10 * np.log10(psds)

                    # Save the powers
                    temp.append(psds)

                    start = start + step
                    stop = start + size

                data.append(np.vstack(temp))
                labels.append([ID[-1:]] * len(np.vstack(temp)))  # add labels (either 0 or 1) for each sample

            data = np.concatenate((data[0], data[1]))
            labels = np.array(sum(labels, []))  # join the data and labels each into one object

            data_2d = data.reshape(len(data), -1)
            data_2d = data_2d / np.std(data_2d)
            clf = SVC(C=1, kernel='linear')
            scores = cross_val_score(clf, data_2d, labels, cv=cv, n_jobs=-1)

        # Printing the results (with some space so we can find them later)
        print("\n\nClassification score: %s (std. %s)\n\n" % (np.mean(scores), np.std(scores)))

elif output == "openvibe":
    mapping = {'OVTK_GDF_End_Of_Trial': 800,
               'OVTK_StimulationId_BaselineStart': 32775,
               'OVTK_StimulationId_BaselineStop': 32776,
               'OVTK_StimulationId_ExperimentStart': 32769,
               'OVTK_StimulationId_ExperimentStop': 32770,
               'OVTK_StimulationId_Label_00': 33024,
               'OVTK_StimulationId_Label_01': 33025,
               'OVTK_StimulationId_Label_10': 33040,
               'OVTK_StimulationId_Label_11': 33041,
               'BAD boundary': 33538,  # mark these as artifacts
               'EDGE boundary': 33538}

    temp = []
    for description in all_raws.annotations.description:
        temp.append(mapping[description])  # convert each description to an integer
    all_raws.annotations.description = np.array(["Stimulus/S " + str(i) for i in temp])  # convert to "Stimulus/S ####"
    # Brainvision requires "stimulus" to be in integer form, not string, so openvibe needs the same!

    mne.export.export_raw("../Data/emotion/exp.vhdr", all_raws, fmt='brainvision', overwrite=True)

else:
    print("Incorrect output option")
