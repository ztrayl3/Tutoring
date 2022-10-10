import pickle
import mne
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load our database of subjects
study = "erp"  # emotion or erp
source = open(study + ".pkl", "rb")
all_raws = pickle.load(source)
source.close()

for raw in all_raws:  # for every subject, we will begin pre-processing
    #######
    # ICA #
    #######

    # Pre-Processing
    artifact_removal = raw.copy()
    artifact_removal.filter(l_freq=1.0, h_freq=None, n_jobs=-1)  # high-pass filter at 1Hz
    artifact_removal.notch_filter(50.0, n_jobs=-1)  # notch filter at 50Hz

    # ICA artifact removal
    ica = mne.preprocessing.ICA(n_components=0.95, random_state=97, max_iter="auto")
    ica.fit(artifact_removal)  # fit the ICA with EEG and EOG information

    # Visually inspect the data
    N = ica.n_components_
    ica.plot_properties(raw, picks=list(range(0, N)))  # further analyze the channels we marked as bad
    matplotlib.pyplot.show(block=True)  # wait until all figures are closed

    # last chance to un-bad components...
    response = input("Type any bad components (0-{}) that should be marked for exclusion (seperated by spaces): ".format(N))
    ica.exclude = [int(x) for x in response.split(" ")]  # mark bad components for removal

    # See the changes we've made
    ica.plot_overlay(raw, exclude=ica.exclude, picks='eeg')
    matplotlib.pyplot.show(block=True)  # wait until all figures are closed

    ica.apply(raw)  # apply ICA to data, removing the artifacts
