import matplotlib
import pandas
import mne


# read raw EEG file
files = ["Data/emotion/raw_exp1.edf", "Data/emotion/raw_exp2.edf", "Data/emotion/raw_exp3.edf",
         "Data/emotion/raw_exp4.edf", "Data/emotion/raw_exp5.edf", "Data/emotion/raw_exp6.edf"]
raws = []

for eeg_path in files:
    raw = mne.io.read_raw_edf(eeg_path, preload=True)
    montage = mne.channels.make_standard_montage('standard_1020')  # load the standard 10-20
    raw.set_montage(montage, on_missing="ignore")  # apply the channel montage
    raws.append(raw)

"""
for eeg_path in files:
    raw = mne.io.read_raw_edf(eeg_path, preload=True)

    # read valence and arousal responses
    labels = pandas.read_csv("Data/emotion/0_offline.csv")

    # provide accurate channel names
    channels = ["F7", "Fp1", "Fp2", "F8", "F3", "Fz", "F4", "C3", "Cz", "P8", "P7", "Pz", "P4", "T3", "P3", "O1", "O2",
                "C4", "T4"]

    # rename channels (only eeg channels [:19])
    mne.rename_channels(raw.info, dict(zip(raw.ch_names[:19], channels)))

    # set montage and channel types
    montage = mne.channels.make_standard_montage('standard_1020')  # load the standard 10-20

    new_types = []
    for i in raw.ch_names:  # for each channel that we have in raw...
        if i in channels:  # if it is a 10-20 channel name...
            new_types.append("eeg")  # label it as EEG
        else:  # if it isn't 10-20...g
            new_types.append("misc")  # label it as misc
    raw.set_channel_types(dict(zip(raw.ch_names, new_types)))  # apply the new channel types
    raw.set_montage(montage, on_missing="ignore")  # apply the channel montage

    # only pick the eeg channels
    raw.pick(["eeg"])

    # map names of annotations to their ID
    mapping = {'OVTK_GDF_End_Of_Trial': 800,
               'OVTK_StimulationId_BaselineStart': 32775,
               'OVTK_StimulationId_BaselineStop': 32776,
               'OVTK_StimulationId_ExperimentStart': 32769,
               'OVTK_StimulationId_ExperimentStop': 32770,
               'OVTK_StimulationId_Label_01': 33025,
               'OVTK_StimulationId_Label_02': 33026,
               'OVTK_StimulationId_Label_03': 33027,
               'OVTK_StimulationId_Label_04': 33028,
               'OVTK_StimulationId_Label_05': 33029}

    # create events from annotations
    events, event_dict = mne.events_from_annotations(raw)#, mapping)

    # extract the annotations as a dataframe to iterate over
    anot = raw.annotations.to_data_frame()

    # create mappings from integer response to high/low valence/arousal
    valence_mapping = {1: "OVTK_StimulationId_Label_00", 2: "OVTK_StimulationId_Label_00", 3: "OVTK_StimulationId_Label_00",
                       4: "OVTK_StimulationId_Label_00", 5: "OVTK_StimulationId_Label_01", 6: "OVTK_StimulationId_Label_01",
                       7: "OVTK_StimulationId_Label_01", 8: "OVTK_StimulationId_Label_01", 9: "OVTK_StimulationId_Label_01"}
    arousal_mapping = {1: "OVTK_StimulationId_Label_10", 2: "OVTK_StimulationId_Label_10", 3: "OVTK_StimulationId_Label_10",
                       4: "OVTK_StimulationId_Label_10", 5: "OVTK_StimulationId_Label_11", 6: "OVTK_StimulationId_Label_11",
                       7: "OVTK_StimulationId_Label_11", 8: "OVTK_StimulationId_Label_11", 9: "OVTK_StimulationId_Label_11"}

    # these are the videos we are interested in
    video_labels = [33025, 33026, 33027, 33028, 33029]

    # create empty lists for the 3 columns
    onsets = []
    durations = []
    descriptions = []
    # identify start time
    start_time = anot["onset"][0]

    # iterate through original annotation dataframe
    for index, row in anot.iterrows():

        # identify annotation id
        anot_id = mapping[row["description"]]

        # find the videos we are interested in
        if anot_id in video_labels:

            # calculate the duration of this video
            duration = (anot["onset"][index + 1] - anot["onset"][index]).total_seconds()

            # extract the reported valence
            val_reported = labels[labels['Unnamed: 0'] == anot_id]["Valence"]
            # map the reported calence to low or high valence
            val_mapped = valence_mapping[int(val_reported)]

            # add appropriate values for the 3 columns
            onsets += [(row["onset"] - start_time).total_seconds() + 0.01]
            durations += [duration]
            descriptions += [val_mapped]

            # repeat the process for arousal
            ar_reported = labels[labels['Unnamed: 0'] == anot_id]["Arousal"]
            ar_mapped = arousal_mapping[int(ar_reported)]

            onsets += [(row["onset"] - start_time).total_seconds() + 0.02]
            durations += [duration]
            descriptions += [ar_mapped]

        # if the annotation is not a video then just copy it down
        else:
            onsets += [(row["onset"] - start_time).total_seconds()]
            durations += [row["duration"]]
            descriptions += [row["description"]]

    # create a dataframe for the new annotations
    emotion_annotations = pandas.DataFrame(data=[onsets, durations, descriptions]).transpose()
    # rename the columns
    emotion_annotations = emotion_annotations.rename(columns={0: "onset", 1: "duration", 2: "description"})

    # set the new annotations for the raw file
    raw_annotations = mne.Annotations(onset=list(emotion_annotations["onset"]),  # onset, in seconds
                                      duration=list(emotion_annotations["duration"]),  # duration in seconds
                                      description=list(emotion_annotations["description"])  # textual descriptors
                                      )
    raw.set_annotations(raw_annotations)

    # let's look at the events with the new annotations
    events, event_dict = mne.events_from_annotations(raw)

    raws.append(raw)
"""

# combine all raw files into one large one
all_raws = mne.concatenate_raws(raws)
events, event_dict = mne.events_from_annotations(all_raws)

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

want_epoch = True
if want_epoch:
    epochs = mne.Epochs(all_raws, events, event_id=event_dict, tmin=-0.2, tmax=30,
                        preload=False, event_repeated="drop", reject=None, reject_by_annotation=None, proj=False)

    relevant = ["OVTK_StimulationId_Label_00", "OVTK_StimulationId_Label_01",
                "OVTK_StimulationId_Label_10", "OVTK_StimulationId_Label_11"]
    for i in relevant:
        epochs[i].compute_psd(method="welch", average='mean', picks=['eeg'], fmax=60).plot()

#mne.export.export_raw("exp.vhdr", raw, fmt='brainvision', overwrite=True)
