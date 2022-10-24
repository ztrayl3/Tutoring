from mne.time_frequency import psd_welch
import numpy as np
import pickle
import pandas
import mne
import os

figures = False
N = 0  # number of subjects in the study
chosen = []  # events of interest in the study
all_epochs = []  # list to contain all epochs of interest
all_raws = []  # list to contain all raw files
study = "emotion"  # emotion or erp study
if study == "emotion":
    N = 4
    chosen = ["video_element_v11",
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
elif study == "erp":
    N = 5
    chosen = ["image_element_common",
              "image_element_rare"]

for subject in range(N):  # for each subject...
    ########################
    # LOADING THE EEG DATA #
    ########################

    # load EDF file
    eeg_path = os.path.join("Data", study, 's' + str(subject+1) + ".edf")  # path to RENAMED edf files
    eve_path = os.path.join("Data", study, 's' + str(subject+1) + ".csv")  # path to RENAMED (and edited if erp) csv files
    raw = mne.io.read_raw_edf(eeg_path, preload=True)

    # load events as annotations from CSV file
    raw_events = pandas.read_csv(eve_path)
    raw_annotations = mne.Annotations(onset=list(raw_events["latency"]),  # onset, in seconds
                                      duration=list(raw_events["duration"]),  # duration in seconds
                                      description=list(raw_events["type"])  # textual descriptors
                                      )
    raw.set_annotations(raw_annotations)

    # we're going to correct channel types, since there are really only 32 EEG, not 117
    montage = mne.channels.make_standard_montage('standard_1020')  # load the standard 10-20
    ten_twenty = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1',
                  'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6',
                  'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']  # define some channels
    new_types = []
    for i in raw.ch_names:  # for each channel that we have in raw...
        if i in ten_twenty:  # if it is a 10-20 channel name...
            new_types.append("eeg")   # label it as EEG
        else:  # if it isn't 10-20...
            new_types.append("misc")  # label it as misc
    raw.set_channel_types(dict(zip(raw.ch_names, new_types)))  # apply the new channel types
    raw.set_montage(montage, on_missing="ignore")  # apply the channel montage

    if figures:
        # inspect the EEG data visually
        raw.plot_psd(fmax=64)  # power at frequency
        raw.plot()  # event markers with "raw" EEG data

    all_raws.append(raw)  # save the raw file

    #########################
    # EPOCHING THE EEG DATA #
    #########################

    # turn annotations into events
    if study == "erp":
        mapping = {"image_element_rare": 1,
                   "image_element_common": 2}
        events, event_dict = mne.events_from_annotations(raw, event_id=mapping)
    else:
        events, event_dict = mne.events_from_annotations(raw)

    if figures:
        # visualize timing, to make sure it looks right
        mne.viz.plot_events(events, sfreq=raw.info['sfreq'],
                            first_samp=raw.first_samp, event_id=event_dict)
    # specify how much of the emotional video we want in order to have even sized epochs
    MAX = 20  # in seconds

    # create our epochs (with -0.2 second baseline, not strictly necessary)
    raw.pick(["eeg"])
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=MAX,
                        preload=True, event_repeated="drop")  # take only first event when 2 occur at the same time

    all_epochs.append(epochs[chosen])

epochs_combined = mne.concatenate_epochs(all_epochs)  # join all epochs into one object

# save all raw data to a file for later
data = open(study + "_raws.pkl", "wb")
pickle.dump(all_raws, data)
data.close()

# save all epochs to a file for later
data = open(study + "_epochs.pkl", "wb")
pickle.dump(epochs_combined, data)
data.close()

if study == "emotion":
    # visually inspect differences in frequency bands using a sliding window across all epochs
    bands = ["alpha", "beta", "gamma"]
    results = pandas.DataFrame(np.zeros((len(chosen), len(bands))), index=chosen, columns=bands)

    ###########################
    # TIME FREQUENCY ANALYSIS #
    ###########################

    for video in chosen:  # for each video...
        # Estimate PSDs based on "mean" averaging via a sliding window (Welch approach)
        kwargs = dict(fmin=2, fmax=60, n_jobs=-1)
        psds_welch_mean, freqs_mean = psd_welch(epochs_combined[video], average='mean', **kwargs,
                                                picks=['eeg'], n_per_seg=50, n_overlap=25)

        # Convert power to dB scale.
        psds_welch_mean = 10 * np.log10(psds_welch_mean)

        alpha_range = np.arange(8.0, 12.5, 0.5)  # 8-12 Hz
        alpha = []
        beta_range = np.arange(12.5, 30.5, 0.5)  # 12-30Hz
        beta = []
        gamma_range = np.arange(30.5, 60.5, 0.5)  # 30-100Hz (limited to 60Hz for us)
        gamma = []
        powers = np.average(psds_welch_mean, axis=(0, 1))  # this averages across all channels and all subjects
        for i in range(len(freqs_mean)):  # for each frequency...
            if i in alpha_range:
                alpha.append(powers[i])
            elif i in beta_range:
                beta.append(powers[i])
            elif i in gamma_range:
                gamma.append(powers[i])

        results["alpha"][video] = np.mean(alpha)
        results["beta"][video] = np.mean(beta)
        results["gamma"][video] = np.mean(gamma)

    print(results)

if study == "erp":
    ################
    # ERP ANALYSIS #
    ################

    # Define a function to print out the channel (ch) containing the
    # peak latency (lat; in msec) and amplitude (amp, in µV), with the
    # time range (tmin and tmax) that was searched.
    # This function will be used throughout the remainder of the tutorial.
    def print_peak_measures(ch, tmin, tmax, lat, amp):
        print(f'Channel: {ch}')
        print(f'Time Window: {tmin * 1e3:.3f} - {tmax * 1e3:.3f} ms')
        print(f'Peak Latency: {lat * 1e3:.3f} ms')
        print(f'Peak Amplitude: {amp * 1e6:.3f} µV')


    # Get peak amplitude and latency from a good time window that contains the peak
    good_tmin, good_tmax = 0.2, 0.4
    for stim in chosen:
        current = epochs_combined[stim]
        ch, lat, amp = current.average().get_peak(ch_type='eeg', tmin=good_tmin, tmax=good_tmax,
                                                  mode='pos', return_amplitude=True)

        # Print output from the good time window that contains the peak
        print('** PEAK MEASURES FOR {} **'.format(stim))
        print_peak_measures(ch, good_tmin, good_tmax, lat, amp)

        # This output calculates the latency and amplitude of "P300" for each condition
        # NOTE: this also reports which channel contains this peak measure. Does that look correct?
