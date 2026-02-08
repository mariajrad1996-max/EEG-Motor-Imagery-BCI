"""
EEG Motor Imagery Classification using CSP + LDA
Dataset: EEGBCI (PhysioNet)
Author: Maria Jrad

"""


# Call libraries needed
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score


import os

import mne
import matplotlib.pyplot as plt
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf


# Load the dataset, read data from subject 1, and combine all runs that include right and left hand imaginery movements, in addition to rest
# Exclude S088, S092, S100, since these subjects had damaged recordings, and too little samples (S104) in their left- and right-hand motor imagery datasets
from mne.datasets import eegbci
subject = 1
runs = [4, 8, 12]  # motor imagery: left hand (T1) vs right hand (T2)
raw_fnames = eegbci.load_data(subject, runs)
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw)  # set channel names
montage = make_standard_montage("standard_1005")
raw.set_montage(montage)


# Plot data and channel locations
raw.plot(duration=8, n_channels=64)
raw.plot_sensors(show_names=True)


# Apply band-pass filter
# Frequency range chosen will be mu band (8-12 Hz)
# Discuss the plots with your colleagues, is that what you expected? If not, then why?
raw.filter(8, 12, fir_design="firwin", skip_by_annotation="edge")
raw.plot(duration=8, n_channels=64)


# Read epochs
event_id = dict(left_hand=2, right_hand=3)
events, _ = events_from_annotations(raw, event_id=dict(T1=2,  T2=3))
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
tmin, tmax = -2, 6
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=False, picks=picks, baseline=(-2,-1), preload=True)



# Epochs separation
# Discuss the plots with your colleagues, is that what you expected? If not, then why?
Left_hand_epochs = epochs['left_hand']
Left_hand_epochs.plot_image(picks=["C3", "C4"])
Right_hand_epochs = epochs['right_hand']
Right_hand_epochs.plot_image(picks=["C3", "C4"])



# Averaging evoked responses
# Discuss the plots with your colleagues, is that what you expected? If not, then why?
Left_hand_epochs_evoked = Left_hand_epochs.average()
Right_hand_epochs_evoked = Right_hand_epochs.average()
mne.viz.plot_compare_evokeds(dict(Left=Left_hand_epochs_evoked, Right=Right_hand_epochs_evoked), legend="upper left", show_sensors="upper right")



# Visualizing Spectrum objects
# Discuss the plots with your colleagues, is that what you expected? If not, then why?
Left_hand_epochs_evoked_spectrum = Left_hand_epochs_evoked.compute_psd()
Left_hand_epochs_evoked_spectrum.plot()
Right_hand_epochs_evoked_spectrum = Right_hand_epochs_evoked.compute_psd()
Right_hand_epochs_evoked_spectrum.plot()



#Time-frequency analysis
# Discuss the plots with your colleagues, is that what you expected? If not, then why?
frequencies = np.arange(8, 12, 1)

power_leftC4 = mne.time_frequency.tfr_morlet(Left_hand_epochs, n_cycles=2, return_itc=False, freqs=frequencies, decim=1)
power_leftC4.plot(["C4"])
power_rightC4 = mne.time_frequency.tfr_morlet(Right_hand_epochs, n_cycles=2, return_itc=False, freqs=frequencies, decim=1)
power_rightC4.plot(["C4"])

power_leftC3 = mne.time_frequency.tfr_morlet(Left_hand_epochs, n_cycles=2, return_itc=False, freqs=frequencies, decim=1)
power_leftC3.plot(["C3"])
power_rightC3 = mne.time_frequency.tfr_morlet(Right_hand_epochs, n_cycles=2, return_itc=False, freqs=frequencies, decim=1)
power_rightC3.plot(["C3"])




# Fitting and ploting the CSP filters to the data
# Discuss the plots with your colleagues, is that what you expected? If not, then why?
from mne.decoding import CSP
csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
labels = epochs.events[:, -1] - 2
epochs_data = epochs.get_data()
csp.fit_transform(epochs_data, labels)
csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)



# Define a monte-carlo cross-validation generator (reduce variance):
epochs_train = epochs.copy().crop(tmin=0, tmax=4)
scores = []
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

# Assemble a classifier
lda = LinearDiscriminantAnalysis()

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([("CSP", csp), ("LDA", lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1.0 - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))



# Fitting and testing the classifier
# Discuss the plots with your colleagues, is that what you expected? If not, then why?
sfreq = raw.info["sfreq"]
w_length = int(sfreq * 0.5)  # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

scores_windows = []

for train_idx, test_idx in cv_split:
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    X_test = csp.transform(epochs_data_train[test_idx])

    # fit classifier
    lda.fit(X_train, y_train)

    # running classifier: test classifier on sliding window
    score_this_window = []
    for n in w_start:
        X_test = csp.transform(epochs_data[test_idx][:, :, n : (n + w_length)])
        score_this_window.append(lda.score(X_test, y_test))
    scores_windows.append(score_this_window)

# Plot scores over time
w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
plt.axvline(0, linestyle="--", color="k", label="Onset")
plt.axhline(0.5, linestyle="-", color="k", label="Chance")
plt.xlabel("time (s)")
plt.ylabel("classification accuracy")
plt.title("Classification score over time")
plt.legend(loc="lower right")
plt.show()



# Load the dataset, read data from subject 1, and combine all runs that include right and left hand imaginery movements, in addition to rest
# Exclude S088, S092, S100, since these subjects had damaged recordings, and too little samples (S104) in their left- and right-hand motor imagery datasets
from mne.datasets import eegbci
subject = 1
runs = [4, 8, 12]  # motor imagery: rest (T0) vs left hand (T1) or vs right hand (T2)
raw_fnames = eegbci.load_data(subject, runs)
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw)  # set channel names
montage = make_standard_montage("standard_1005")
raw.set_montage(montage)







