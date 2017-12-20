from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

"""
with open('all_alignments.json') as f:
    alignments_all = json.load(f)
feats_all = np.load('numpy_features.npz')
feats_vc, align_vc = subsample(feats_all, alignments_all, 0.05)
pitch_and_power_vc = get_all_pitch_and_power(feats_vc)
vowels = assemble_vowels(pitch_and_power_vc, align_vc)
vc = get_combined_vowel_clusters(vowels)
with open('combined_vowel_clusters.pkl', 'wb') as f:
    pickle.dump(vc, f)


feats_tr, align_tr = subsample(feats_all, alignments_all, 0.001, 6999)
pitch_and_power_tr = get_all_pitch_and_power(feats_tr)
transcribe_all(pitch_and_power_tr, align_tr, vc)
"""


def train_test_split(feats, alignments, test_frac=0.1, random_state=6998):
    np.random.seed(random_state)
    all_keys = np.array(list(set(alignments.keys()).intersection(feats.keys())))
    test_ids = np.random.choice(
        np.arange(len(all_keys)),
        int(test_frac * len(all_keys)),
        replace=False
    )
    test_mask = np.isin(np.arange(len(all_keys)), test_ids)
    train_keys = all_keys[~test_mask]
    test_keys = all_keys[test_mask]

    feats_train, alignments_train = {}, {}
    for k in train_keys:
        feats_train[k] = feats[k]
        alignments_train[k] = alignments[k]

    feats_test, alignments_test = {}, {}
    for k in test_keys:
        feats_test[k] = feats[k]
        alignments_test[k] = alignments[k]

    return feats_train, feats_test, alignments_train, alignments_test


def subsample(feats, alignments, frac=0.1, random_state=6998):
    np.random.seed(random_state)
    all_keys = np.array(list(set(alignments.keys()).intersection(feats.keys())))
    sample_ids = np.random.choice(
        np.arange(len(all_keys)),
        int(frac * len(all_keys)),
        replace=False
    )

    sample_mask = np.isin(np.arange(len(all_keys)), sample_ids)
    sample_keys = all_keys[sample_mask]

    feats_sample, alignments_sample = {}, {}
    for k in sample_keys:
        feats_sample[k] = feats[k]
        alignments_sample[k] = alignments[k]

    return feats_sample, alignments_sample


def plot_utterance(key, pitch_and_power, alignments):
    pp = pitch_and_power[key]
    align = alignments[key]
    num_frames = pp['pitch'].shape[0]
    print(num_frames)
    print(np.sum([phone['duration'] for phone in align]))
    fig, ax1 = plt.subplots(figsize=(8, num_frames/10))
    ax2 = ax1.twiny()

    ax1.plot(pp['pitch'], np.arange(num_frames), color='b', label='pitch')
    ax1.set_xlabel('pitch', color='b')
    ax1.tick_params('x', colors='b')
    ax2.plot(pp['power'], np.arange(num_frames), color='r', label='power')
    ax2.set_xlabel('power', color='r')
    ax2.tick_params('x', colors='r')

    for phone in align:
        ax1.axhline(y=100*phone['start_time'])

    ax1.set_yticks([100 * phone['start_time'] for phone in align])
    ax1.set_yticklabels([
        (phone['phone'][:-2] if phone['phone'] != 'SIL' else 'SIL') +
        ' ({0})'.format(int(100 * phone['start_time']))
        for phone in align
    ])
    ax1.set_ylim(num_frames-1, 0)
    plt.show()


def get_pitch_and_power(feat):
    return {
        'pitch': feat[:, -2],
        'power': (feat[:, 0] - np.mean(feat[:, 0])) / np.std(feat[:, 0])
    }


def get_all_pitch_and_power(feats):
    return {
        k: get_pitch_and_power(v)
        for k, v in feats.items()
    }


def phone_to_inst(phone, pp):
    start_frame = int(np.maximum(0.0, -2.0 + 100 * phone['start_time']))
    end_frame = int(2.0 + 100 * (phone['start_time'] + phone['duration']))
    return {
        'power': pp['power'][start_frame:end_frame],
        'pitch': pp['pitch'][start_frame:end_frame],
        'duration': phone['duration'],
    }


def assemble_vowels(pitch_and_power, alignments):
    vowels = []
    for key, algn in alignments.items():
        for phone in algn:
            if phone['phone'][0] in ['A', 'E', 'I', 'O', 'U']:
                vowels.append(phone_to_inst(phone, pitch_and_power[key]))
    return vowels


def get_features_for_one_phoneme(inst):
    power = inst['power']
    pitch = inst['pitch']
    num_frames = len(power)
    if num_frames < 2:
        print(inst)
        raise ValueError("yoooo")
    xs = np.linspace(-1, 1, num_frames)
    return np.concatenate((
        np.polynomial.legendre.legfit(x=xs, y=pitch, deg=2),
        np.polynomial.legendre.legfit(x=xs, y=power, deg=2),
        [inst['duration']]
    ))


def get_vowel_features(vowels):
    return np.array([
        get_features_for_one_phoneme(inst)
        for inst in vowels
    ])


def get_combined_vowel_clusters(vowels, n_clusters=8):
    vowel_features = get_vowel_features(vowels)
    ss = StandardScaler()
    ss.fit(vowel_features)
    phone_scaled = ss.transform(vowel_features)
    km = KMeans(n_clusters=n_clusters)
    km.fit(phone_scaled)
    return {
        'scaler': ss,
        'kmeans': km
    }


def cluster_phoneme(key_features, vowel_clusters, phone):
    if phone['phone'] == 'SIL':
        return phone['phone']

    if phone['phone'][-2:] == '_E':
        word_boundary = True
    else:
        word_boundary = False

    if phone['phone'][0] not in ['A', 'E', 'I', 'O', 'U']:
        short_phone = phone['phone'][:-2]
    else:
        inst = phone_to_inst(phone, key_features)
        features = np.array([get_features_for_one_phoneme(inst)])
        scaled_features = vowel_clusters['scaler'].transform(features)
        km_cluster = vowel_clusters['kmeans'].predict(scaled_features)
        cluster = km_cluster[0]
        short_phone = '{0} VOWEL{1}'.format(phone['phone'][:-2], cluster)

    if word_boundary:
        return short_phone + ' sp'
    else:
        return short_phone


def transcribe_key(feats, alignments, vowel_clusters, key):
    return ' '.join([
        cluster_phoneme(feats[key], vowel_clusters, align)
        for align in alignments[key]
    ])


def transcribe_all(feats, alignments, vowel_clusters):
    return {key: transcribe_key(feats, alignments, vowel_clusters, key)
            for key in feats.keys()}


def save_transcriptions(filename, transcriptions, strip_four=True):
    with open(filename, 'w') as f:
        for k, v in transcriptions.items():
            if strip_four:
                keyname = k[:-4]
            else:
                keyname = k
            f.write('{0}|{1}|{1}\n'.format(
                keyname,
                v
            ))


def make_vowel_clusters(alignments_file, features_npz_file, vowel_pkl_file, subsampling=0.05):
    with open(alignments_file) as f:
        alignments_all = json.load(f)
    feats_all = np.load(features_npz_file)
    feats_vc, align_vc = subsample(feats_all, alignments_all, subsampling)
    pitch_and_power_vc = get_all_pitch_and_power(feats_vc)
    vowels = assemble_vowels(pitch_and_power_vc, align_vc)
    vc = get_combined_vowel_clusters(vowels)
    with open(vowel_pkl_file, 'wb') as f:
        pickle.dump(vc, f)


def make_transcriptions_subsample(alignments_file, features_npz_file, vowel_pkl_file, sample_metadata_file, subsampling=0.05, random_state=6999, strip_four=True):
    with open(alignments_file) as f:
        alignments_all = json.load(f)
    feats_all = np.load(features_npz_file)
    with open(vowel_pkl_file, 'rb') as f:
        vc = pickle.load(f)
    feats_samp, align_samp = subsample(feats_all, alignments_all, subsampling, random_state)
    pitch_and_power_samp = get_all_pitch_and_power(feats_samp)
    sample_transcriptions = transcribe_all(pitch_and_power_samp, align_samp, vc)
    save_transcriptions(sample_metadata_file, sample_transcriptions, strip_four=strip_four)


def plot_utterances(alignments_file, features_npz_file):
    with open(alignments_file) as f:
        alignments_all = json.load(f)
    feats_all = np.load(features_npz_file)
    pitch_and_power = get_all_pitch_and_power(feats_all)
    print(type(feats_all))
    print(feats_all.keys())
    print(type(pitch_and_power))
    print(pitch_and_power.keys())


def make_transcriptions_train_test(alignments_file, features_npz_file, vowel_pkl_file, train_metadata_file, test_metadata_file):
    with open(alignments_file) as f:
        alignments_all = json.load(f)
    feats_all = np.load(features_npz_file)
    with open(vowel_pkl_file, 'rb') as f:
        vc = pickle.load(f)
    feats_train, feats_test, align_train, align_test = train_test_split(feats_all, alignments_all)
    pitch_and_power_train, pitch_and_power_test = get_all_pitch_and_power(feats_train), get_all_pitch_and_power(feats_test)
    train_transcriptions = transcribe_all(pitch_and_power_train, align_train, vc)
    test_transcriptions = transcribe_all(pitch_and_power_test, align_test, vc)
    save_transcriptions(train_metadata_file, train_transcriptions)
    save_transcriptions(test_metadata_file, test_transcriptions)
