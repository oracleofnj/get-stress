import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT_PATH="/home/jss2272"
TACOTRON_PATH=os.path.join(ROOT_PATH, "tacotron")
STRESS=os.path.join(ROOT_PATH, "get-stress")

def train_test_split(feats, alignments, test_frac=0.1, random_state=6998):
    np.random.seed(random_state)
    test_ids = np.random.choice(
        np.arange(len(feats.keys())),
        int(test_frac * len(feats.keys())),
        replace=False
    )
    all_keys = np.array(feats.keys())
    test_mask = np.isin(np.arange(len(feats.keys())), test_ids)
    train_keys = all_keys[~test_mask]
    test_keys = all_keys[test_mask]

    feats_train = {k: v for k, v in feats.items() if k in train_keys}
    feats_test = {k: v for k, v in feats.items() if k in test_keys}
    alignments_train = {k: v for k, v in alignments.items() if k in train_keys}
    alignments_test = {k: v for k, v in alignments.items() if k in test_keys}
    return feats_train, feats_test, alignments_train, alignments_test


def plot_utterance(key, feats, alignments):
    fst = feats[key]
    align = alignments[key]
    print(len(fst))
    print(np.sum([phone['duration'] for phone in align]))
    fig, ax1 = plt.subplots(figsize=(8,len(fst)/20))
    ax2 = ax1.twiny()

    ax1.plot(fst[:,-2], np.arange(len(fst)), color='b', label='pitch')
    ax1.set_xlabel('pitch', color='b')
    ax1.tick_params('x', colors='b')
    ax2.plot(fst[:,0], np.arange(len(fst)), color='r', label='power')
    ax2.set_xlabel('power', color='r')
    ax2.tick_params('x', colors='r')

    for phone in align:
        ax1.axhline(y=100*phone['start_time'])

    ax1.set_yticks([100 * phone['start_time'] for phone in align])
    ax1.set_yticklabels([phone['phone'][:-2] + ' ({0})'.format(int(100 * phone['start_time'])) for phone in align])
    ax1.invert_yaxis()
    plt.show()


def assemble_phonedicts(feats, alignments):
    phonedicts = {}
    for key, align in alignments.items():
        for phone in align:
            short_phone = phone['phone'][:-2] if phone['phone'][-2] == '_' else phone['phone']
            if short_phone not in phonedicts:
                phonedicts[short_phone] = []
            start_frame = int(100 * phone['start_time'])
            end_frame = int(100 * (phone['start_time'] + phone['duration']))
            phonedicts[short_phone].append({
                'key': key,
                'start_time': phone['start_time'],
                'duration': phone['duration'],
                'start_frame': start_frame,
                'end_frame': end_frame,
                'power': feats[key][start_frame:end_frame,0],
                'pitch': feats[key][start_frame:end_frame,-2],
            })
    return phonedicts


def get_features_for_one_phoneme(inst):
    power = inst['power']
    pitch = inst['pitch']
    num_frames = len(power)
    xs = np.linspace(-1, 1, num_frames)
    return np.concatenate((
        np.polynomial.legendre.legfit(x=xs, y=pitch, deg=2),
        np.polynomial.legendre.legfit(x=xs, y=power, deg=2),
        [num_frames]
    ))


def get_features(phonedicts, phone):
    return np.array([
        get_features_for_one_phoneme(inst)
        for inst in phonedicts[phone]
    ])


def get_clusters(phonedicts, phone, n_clusters=4):
    phone_features = get_features(phonedicts, phone)
    ss = StandardScaler()
    ss.fit(phone_features)
    phone_scaled = ss.transform(phone_features)
    km = KMeans(n_clusters=n_clusters)
    km.fit(phone_scaled)
    return {
        'scaler': ss,
        'kmeans': km
    }


def get_vowel_clusters(phonedicts, n_clusters=4):
    vowel_phones = [c for c in phonedicts.keys() if c[0] in ['A', 'E', 'I', 'O', 'U']]
    all_clusters = {vowel: get_clusters(phonedicts, vowel, n_clusters=n_clusters)
                    for vowel in vowel_phones}
    return all_clusters



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
        start_frame = int(100 * phone['start_time'])
        end_frame = int(100 * (phone['start_time'] + phone['duration']))
        features = np.array([get_features_for_one_phoneme({
                'power': key_features[start_frame:end_frame,0],
                'pitch': key_features[start_frame:end_frame,-2],
        })])
        vc_model = vowel_clusters[phone['phone'][:-2]]
        scaled_features = vc_model['scaler'].transform(features)
        km_cluster = vc_model['kmeans'].predict(scaled_features)
        cluster = km_cluster[0]
        short_phone = '{0}_CL{1}'.format(phone['phone'][:-2], cluster)

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


def save_transcriptions(filename, transcriptions):
    with open(filename, 'w') as f:
        for k, v in transcriptions.items():
            f.write('{0}|{1}|{1}\n'.format(
                k[:-4],
                v
            ))


def make_models_and_metadata(alignments_file, features_npz_file, vowel_pkl_file, train_metadata_file, test_metadata_file):
    with open(alignments_file) as f:
        alignments_all = json.load(f)
    feats_all = np.load(features_npz_file)
    feats_train, feats_test, alignments_train, alignments_test = train_test_split(feats_all, alignments_all)
    phonedicts = assemble_phonedicts(feats_train, alignments_train)
    vc = get_vowel_clusters(phonedicts)
    with open(vowel_pkl_file, 'wb') as f:
        pickle.dump(vc, f)
    with open(vowel_pkl_file, 'rb') as f:
        vc = pickle.load(f)
    train_transcriptions = transcribe_all(feats_train, alignments_train, vc)
    test_transcriptions = transcribe_all(feats_test, alignments_test, vc)
    save_transcriptions(train_metadata_file, train_transcriptions)
    save_transcriptions(test_metadata_file, test_transcriptions)


if __name__ == "__main__":
    make_models_and_metadata(
        os.path.join(STRESS, sys.argv[1]),     # all_alignments.json
        os.path.join(STRESS, sys.argv[2]),     # numpy_features.npz
        os.path.join(STRESS, sys.argv[3]),     # vowel_models.pkl
        os.path.join(TACOTRON_PATH, sys.argv[4]), # train_metadata.csv
        os.path.join(TACOTRON_PATH, sys.argv[5]), # test_metadata.csv'
    )
