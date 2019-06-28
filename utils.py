from data_process import *
import numpy as np


n_lstm_steps = 80
VIDEO_DIR = '/data/video-captioning/Data/Features_VGG/'
TXT_DIR = '/data/video-captioning/text_files/'
Vid2Url = eval(open(TXT_DIR+'Vid2Url_Full.txt').read())
Vid2Cap_train = eval(open(TXT_DIR+'Vid2Cap_train.txt').read())
Vid2Cap_val = eval(open(TXT_DIR+'Vid2Cap_val.txt').read())
Vid2Cap_test = eval(open(TXT_DIR+'Vid2Cap_test.txt').read())
video_train = list(Vid2Cap_train.keys())
video_val = list(Vid2Cap_val.keys())
video_test = list(Vid2Cap_test.keys())
word_counts, unk_required = build_vocab(word_count_threshold=0)
word2id, id2word = word_to_ids(word_counts, unk_requried=unk_required)


# fetch features of train videos
def fetch_train_data(batch_size):
    vid = np.random.choice(video_train, batch_size)
    url = [Vid2Url[video] for video in vid]
    cur_vid = np.array([np.load(VIDEO_DIR+video+'.npy') for video in url])
    feats_idx = np.linspace(0, 79, n_lstm_steps).astype(int)
    cur_vid = cur_vid[:, feats_idx, :]
    captions = [np.random.choice(Vid2Cap_train[video], 1)[0] for video in vid]
    captions, cap_mask = convert_caption(captions, word2id, n_lstm_steps)
    return cur_vid, captions, cap_mask
# vid_size = [batch_size, 80, 4096]     caption_size = [batch_size, 80]


# fetch features of val videos
def fetch_val_data(batch_size):
    vid = np.random.choice(video_val, batch_size)
    url = [Vid2Url[video] for video in vid]
    cur_vid = np.array([np.load(VIDEO_DIR+video+'.npy') for video in url])
    feats_idx = np.linspace(0, 79, n_lstm_steps).astype(int)
    cur_vid = cur_vid[:, feats_idx, :]
    captions = [np.random.choice(Vid2Cap_val[video], 1)[0] for video in vid]
    captions, cap_mask = convert_caption(captions, word2id, n_lstm_steps)
    return cur_vid, captions, cap_mask


def fetch_val_data_orderly(idx, batch_size):
    vid = video_test[idx:idx+batch_size]
    url = [Vid2Url[video] for video in vid]
    cur_vid = np.array([np.load(VIDEO_DIR+video+'.npy') for video in url])
    feats_idx = np.linspace(0, 79, n_lstm_steps).astype(int)
    cur_vid = cur_vid[:, feats_idx, :]
    captions = [np.random.choice(Vid2Cap_test[video], 1)[0] for video in vid]
    captions, cap_mask = convert_caption(captions, word2id, n_lstm_steps)
    return cur_vid, captions, cap_mask, vid


# print captions
def print_in_english(captions):
    captions_english = [[id2word[word] for word in caption] for caption in captions]
    # captions_english = [[row[i] for row in captions_english] for i in range(len(captions_english[0]))]
    for cap in captions_english:
        if '<EOS>' in cap:
            cap = cap[0:cap.index('<EOS>')]
        print(' ' + ' '.join(cap))


def save_val_result(vid, captions, val_result):
    cur_dict = {}
    cur_caps = []
    captions_english = [[id2word[word] for word in caption] for caption in captions]
    for cap in captions_english:
        if '<EOS>' in cap:
            cap = cap[0:cap.index('<EOS>')]
            cap = [' '.join(cap)]
            cur_caps.append(cap)
    for i, id in enumerate(vid):
        cur_dict[id] = cur_caps[i]
    val_result = dict(val_result, **cur_dict)
    return val_result




