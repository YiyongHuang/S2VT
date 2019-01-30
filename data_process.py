import numpy as np

TXT_DIR = '/data/video-captioning/text_files/'


def build_vocab(word_count_threshold, unk_requried=False):
    sentance_train = open(TXT_DIR+'sents_train_lc_nopunc.txt', 'r').read().splitlines()
    sentance_val = open(TXT_DIR+'sents_val_lc_nopunc.txt', 'r').read().splitlines()
    sentance_test = open(TXT_DIR+'sents_test_lc_nopunc.txt', 'r').read().splitlines()
    all_captions = []
    word_counts = {}
    for cap in sentance_test+sentance_train+sentance_val:
        caption = cap.split('\t')[-1]
        caption = '<BOS> ' + caption + ' <EOS>'
        all_captions.append(caption)
        for word in caption.split(' '):
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        for word in word_counts:
            if word_counts[word] < word_count_threshold:
                word_counts.pop(word)
                unk_requried = True
    return word_counts, unk_requried


def word_to_ids(word_counts, unk_requried):
    word_to_id = {}
    id_to_word = {}
    count = 0
    if unk_requried:
        word_to_id['<UNK>'] = count
        id_to_word[count] = '<UNK>'
        count += 1
        print("<UNK> True")
    for word in word_counts:
        word_to_id[word] = count
        id_to_word[count] = word
        count += 1
    return word_to_id, id_to_word


# convert each word of captions to the index
def convert_caption(captions, word_to_id, max_length):
    if type(captions) == 'str':
        captions = [captions]
    caps, cap_mask = [], []
    for cap in captions:
        nWord = len(cap.split(' '))
        cap = cap + ' <EOS>'*(max_length-nWord)
        cap_mask.append([1.0]*nWord + [0.0]*(max_length-nWord))
        cap_ids = []
        for word in cap.split(' '):
            if word in word_to_id:
                cap_ids.append(word_to_id[word])
            else:
                cap_ids.append(word_to_id['<UNK>'])
        caps.append(cap_ids)
    return np.array(caps), np.array(cap_mask)
