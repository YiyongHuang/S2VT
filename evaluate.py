from module import *
from utils import *
import json

vovab_size = len(word_counts)
BATCH_SIZE = 10

if __name__ == "__main__":
    s2vt = S2VT(vocab_size=vovab_size, batch_size=BATCH_SIZE)
    s2vt = s2vt.cuda()
    s2vt.load_state_dict(torch.load("/data/video-captioning/Data/s2vt_params.pkl"))
    s2vt.eval()
    val_result = {}
    for idx in range(385):
        video, caption, cap_mask, vid = fetch_val_data_orderly(idx, batch_size=BATCH_SIZE)
        video = torch.FloatTensor(video).cuda()

        cap_out = s2vt(video)

        captions = []
        for tensor in cap_out:
            captions.append(tensor.tolist())

        captions = [[row[i] for row in captions] for i in range(len(captions[0]))]

        print('............................\nGT Caption:\n')
        print_in_english(captions)
        print('............................\nLABEL Caption:\n')
        print_in_english(caption)
        val_result = save_val_result(vid, captions, val_result)
    with open("/data/video-captioning/result.json", "a+") as f:
        json.dump(val_result, f)

