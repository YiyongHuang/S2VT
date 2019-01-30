from module import *
from utils import *

vovab_size = len(word_counts)
BATCH_SIZE = 10

if __name__ == "__main__":
    s2vt = S2VT(vocab_size=vovab_size, batch_size=BATCH_SIZE)
    s2vt = s2vt.cuda()
    s2vt.load_state_dict(torch.load("/data/video-captioning/Data/s2vt_params.pkl"))
    s2vt.eval()
    for i in range(10):
        video, caption, cap_mask = fetch_val_data(batch_size=BATCH_SIZE)
        video = torch.FloatTensor(video).cuda()

        cap_out = s2vt(video)

        captions = []
        for tensor in cap_out:
            captions.append(tensor.tolist())
        # size of captions : [79, batch_size]

        # transform captions to [batch_size, 79]
        captions = [[row[i] for row in captions] for i in range(len(captions[0]))]

        print('............................\nGT Caption:\n')
        print_in_english(captions)
        print('............................\nLABEL Caption:\n')
        print_in_english(caption)








