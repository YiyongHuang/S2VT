from module import *
import torch
from torch import nn
from utils import *


EPOCH = 30
nIter = 1576
BATCH_SIZE = 10
LEARNING_RATE = 0.0001
vovab_size = len(word_counts)

# save training log
def write_txt(epoch, iteration, loss):
    with open("/data/video-captioning/training_log.txt", 'a+') as f:
        f.write("Epoch:[ %d ]\t Iteration:[ %d ]\t loss:[ %f ]\n" % (epoch, iteration, loss))


if __name__ == "__main__":
    pkl_file = None
    s2vt = S2VT(vocab_size=vovab_size, batch_size=BATCH_SIZE)
    if pkl_file:
        s2vt.load_state_dict(torch.load("/data/video-captioning/Data/s2vt_params.pkl"))
    s2vt = s2vt.cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(s2vt.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH):
        for i in range(nIter):
            video, caption, cap_mask = fetch_train_data(BATCH_SIZE)
            video, caption, cap_mask = torch.FloatTensor(video).cuda(), torch.LongTensor(caption).cuda(), \
                                       torch.FloatTensor(cap_mask).cuda()

            cap_out = s2vt(video, caption)
            cap_labels = caption[:, 1:].contiguous().view(-1)       # size [batch_size, 79]
            cap_mask = cap_mask[:, 1:].contiguous().view(-1)        # size [batch_size, 79]

            logit_loss = loss_func(cap_out, cap_labels)
            masked_loss = logit_loss*cap_mask
            loss = torch.sum(masked_loss)/torch.sum(cap_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%20 == 0:
                # print("Epoch: %d  iteration: %d , loss: %f" % (epoch, i, loss))
                write_txt(epoch, i, loss)
            if i%2000 == 0:
                torch.save(s2vt.state_dict(), "/data/video-captioning/Data/s2vt_params.pkl")
                print("Epoch: %d iter: %d save successed!" % (epoch, i))


