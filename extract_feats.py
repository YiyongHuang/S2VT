import sys
import cv2
import imageio
imageio.plugins.ffmpeg.download()
import pylab
import numpy as np
sys.path.insert(0,'/usr/local/bin/caffe/python')
import caffe
import skimage.transform
import os
def extract_feats(filenames,batch_size):
    """Function to extract VGG-16 features for frames in a video.
       Input:
            filenames:  List of filenames of videos to be processes
            batch_size: Batch size for feature extraction
       Writes features in .npy files"""
    model_file = '/data/video-captioning/VGG_ILSVRC_16_layers.caffemodel'
    deploy_file = '/data/video-captioning/VGG16_deploy.prototxt'
    net = caffe.Net(deploy_file,model_file,caffe.TEST)
    layer = 'fc7'
    mean_file = '/data/video-captioning/ilsvrc_2012_mean.npy'
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_mean('data',np.load(mean_file).mean(1).mean(1))
    transformer.set_transpose('data',(2,0,1))
    transformer.set_raw_scale('data',255.0)
    net.blobs['data'].reshape(batch_size,3,224,224)
    print("VGG Network loaded")
    #Read videos and extract features in batches
    for file in filenames:
        vid = imageio.get_reader("/data/video-captioning/Data/YouTubeClips/" + file, 'ffmpeg')
        curr_frames = []
        for frame in vid:
            frame = skimage.transform.resize(frame,[224,224])
            if len(frame.shape)<3:
                frame = np.repeat(frame,3).reshape([224,224,3])
            curr_frames.append(frame)
        curr_frames = np.array(curr_frames)
        print("Shape of frames: {0}".format(curr_frames.shape))
        idx = np.linspace(0,len(curr_frames)-1,80).astype(int)    #get 80 frames per vid
        curr_frames = curr_frames[idx,:,:,:]
        print("Captured 80 frames: {0}".format(curr_frames.shape))
        curr_feats = []
        for i in range(0,80,batch_size):
            caffe_in = np.zeros([batch_size,3,224,224])
            curr_batch = curr_frames[i:i+batch_size,:,:,:]
            for j in range(batch_size):
                caffe_in[j] = transformer.preprocess('data',curr_batch[j])
            out = net.forward_all(blobs=[layer],**{'data':caffe_in})
            curr_feats.extend(out[layer])
            print("Appended {} features {}".format(j+1,out[layer].shape))
        curr_feats = np.array(curr_feats)
        np.save("/data/video-captioning/Data/Features_VGG/"+file[:-4] + '.npy',curr_feats)
        print("Saved file {}\nExiting".format(file[:-4] + '.npy'))


if __name__ == "__main__":
    namelist = os.listdir("/data/video-captioning/Data/YouTubeClips")
    extract_feats(namelist, 10)