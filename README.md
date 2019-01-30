**Video Caption using S2VT**

**Requirements**

For running my code and reproducing the results, the following packages need to be installed first. I have used Python 3.6 for the whole of this project.

PyTorch  
Caffe  
NumPy  
cv2  
imageio  
scikit-image

**Running instructions**

1.Install all the packages mentioned in the 'Requirements' section for the smooth running of this project.  
2.Download the MSVD dataset in "YouTubeClips"  
3.Change all the path in these python files to point to directories in your workspace  
4.Run extract_feats.py to extract the RGB features of videos  
5.Run train.py to train the model  
6.Run test.py to generate the caption of test videos  

**Acknowledgement**

Some code copy from vijayvee(https://github.com/vijayvee/video-captioning)