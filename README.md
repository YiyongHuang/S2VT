# Video Caption using S2VT

## Requirements

For running my code and reproducing the results, the following packages need to be installed first. I have used Python 3.6 for the whole of this project.

* PyTorch  
* Caffe  
* NumPy  
* cv2  
* imageio  
* scikit-image

## Running instructions

>1.Install all the packages mentioned in the 'Requirements' section for the smooth running of this project.  
>2.Download the MSVD dataset to [Data/YouTubeClips](https://github.com/YiyongHuang/S2VT/tree/master/Data/YouTubeClips)  
>3.Change all the path in these python files to point to directories in your workspace  
>4.Run [extract_feats.py](https://github.com/YiyongHuang/S2VT/blob/master/extract_feats.py) to extract the RGB features of videos  
>5.Run [train.py](https://github.com/YiyongHuang/S2VT/blob/master/train.py) to train the model  
>6.Run [test.py](https://github.com/YiyongHuang/S2VT/blob/master/test.py) to generate the caption of test videos  

or you can directly extract features from a video and generate captions using [test_from_video.py](https://github.com/YiyongHuang/S2VT/blob/master/test_from_video.py)

## DataSet  
You can download the MSVD dataset [here](http://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar)  
You can download the extracted video features at [Features_VGG](https://pan.baidu.com/s/1mylv8GCHKlVl4E2L8xVd7w), and 
unzip it to "Data/Features_VGG"

##Result 
We use some metrics to evaluate the model:  
![](https://github.com/YiyongHuang/S2VT/blob/master/evaluation.jpg)

## Acknowledgement

Some code copy from vijayvee(https://github.com/vijayvee/video-captioning)
