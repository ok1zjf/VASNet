# Video Summarization with Attention
A PyTorch implementation of our paper [Video Summarization with Attention](https://arxiv.org/abs/1812.01969) by 
Jiri Fajtl, Hajar Sadeghi Sokeh, Vasileios Argyriou, Dorothy Monekosso and Paolo Remagnino.
This paper was presented at [ACCV 2018](http://accv2018.net/program/) [AIU2018 workshop](http://www.sys.info.hiroshima-cu.ac.jp/aiu2018/).

## Installation
The development and evaluation was done on the following configuration:
### System configuration
- Platform :   Linux-4.15.0-43-generic-x86_64-with-Ubuntu-16.04-xenial
- Display driver : NVRM version: NVIDIA UNIX x86_64 Kernel Module  384.130  Wed Mar 21 03:37:26 PDT 2018
	GCC version:  gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.10)
- GPU: NVIDIA Titan Xp
- CUDA:  9.0.176
- CUDNN: 7.1.2

### Python packages
- Python: 3.5.2
- PyTorch:  0.4.1
- NumPy: 1.16.1 
- json: 2.0.9
- h5py: 2.8.0
- ortools: 6.9.5824


## Datasets and pretrained models
Preprocessed datasets [TVSum](https://github.com/yalesong/tvsum), [SumMe](https://gyglim.github.io/me/vsum/index.html), 
[YouTube](https://sites.google.com/site/vsummsite/download) and [OVP](https://sites.google.com/site/vsummsite/download) 
as well as VASNet pretrained models you can download by running the following command:
```
./download.sh datasets_models_urls.txt
```
You will need about 820MB space on your HDD. Datasets will be stored in ```./datasets``` 
directory and models, with corresponding split files, in ```./data/models``` and ```./data/splits``` respectively.

Original version of the datasets can be downloaded from 
[http://www.eecs.qmul.ac.uk/~kz303/vsumm-reinforce/datasets.tar.gz](http://www.eecs.qmul.ac.uk/~kz303/vsumm-reinforce/datasets.tar.gz) 
or
[https://www.dropbox.com/s/ynl4jsa2mxohs16/data.zip?dl=0](https://www.dropbox.com/s/ynl4jsa2mxohs16/data.zip?dl=0).

## Evaluation
To evaluate all splits in ```./data/splits``` with corresponding trained models in ```./data/models``` 
run the following: 
```
python3 main.py
```

For experiment saved in different than ```./data``` directory use parameter ```-o <directory_name>``` Results for 
the default split files and given hw/sw configuration are as follows:

```

---------------------------------------------------------
  No   Split                                Mean F-score
=========================================================
  1    splits/tvsum_splits.json             61.428% 
  2    splits/summe_splits.json             49.631% 
  3    splits/tvsum_aug_splits.json         62.457% 
  4    splits/summe_aug_splits.json         51.11%  
---------------------------------------------------------

```
  

## Training
To train the VASNet on all split files in the ```./splits``` directory run this command:
```
python3 main.py --train
```

Results, including a copy of the split and python files, will be stored in ```./data``` directory. 
You can specify different directory with a parameter ```-o <directory_name>``` This is convenient if you 
are running a number of experiments and want to preserve the results and configuration. 

The final results will be recorded in ```./data/results.txt``` with corresponding models in 
the ```./data/models``` directory.    

By default, the training is done with split files in ```./splits``` directory. These files were created 
with ```create_split.py```. For example, to create 5 fold split file for the SumMe dataset run the following command:  
```
python3 create_split.py -d datasets/eccv16_dataset_summe_google_pool5.h5 --save-dir splits --save-name summe_splits --num-splits 5
```
The split file will be saved as ```./splits/summe_splits.json```
   

## Acknowledgement
We would like to thank to [K. Zhou et al.](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) 
and [K Zhang et al.]() for making the preprocessed datasets publicly available and also [K. Zhou et al.](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce)
for code ```vsum_tools.py``` and ```create_split.py``` which we copied from [https://github.com/KaiyangZhou/pytorch-vsumm-reinforce](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) 
and slightly modified. 

This work is co-funded by the NATO within the WITNESS project under grant agreement number G5437.
We gratefully acknowledge the support of [NVIDIA Corporation](https://www.nvidia.com/en-gb/deep-learning-ai/solutions/) 
with the donation of the TITAN Xp GPU used for this research. 


## Cite
If you use this code or reference our paper in your work please cite this publication as:
```
@misc{fajtl2018summarizing,
    title={Summarizing Videos with Attention},
    author={Jiri Fajtl and Hajar Sadeghi Sokeh and Vasileios Argyriou and Dorothy Monekosso and Paolo Remagnino},
    year={2018},
    eprint={1812.01969},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

