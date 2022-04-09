ASFS
======
**This is an implementation of  Hyperspectral Image Super-Resolution via Adjacent Spectral Fusion Strategy.**

Motivation
=======
**The hyperspectral image has the remarkable characteristics of high similarity between adjacent bands. When reconstructing the current band, if the adjacent bands are employed effectively, the complementary information would be beneficial to recover more missing details. Besides,  in a certain spectral range, the sharpness of the edge in the image varies with the bands. It indicates the information of different bands complements each other.**

Flowchat
=====
![Image text](https://github.com/qianngli/Images/blob/master/asfs.jpg)

**The network mainly contains feature extraction (FE) and image reconstruction (IR). Inspired by the high similarity among adjacent bands, neighboring band partition is proposed to divide the adjacent bands into several groups. Through the current  band,  the  adjacent bands is guided to enhance the exploration ability. To explore more complementary information, an alternative fusion mechanism, i.e., intra-group fusion and inter-group fusion, is designed, which helps to recover the missing details in the current band.**

Dataset
------
**Three public datasets, i.e., [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ "CAVE"), [Harvard](http://vision.seas.harvard.edu/hyperspec/explore.html "Harvard"), [Foster](https://personalpages.manchester.ac.uk/staff/d.h.foster/Local\_Illumination\_HSIs/Local\_Illumination\_HSIs\_2015.html "Foster"), are employed to verify the effectiveness of the  proposed MCNet. Since there are too few images in these datasets for deep learning algorithm, we augment the training data. With respect to the specific details, please see the implementation details section.**

**Moreover, we also provide the code about data pre-processing in folder [data pre-processing](https://github.com/qianngli/MCNet "data pre-processing"). The folder contains three parts, including training set augment, test set pre-processing, and band mean for all training set.**

Requirement
---------
**python 2.7, Pytorch 0.3.1, cuda 9.0**

Training
--------
**The ADAM optimizer with beta_1 = 0.9, beta _2 = 0.999 is employed to train our network.  The learning rate is initialized as 10^-4 for all layers, which decreases by a half at every 35 epochs.**

**You can train or test directly from the command line as such:**

###### # python train.py --cuda --datasetName CAVE  --upscale_factor 4
###### # python test.py --cuda --model_name checkpoint/model_4_epoch_xx.pth

Citation 
--------
**Please consider cite ASFS for hyperspectral image super-resolution if you find it helpful.**

@article{Li2021Hyper,

	title={Hyperspectral Image Super-Resolution via Adjacent Spectral Fusion Strategy},
	author={Q. Li and Q. Wang and X. Li},
	journal={International Conference on Acoustic, Speech, and Signal Processing},
	year={2021}}
  
--------
If you has any questions, please send e-mail to liqmges@gmail.com.

