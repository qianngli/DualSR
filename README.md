<div align="justify">
  <div align="center">
    
  # [Dual-Stage Approach Toward Hyperspectral Image Super-Resolution](https://ieeexplore.ieee.org/document/9953047 "Dual-Stage Approach Toward Hyperspectral Image Super-Resolution")  
 
  </div>

## Update
**[2022-04-09]** DualSR v0.1 is modified.  

## Abstract  
![Image text](https://raw.githubusercontent.com/qianngli/Images/master/DualSR/architecture.png)  
Hyperspectral image produces high spectral resolution at the sacrifice of spatial resolution. Without reducing the spectral resolution, improving the resolution in the spatial domain is a very challenging problem. Motivated by the discovery that hyperspectral image exhibits high similarity between adjacent bands in a large spectral range, in this paper, we explore a new structure for hyperspectral image super-resolution (DualSR), leading to a dual-stage design, i.e., coarse stage and fine stage. In coarse stage, five bands with high similarity in a certain spectral range are divided into three groups, and the current band is guided to study the potential knowledge. Under the action of alternative spectral fusion mechanism, the coarse SR image is super-resolved in band-by-band. In order to build model from a global perspective, an enhanced back-projection method via spectral angle constraint is developed in fine stage to learn the content of spatial-spectral consistency, dramatically improving the performance gain. Extensive experiments demonstrate the effectiveness of the proposed coarse stage and fine stage. Besides, our network produces state-of-the-art results against existing works in terms of spatial reconstruction and spectral fidelity.  

## Motivation  
Hyperspectral image produces high spectral resolution at the sacrifice of spatial resolution, which cannot meet the requirements of some scene applications. Considering this dilemma, the researchers propose hyperspectral image super-resolution (SR). Without reducing the number of bands, hyperspectral image SR aims to find a high-resolution (HR) image with better visual quality and refined details from counterpart low-resolution (LR) version in spatial domain. SR is a greatly challenging task in computer vision, because it is an ill-posed inverse problem.  
- Single hyperspectral image SR without any auxiliary information solely focuses on spatial knowledge, leading to inferior spectral fidelity.  
- Various hyperspectral image SR algorithms based on 3D convolution ignore the differential treatment of spectral and spatial domain analysis.  
- Inspired by high similarity between adjacent bands, Wang et al. create a novel dual-channel structure to establish network. Impressively, unlike existing methods, it integrates the information of single LR band and two adjacent LR bands to achieve super-resolved band. At present, there is extremely little research using this novel input mode.  
- In fact, within a certain spectral range, relatively distant bands can also explicitly assist the current band to reconstruction, because these bands are also similar, but the similarity is relatively small. If more adjacent bands within a relatively large spectral range are utilized, it is beneficial to supplement the missing knowledge during the reconstruction of current band. Therefore, the key problem is how to effectively use the adjacent bands to boost performance.  

## Architecture
Our method consists of two steps: coarse stage and fine stage. In coarse stage, the coarse hyperspectral image is restored in band-by-band using supervised way. Note that the method is our [conference version](https://ieeexplore.ieee.org/document/9413980). For ease of description, the method is named CoarSR. During this process, the current band is assisted to enhance the exploration ability through four adjacent bands. After obtaining coarse result, we adopt unsupervised manner in fine stage to globally learn the information, which further optimizes result.  

### Coarse stage
We propose a novel structure for hyperspectral image SR via adjacent spectral fusion, whose flowchart is shown in Fig. 1. The overview of coarse stage mainly covers three modules, involving neighboring band partition (NBP), adjacent spectral fusion mechanism (ASFM), and feature context fusion (FCF).

<div align="center">
  
  ![CoarSR](https://raw.githubusercontent.com/qianngli/Images/master/DualSR/CoarSR.png)  
  
</div>

 *Fig. 1. Overview of the proposed CoarSR for hyperspectral image SR in coarse stage.*
 
- **Neighboring Band Partition (NBP)** groups a target band with its adjacent bands to enhance the relevance and utilization of band information, optimizing the super-resolution reconstruction process.
- **Adjacent Spectral Fusion Mechanism (ASFM)** enhances image quality in hyperspectral image super-resolution by fusing information within and between groups, strengthening the integration of spatial and spectral data.
- **Feature Context Fusion (FCF)** module enhances inter-band consistency and feature expression by fusing features of consecutive bands, akin to the operation of Recurrent Neural Networks (RNNs).

### Fine stage
Back-projection optimizes the reconstruction error through an efficient iterative strategy. For this algorithm, it usually utilizes multiple upsampling descriptors to upsample LR image and iteratively calculate the reconstruction error. Currently, back-projection is widely introduced in natural image SR which has been proved to develop the quality of SR image. However, the initialization which leads to an optimal solution remains unknown. The main reason is that the algorithm involves predefined hyperparameters, such as number of iteration and convolution kernel.To extend this algorithm, we further develop back-projection without predefined hyperparameters, which is shown in Fig. 2. 

<div align="center">
  
  ![Enhanced back-projection](https://raw.githubusercontent.com/qianngli/Images/master/DualSR/Enhanced_back_projection.png)  
 
</div>

*Fig. 2. Enhanced back-projection method via spectral angle constraint.*

## Algorithm  

*Dual-Stage Hyperspectral Image SR Algorithm (DualSR)*

> **Input:** Hyperspectral image dataset containg LR-HR pair, scale factor $s$  
> **Output:** Super-resolved hyperspectral image $I_{SR}$  
> Randomly initialize coarse model parameters $\theta$ ;  
> **while** not *converged* **do**  
> &ensp;&ensp; Sample LR-HR batch ;  
> &ensp;&ensp; **while** $i≤L$ **do**  
> &ensp;&ensp;&ensp;&ensp; Partiton bands into three groups ;  
> &ensp;&ensp;&ensp;&ensp; Update $\theta$ by excuting coarse model ;  
> &ensp;&ensp;&ensp;&ensp; $i \gets i+1$;  
> &ensp;&ensp; **end**  
> **end** 
> Generate coarse model parameters $\theta_{c}$ ;  
> Obtain coarse SR results $U$ and $V$ in terms of scale factor $s$ ;  
> Compute the reconstruction error under spectral angle constrain  
> Obtain fine SR result $I_{SR}$ using  
  


## Dependencies  
**PyTorch, MATLAB, NVIDIA GeForce GTX 1080 GPU.**
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard: 
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`

## Dataset Preparation 
Two public datasets, i.e., [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ "CAVE") and [Harvard](https://dataverse.harvard.edu/ "Harvard") are employed to verify the effectiveness of the proposed DualSR.  

- In our work, we randomly select **80%** of the data as the training set and the rest for testing.  
- We augment the given training data by choosing **24** patches. With respect to each patch, its size is scaled **1**, **0.75**, and **0.5** times, respectively. We rotate these patches **90°** and flip them horizontally. Through various blur kernels, we then subsample these patches into LR hyperspectral images with the size of **L × 32 × 32**.  

## Implementation  
- For our network, the convolution kernel after concatenation is set to **1 × 1**, which reduces the number of channels.  
- We adopt sub-pixel convolution layer to upscale the features into HR space in terms of upsampling operation.  
- The kernel of other convolution operations involved in the network is fixed to **3 × 3**, and the number of convolution kernels is defined as **64**.  

        parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
  
- In the training phase, our network is trained using **L1** loss function. The mini-batch is set to **64**.  
- We optimize our network using **ADAM** optimizer with **β1=0.9** and **β2=0.999** and initial learning rate **10^−4**.  

        parser.add_argument("--lr", type=int, default=1e-4, help="lerning rate")
  
- For learning rate, it is gradually updated by a half at every **30** epochs.  

## Result  
- To quantitatively evaluate the proposed method, we apply Peak Signal-to-Noise Ratio (**PSNR**), Structural SIMilarity (**SSIM**), and Spectral Angle Mapper (**SAM**). Among these metrics, PSNR and SSIM are to evaluate the performance of super-resolved hyperspectral image in spatial domain. Generally, the higher their values are, the better the performance is. SAM is to analyze the performance of restored image in spectral domain. The smaller the value is, the less the spectral distortion is.
- Using known bicubic downsampling condition, we compare our proposed DualSR with existing multiple approaches on CAVE and Harvard datasets, including **3D-FCNN**, **EDSR**, **SSPSR**, **MCNet**, **SFCSR**, **ERCSR**.  
### Quantitative Evaluation

<div align="center">

  ![TABLE_VI](https://raw.githubusercontent.com/qianngli/Images/master/DualSR/TABLE_VI.png)  
  ![TABLE_VII](https://raw.githubusercontent.com/qianngli/Images/master/DualSR/TABLE_VII.png)  

</div>

### Qualitative Evaluation

<div align="center">
      
  ![Fig8](https://raw.githubusercontent.com/qianngli/Images/master/DualSR/Fig8.png)

</div>

*Fig. 3. Visual results in terms of spatial domain with existing SR methods on CAVE dataset. The results of balloons image are evaluated for scale factor × 4. The first line denotes SR results of 10-th band, and the second line denotes SR results of 20-th band.*  
  
<div align="center">
   
  ![Fig9](https://raw.githubusercontent.com/qianngli/Images/master/DualSR/Fig9.png)  
    
</div>

*Fig. 4. Visual results in terms of spatial domain with existing SR methods on Harvard dataset. The results of imgd5 image are evaluated for scale factor × 4. The first line denotes SR results of 10-th band, and the second line denotes SR results of 20-th band. One observe that our method produces low absolute errors. In particular, there are more shallow edges in some positions, which indicates that the proposed approach can generate sharper edges and finer details. It is consistent with the analysis in Tables VI and VII, which further demonstrates that our approach can simultaneously learn spectral and spatial knowledge while generating diverse textures.*  
  
<div align="center">

  ![Fig10](https://raw.githubusercontent.com/qianngli/Images/master/DualSR/Fig10.png)  
    
</div>

*Fig. 5. Visual comparison in terms of spectral domain by randomly selecting two pixels for scale factor × 4. The two on the left are the results of balloons image on CAVE dataset. The two on the right are the results of imgd5 image on Harvard dataset. Note that to avoid confusion, only two representative algorithms are compared with our methods.We can see that our DualSR maintains the same curve as the ground-truth in most cases. It validates that the proposed method can yield higher spectral fidelity against other approaches.*  

## Citation 
[1] **Q. Li**, Q. Wang, and X. Li, “Exploring the Relationship Between 2D/3D Convolution for Hyperspectral Image Super-Resolution,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 59, no. 10, pp. 8693-8703, 2021.  
[2] Q. Wang, **Q. Li**, and X. Li, “Hyperspectral Image Superresolution Using Spectrum and Feature Context,” *IEEE Transactions on Industrial Electronics*, vol. 68, no. 11, pp. 11276-11285, 2021.  
[3] **Q. Li**, Q. Wang, and X. Li, “Mixed 2D/3D convolutional network for hyperspectral image super-resolution,” *Remote Sensing*, vol. 12, no. 10, pp. 1660, 2020.  
[4] S. Jia, G. Tang, J. Zhu, and **Q. Li**, “A Novel Ranking-Based Clustering Approach for Hyperspectral Band Selection,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 54, no. 1, pp. 88-102, 2016.  
[5] **Q. Li**, Q. Wang and X. Li, “Hyperspectral Image Super-Resolution Via Adjacent Spectral Fusion Strategy,” *Proc. IEEE International Conference on Acoustics*, pp. 1645-1649, 2021.  

--------
If you has any questions, please send e-mail to liqmges@gmail.com.

</div>
