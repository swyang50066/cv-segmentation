# Algorithmic Methods of Model-Based Medical Image Segmentation Using Python
![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)

## Introduction
This repository includes diverse algorithmic method of model-based medical image segmentation.  
The terminology of `model-based` means one which is hypothesized and parameterized model,  
so it is a bit free from the requirement of plenty of labeling data (the counter example is usually called data-driven method).  

The below list describes the classification of model-based method and the representative models belonging to it. 

* **Region-Based**
	* Region Growing
	* Grow-Cut
* **Active Contour**
	* Snake Model 
	* Gradient Vector Flow 
* **Level-Set**
	* Level-set Method 
	* Chan-Vese Model 
	* Morphological Chan-Vese Model 
	* Region Scalable Fitting Method (RSF)
	* Distance Regularized Level Set Evolution (DRLSE)
	* Kullback-Leibler-Based Level-set Method (***to be updated***)
* **Graph-Based**
	* Graph-Cut 
	* Simulated Annealing Method 
	* Random Walker with Prior Model 
	* Power Watershed Random Field (***to be updated***)
* **Clustering**
	* Gaussian Mixture Model 
	* Taboggan/Superpixel Clustering 

Currently, some methods are stiil in maintenance 

**GUI application for user-interactive medical image segmentation is now building**

## Installation
You can simply use functions in this repository by cloning repository,

~~~
{user}@{work-node}: {workplace}$ git clone https://github.com/swyang50066/segmentation.git
~~~

or using below command.

~~~
{user}@{work-node}: {workplace}$ sudo pip install git+https://github.com/swyang50066/segmentation.git 
~~~

## Usage
***To be updated***

## Contribution
***To be updated***

## Requirements
 All codes have developed in python and tested with `python>=3.6.0` environment (but, it stably supports functions with `python==3.8.0`). Numerical schemes (e.g., matrix operation, linear algebra, graph algorithm, ... etc.) used in the models is designed with  `numpy` and `scipy` modules for efficient computing. 
 
```
 requirements={
 	"python>=3.6.0",
 	"opencv-python>3.4.0",
 	"numpy>=1.18.0",
 	"scipy>=1.1.0",
 	"scikit-learn>=0.20.0",
 	"scikit-build>=0.12.0",
 	"opencv-python>3.4.0",
 	"PyMaxFlow>=1.2.0",
 }
```

## CAVEAT
***To be announced***

## References
[[1]](http://iacl.ece.jhu.edu/pubs/p087c.pdf) *C. Xu and J.L. Prince, "Gradient Vector Flow: A New External Force for Snakes," Proc. IEEE Conf. on Comp. Vis. Patt. Recog. (CVPR), Los Alamitos: Comp. Soc. Press, pp. 66–71, June 1997*

[[2]](https://www.graphicon.ru/oldgr/en/publications/text/gc2005vk.pdf)
*Vezhnevets, Vladimir and Vadim Konouchine. ““GrowCut”-Interactive Multi-Label N-D Image Segmentation By Cellular Automata.” (2005).* 

[[3]](https://www.math.ucla.edu/~lvese/PAPERS/IEEEIP2001.pdf) *T. F. Chan and L. A. Vese, "Active contours without edges," in IEEE Transactions on Image Processing, vol. 10, no. 2, pp. 266-277, Feb. 2001, doi: 10.1109/83.902291.*

[[4]](http://www.dia.fi.upm.es/~lbaumela/WEB/publications/pami2013.pdf) *A Morphological Approach to Curvature-based Evolution of Curves and Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez. In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI),2014, :DOI:`10.1109/TPAMI.2013.106.*

[[5]](https://www.researchgate.net/publication/3328985_Minimization_of_Region-Scalable_Fitting_Energy_for_Image_Segmentation) *C. Li, C. Kao, J. C. Gore and Z. Ding, "Minimization of Region-Scalable Fitting Energy for Image Segmentation," in IEEE Transactions on Image Processing, vol. 17, no. 10, pp. 1940-1949, Oct. 2008, doi: 10.1109/TIP.2008.2002304.*

[[6]](https://www.researchgate.net/publication/224169952_Distance_Regularized_Level_Set_Evolution_and_Its_Application_to_Image_Segmentation) *C. Li, C. Xu, C. Gui and M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to Image Segmentation," in IEEE Transactions on Image Processing, vol. 19, no. 12, pp. 3243-3254, Dec. 2010, doi: 10.1109/TIP.2010.2069690.* 

[[7]](https://www.researchgate.net/publication/324907041_Active_Contours_Based_Segmentation_and_Lesion_Periphery_Analysis_For_Characterization_of_Skin_Lesions_in_Dermoscopy_Images) *F. Riaz, S. Naeem, R. Nawaz, and M. Coimbra, “Active contours based segmentation and lesion periphery analysis for characterization of skin lesions in dermoscopy images,” IEEE journal of biomedical and health informatics, vol. 23, no. 2, pp. 489–500, 2018.*

[[8]](https://cs.uwaterloo.ca/~yboykov/Papers/emmcvpr01.pdf) *Y. Boykov and V. Kolmogorov, "An experimental comparison of min-cut/max- flow algorithms for energy minimization in vision," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 26, no. 9, pp. 1124-1137, Sept. 2004, doi: 10.1109/TPAMI.2004.60.*

[[9]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.118.6898&rep=rep1&type=pdf) *L. Grady, "Multilabel random walker image segmentation using prior models," 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), 2005, pp. 763-770 vol. 1, doi: 10.1109/CVPR.2005.239.*

[[10]](https://arxiv.org/abs/1606.00915) *L. -C. Chen, G. Papandreou, I. Kokkinos, K. Murphy and A. L. Yuille, "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 4, pp. 834-848, 1 April 2018, doi: 10.1109/TPAMI.2017.2699184.*

[[11]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.186.194&rep=rep1&type=pdf) *C. Couprie, L. Grady, L. Najman and H. Talbot, "Power Watershed: A Unifying Graph-Based Optimization Framework," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 7, pp. 1384-1399, July 2011, doi: 10.1109/TPAMI.2010.200.*

[[12]](http://eprints.bournemouth.ac.uk/30152/1/imageSeg-mmlbf.pdf) *Cheng, D., Tian, F., Liu, L. et al. Image segmentation based on multi-region multi-scale local binary fitting and Kullback–Leibler divergence. SIViP 12, 895–903 (2018).*

## License
[MIT License](./LICENSE)
