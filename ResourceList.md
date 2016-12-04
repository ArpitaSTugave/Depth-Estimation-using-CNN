<html>
<head>
<title>References</title>

<table id="qs_table" border="1">
<thead><tr><th width="3%">Index</th><th width="20%">Author</th><th width="30%">Title</th><th width="5%">Year</th><th width="27%">Journal/Proceedings</th><th width="10%">Reftype</th><th width="5%">DOI/URL</th></tr></thead>
<tbody>

<tr id="id1" class="parent">
     <td colspan="7">Topics</td>
</tr>

<tr id="id2" class="parent">
	<td>1 </td>
    <td colspan="6"> Database: </td>
</tr>

<tr id="stereo_database1" class="entry">
	<td>1. a</td>
	<td colspan="6"><a href="http://vision.middlebury.edu/stereo/data/scenes2014/"> Stereo Database 1</a> &nbsp;</td>
</tr>

<tr id="stereo_database2" class="entry">
	<td>1. b</td>
	<td colspan="6"><a href="https://github.com/LouisFoucard/DepthMap_dataset"> Stereo Database 2 created using blender</a> &nbsp;</td>
</tr>

<tr id="Scharstein" class="entry">
	<td>1. c</td>
	<td>Scharstein</td>
	<td>High-Resolution Stereo Datasets with Subpixel-Accurate Ground Truth <p class="infolinks">[<a href="javascript:toggleInfo('Scharstein','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('Scharstein','review')">Review</a>] [<a href="javascript:toggleInfo('Scharstein','bibtex')">BibTeX</a>]</p></td>
	<td>2014</td>
	<td>GCPR</td>
	<td>INPROCEEDINGS</td>
	<td><a href="http://www.cs.middlebury.edu/~schar/papers/datasets-gcpr2014.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_Scharstein" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
We present a structured lighting system for creating highresolution
stereo datasets of static indoor scenes with highly accurate
ground-truth disparities. The system includes novel techniques for effi-
cient 2D subpixel correspondence search and self-calibration of cameras
and projectors with modeling of lens distortion. Combining disparity
estimates from multiple projector positions we are able to achieve a disparity
accuracy of 0.2 pixels on most observed surfaces, including in halfoccluded
regions. We contribute 33 new 6-megapixel datasets obtained
with our system and demonstrate that they present new challenges for
the next generation of stereo algorithms
	</ul>
	</td>
</tr>
<tr id="rev_Scharstein" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>With bundle adjustment perfect images are produced. </li>
	  <li>Block diagram of how images are produced. (various techniques shown)</li>
	</ul></td>
</tr>
<tr id="bib_Scharstein" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
@inproceedings{conf/dagm/ScharsteinHKKNWW14,
  added-at = {2014-10-15T00:00:00.000+0200},
  author = {Scharstein, Daniel and Hirschmüller, Heiko and Kitajima, York and Krathwohl, Greg and Nesic, Nera and Wang, Xi and Westling, Porter},
  biburl = {http://www.bibsonomy.org/bibtex/2c6647ae773975aa5e48a8b29737478ab/dblp},
  booktitle = {GCPR},
  crossref = {conf/dagm/2014},
  editor = {Jiang, Xiaoyi and Hornegger, Joachim and Koch, Reinhard},
  ee = {http://dx.doi.org/10.1007/978-3-319-11752-2_3},
  interhash = {5474bf4e60bdf8fe8b595af37dfd7367},
  intrahash = {c6647ae773975aa5e48a8b29737478ab},
  isbn = {978-3-319-11751-5},
  keywords = {dblp},
  pages = {31-42},
  publisher = {Springer},
  series = {Lecture Notes in Computer Science},
  timestamp = {2015-06-18T22:44:59.000+0200},
  title = {High-Resolution Stereo Datasets with Subpixel-Accurate Ground Truth.},
  url = {http://dblp.uni-trier.de/db/conf/dagm/gcpr2014.html#ScharsteinHKKNWW14},
  volume = 8753,
  year = 2014
}
</pre>
</td>
</tr>

<tr id="id3" class="parent">
	<td>2 </td>
    <td colspan="6"> Deep Learning: </td>
</tr>

<tr id="Zeiler" class="entry">
	<td>2. a</td>
	<td>Eigen, David and Puhrsch, Christian and Fergus, Rob</td>
	<td>Depth Map Prediction from a Single Image using a Multi-Scale Deep Network <p class="infolinks">[<a href="javascript:toggleInfo('Zeiler','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('Zeiler','review')">Review</a>] [<a href="javascript:toggleInfo('Zeiler','bibtex')">BibTeX</a>]</p></td>
	<td>2014</td>
	<td>book</td>
	<td>incollection</td>
	<td><a href="https://arxiv.org/pdf/1406.2283v1.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_Zeiler" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
Predicting depth is an essential component in understanding the 3D geometry of
a scene. While for stereo images local correspondence suffices for estimation,
finding depth relations from a single image is less straightforward, requiring integration
of both global and local information from various cues. Moreover, the
task is inherently ambiguous, with a large source of uncertainty coming from the
overall scale. In this paper, we present a new method that addresses this task by
employing two deep network stacks: one that makes a coarse global prediction
based on the entire image, and another that refines this prediction locally. We also
apply a scale-invariant error to help measure depth relations rather than scale. By
leveraging the raw datasets as large sources of training data, our method achieves
state-of-the-art results on both NYU Depth and KITTI, and matches detailed depth
boundaries without the need for superpixelation.
	</ul>
	</td>
</tr>
<tr id="rev_Zeiler" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>Uses CNNs to extract local and global component in a very efficient and simple way. </li>
	</ul></td>
</tr>
<tr id="bib_Zeiler" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
@incollection{NIPS2014_5539,
title = {Depth Map Prediction from a Single Image using a Multi-Scale Deep Network},
author = {Eigen, David and Puhrsch, Christian and Fergus, Rob},
booktitle = {Advances in Neural Information Processing Systems 27},
editor = {Z. Ghahramani and M. Welling and C. Cortes and N. D. Lawrence and K. Q. Weinberger},
pages = {2366--2374},
year = {2014},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf}
}
</pre>
</td>
</tr>

<tr id="Fischer" class="entry">
	<td>2. b</td>
	<td>A. Dosovitskiy and P. Fischer</td>
	<td>FlowNet: Learning Optical Flow with Convolutional Networks <p class="infolinks">[<a href="javascript:toggleInfo('Fischer','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('Fischer','review')">Review</a>] [<a href="javascript:toggleInfo('Fischer','bibtex')">BibTeX</a>]</p></td>
	<td>2015</td>
	<td>ICCV</td>
	<td>InProceedings</td>
	<td><a href="https://arxiv.org/pdf/1504.06852v2.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_Fischer" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
Convolutional neural networks (CNNs) have recently
been very successful in a variety of computer vision tasks,
especially on those linked to recognition. Optical flow estimation
has not been among the tasks where CNNs were successful.
In this paper we construct appropriate CNNs which
are capable of solving the optical flow estimation problem
as a supervised learning task. We propose and compare
two architectures: a generic architecture and another one
including a layer that correlates feature vectors at different
image locations.
Since existing ground truth datasets are not sufficiently
large to train a CNN, we generate a synthetic Flying Chairs
dataset. We show that networks trained on this unrealistic
data still generalize very well to existing datasets such as
Sintel and KITTI, achieving competitive accuracy at frame
rates of 5 to 10 fps.

	</ul>
	</td>
</tr>
<tr id="rev_Fischer" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>Uses CNN for optical flow. </li>
	</ul></td>
</tr>
<tr id="bib_Fischer" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
@InProceedings{DFIB15,
  author       = "A. Dosovitskiy and P. Fischer and E. Ilg and P. H{\"a}usser and C. Haz\ırba\ş and V. Golkov and P. v.d. Smagt and D. Cremers and T. Brox",
  title        = "FlowNet: Learning Optical Flow with Convolutional Networks",
  booktitle    = "IEEE International Conference on Computer Vision (ICCV)",
  month        = "Dec",
  year         = "2015",
  url          = "http://lmb.informatik.uni-freiburg.de//Publications/2015/DFIB15"
}
</pre>
</td>
</tr>

<tr id="Shen" class="entry">
	<td>2. c</td>
	<td>B. Li and  C. Shen and  Y. Dai</td>
	<td>Depth and surface normal estimation from monocular images using regression on deep features and hierarchical {CRFs}<p class="infolinks">[<a href="javascript:toggleInfo('Shen','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('Shen','review')">Review</a>] [<a href="javascript:toggleInfo('Shen','bibtex')">BibTeX</a>]</p></td>
	<td>2015</td>
	<td>CVPR</td>
	<td>InProceedings</td>
	<td><a href="https://arxiv.org/pdf/1504.06852v2.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_Shen" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
Convolutional neural networks (CNNs) have recently
been very successful in a variety of computer vision tasks,
especially on those linked to recognition. Optical flow estimation
has not been among the tasks where CNNs were successful.
In this paper we construct appropriate CNNs which
are capable of solving the optical flow estimation problem
as a supervised learning task. We propose and compare
two architectures: a generic architecture and another one
including a layer that correlates feature vectors at different
image locations.
Since existing ground truth datasets are not sufficiently
large to train a CNN, we generate a synthetic Flying Chairs
dataset. We show that networks trained on this unrealistic
data still generalize very well to existing datasets such as
Sintel and KITTI, achieving competitive accuracy at frame
rates of 5 to 10 fps.

	</ul>
	</td>
</tr>
<tr id="rev_Shen" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>Uses CRFs for depth estimation. </li>
	</ul></td>
</tr>
<tr id="bib_Shen" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
 @inproceedings{CVPR15h,
   author    = "B. Li and  C. Shen and  Y. Dai and  A. {van den Hengel} and  M. He",
   title     = "Depth and surface normal estimation from monocular images using regression on deep features and hierarchical {CRFs}",
   booktitle = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR'15)",
   year      = "2015",
 }
</pre>
</td>
</tr>

<tr id="LeCun" class="entry">
	<td>2. d</td>
	<td>Jure Zbontar and Yann LeCun</td>
	<td>Stereo Matching by Training a Convolutional Neural Network to Compare
       <br>Image Patches<p class="infolinks">[<a href="javascript:toggleInfo('LeCun','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('LeCun','review')">Review</a>] [<a href="javascript:toggleInfo('LeCun','bibtex')">BibTeX</a>]</p></td>
	<td>2016</td>
	<td>CoRR</td>
	<td>article</td>
	<td><a href="https://arxiv.org/pdf/1510.05970v2.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_LeCun" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
We present a method for extracting depth information from a rectified image pair. Our
approach focuses on the first stage of many stereo algorithms: the matching cost computation.
We approach the problem by learning a similarity measure on small image patches
using a convolutional neural network. Training is carried out in a supervised manner by
constructing a binary classification data set with examples of similar and dissimilar pairs
of patches. We examine two network architectures for this task: one tuned for speed, the
other for accuracy. The output of the convolutional neural network is used to initialize the
stereo matching cost. A series of post-processing steps follow: cross-based cost aggregation,
semiglobal matching, a left-right consistency check, subpixel enhancement, a median
filter, and a bilateral filter. We evaluate our method on the KITTI 2012, KITTI 2015, and
Middlebury stereo data sets and show that it outperforms other approaches on all three
data sets.
	</ul>
	</td>
</tr>
<tr id="rev_LeCun" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>Uses image patches. </li>
	</ul></td>
</tr>
<tr id="bib_LeCun" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
@article{DBLP:journals/corr/ZbontarL15,
  author    = {Jure Zbontar and
               Yann LeCun},
  title     = {Stereo Matching by Training a Convolutional Neural Network to Compare
               Image Patches},
  journal   = {CoRR},
  volume    = {abs/1510.05970},
  year      = {2015},
  url       = {http://arxiv.org/abs/1510.05970},
  timestamp = {Sun, 01 Nov 2015 17:30:45 +0100},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/ZbontarL15},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
</pre>
</td>
</tr>

<tr id="id4" class="parent">
	<td>3 </td>
    <td colspan="6"> Related papers: </td>
</tr>

<tr id="Zeiler" class="entry">
	<td>3. a</td>
	<td>Matthew D. Zeiler and Rob Fergus</td>
	<td>Visualizing and Understanding Convolutional Networks <p class="infolinks">[<a href="javascript:toggleInfo('Zeiler','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('Zeiler','review')">Review</a>] [<a href="javascript:toggleInfo('Zeiler','bibtex')">BibTeX</a>]</p></td>
	<td>2014</td>
	<td>CoRR</td>
	<td>article</td>
	<td><a href="http://www.matthewzeiler.com/pubs/arxive2013/arxive2013.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_Zeiler" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
Large Convolutional Network models have
recently demonstrated impressive classification
performance on the ImageNet benchmark
(Krizhevsky et al., 2012). However
there is no clear understanding of why they
perform so well, or how they might be improved.
In this paper we address both issues.
We introduce a novel visualization technique
that gives insight into the function of intermediate
feature layers and the operation of
the classifier. Used in a diagnostic role, these
visualizations allow us to find model architectures
that outperform Krizhevsky et al. on
the ImageNet classification benchmark. We
also perform an ablation study to discover
the performance contribution from different
model layers. We show our ImageNet model
generalizes well to other datasets: when the
softmax classifier is retrained, it convincingly
beats the current state-of-the-art results on
Caltech-101 and Caltech-256 datasets.

	</ul>
	</td>
</tr>
<tr id="rev_Zeiler" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>With bundle adjustment perfect images are produced. </li>
	  <li>Block diagram of how images are produced. (various techniques shown)</li>
	</ul></td>
</tr>
<tr id="bib_Zeiler" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
@article{DBLP:journals/corr/ZeilerF13,
  author    = {Matthew D. Zeiler and
               Rob Fergus},
  title     = {Visualizing and Understanding Convolutional Networks},
  journal   = {CoRR},
  volume    = {abs/1311.2901},
  year      = {2013},
  url       = {http://arxiv.org/abs/1311.2901},
  timestamp = {Tue, 03 Dec 2013 15:04:22 +0100},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/ZeilerF13},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
</pre>
</td>
</tr>

<tr id="Andrew" class="entry">
	<td>3. b</td>
	<td>Ashutosh Saxena, Sung H. Chung, and Andrew Y. Ng</td>
	<td>Learning Depth from Single Monocular Images <p class="infolinks">[<a href="javascript:toggleInfo('Andrew','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('Andrew','review')">Review</a>] [<a href="javascript:toggleInfo('Andrew','bibtex')">BibTeX</a>]</p></td>
	<td>2005</td>
	<td>NIPS</td>
	<td>INPROCEEDINGS</td>
	<td><a href="https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/saxena-nips-05.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_Andrew" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
We consider the task of depth estimation from a single monocular image.
We take a supervised learning approach to this problem, in which
we begin by collecting a training set of monocular images (of unstructured
outdoor environments which include forests, trees, buildings, etc.)
and their corresponding ground-truth depthmaps. Then, we apply supervised
learning to predict the depthmap as a function of the image.
Depth estimation is a challenging problem, since local features alone are
insufficient to estimate depth at a point, and one needs to consider the
global context of the image. Our model uses a discriminatively-trained
Markov Random Field (MRF) that incorporates multiscale local- and
global-image features, and models both depths at individual points as
well as the relation between depths at different points. We show that,
even on unstructured scenes, our algorithm is frequently able to recover
fairly accurate depthmaps.

	</ul>
	</td>
</tr>
<tr id="rev_Andrew" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>Shows first level convolutional kernels. </li>
	</ul></td>
</tr>
<tr id="bib_Andrew" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
@INPROCEEDINGS{Saxena05learningdepth,
    author = {Ashutosh Saxena and Sung H. Chung and Andrew Y. Ng},
    title = {Learning depth from single monocular images},
    booktitle = {In NIPS 18},
    year = {2005},
    publisher = {MIT Press}
}
</pre>
</td>
</tr>

<tr id="Ioffe" class="entry">
	<td>3. c</td>
	<td> Sergey Ioffe and Christian Szegedy </td>
	<td>Batch Normalization: Accelerating Deep Network Training by
	<br>Reducing Internal Covariate Shift <p class="infolinks">[<a href="javascript:toggleInfo('Ioffe','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('Ioffe','review')">Review</a>] [<a href="javascript:toggleInfo('Ioffe','bibtex')">BibTeX</a>]</p></td>
	<td>2015</td>
	<td>COrR</td>
	<td>article</td>
	<td><a href="https://arxiv.org/pdf/1502.03167v3.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_Ioffe" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
Training Deep Neural Networks is complicated by the fact
that the distribution of each layer’s inputs changes during
training, as the parameters of the previous layers change.
This slows down the training by requiring lower learning
rates and careful parameter initialization, and makes it no
toriously hard to train models with saturating nonlinearities.
We refer to this phenomenon as internal covariate
shift, and address the problem by normalizing layer inputs.
Our method draws its strength from making normalization
a part of the model architecture and performing the
normalization for each training mini-batch. Batch Normalization
allows us to use much higher learning rates and
be less careful about initialization. It also acts as a regularizer,
in some cases eliminating the need for Dropout.
Applied to a state-of-the-art image classification model,
Batch Normalization achieves the same accuracy with 14
times fewer training steps, and beats the original model
by a significant margin. Using an ensemble of batchnormalized
networks, we improve upon the best published
result on ImageNet classification: reaching 4.9% top-5
validation error (and 4.8% test error), exceeding the accuracy
of human raters.

	</ul>
	</td>
</tr>
<tr id="rev_Ioffe" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>Batch normalization as used by the code. </li>
	</ul></td>
</tr>
<tr id="bib_Ioffe" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
@article{DBLP:journals/corr/IoffeS15,
  author    = {Sergey Ioffe and
               Christian Szegedy},
  title     = {Batch Normalization: Accelerating Deep Network Training by Reducing
               Internal Covariate Shift},
  journal   = {CoRR},
  volume    = {abs/1502.03167},
  year      = {2015},
  url       = {http://arxiv.org/abs/1502.03167},
  timestamp = {Mon, 02 Mar 2015 14:17:34 +0100},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/IoffeS15},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
</pre>
</td>
</tr>


<tr id="id4" class="parent">
	<td>4 </td>
    <td colspan="6"> Links: </td>
</tr>

<tr id="deep-learning-code" class="parent">
	<td>4. a</td>
    <td colspan="6"><a href="https://github.com/LouisFoucard/StereoConvNet"> CNN code for depth estimation</a> &nbsp;</td>
</tr>

</tbody>
</table>