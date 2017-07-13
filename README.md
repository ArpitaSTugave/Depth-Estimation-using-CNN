<BODY>

<DIV id="id_1">
<H1> Depth Estimation using Data Driven Approaches</H1>
</DIV>
<DIV id="id_2_1">
<P class="p9 ft6"><H2> Introduction </H2></P>
<P class="p10 ft6">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time of Flight, Structured light and Stereo technology have been used widely for 
Depth Map estimation. Each have these come with their own pros and cons in terms of speed of image capture, structural description and 
ambient light performance. Monocular cues such as: Texture and Gradient Variation, Shading , color/Haze, and defocus aid in accurate 
depth estimation. These are complex statistical models which are susceptible to noise. Recently, data driven approaches as in deep learning 
has been employed for depth estimation. These data driven approaches are less prone to noise if presented with enough data to learn coarser
and finer details.
</P>
</DIV>
<DIV id="id_2_2">
<P class="p9 ft6"><H2> Convolution Neural Networks - CNN </H2></P>
<P class="p12 ft6">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In deep learning, CNNs are widely used in the image processing applications. Convolution layers are the basic building block of CNN and it combines with Pooling and ReLU activation layers. Kernel learns during each layer using back propagation.The CNN learns the features from the input images by applying the varied filters across the image generating feature maps at each layer. As we go deeper into the network the feature maps are able to identify complex features and objects intuitively. ConvNets have been very successful for image classification, but recently have been used for image prediction and other applications. The addition of upscaling and deconvolution layers have given way to upscale the compressed feature map for data prediction over class.
</P>
</DIV>
<p><a href="https://cloud.githubusercontent.com/assets/11435669/20927466/c186f656-bb8f-11e6-86a8-2d6661db827c.png" target="_blank"><img src="https://cloud.githubusercontent.com/assets/11435669/20927466/c186f656-bb8f-11e6-86a8-2d6661db827c.png" alt="image" style="max-width:100%;"></a></p>
<DIV id="id_2_3">
<P class="p14 ft6"><H2> Related Work </H2></P>
<P class="p15 ft6">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A fully automatic 2D-to-3D conversion algorithm: Deep3D [1] that takes 2D images or video frames as input and outputs 3D stereo image pairs. David Eigen from NYU proposed a single monocular image based architecture that employs two deep network stacks 
called Multi Scale Network [2]: one that makes a coarse global prediction based on the entire image, and another that refines this prediction locally. It is trained on real world dataset. “FlowNet: Learning Optical Flow with Convolutional Networks” [3] uses video created virtually to make the network learn motion parameters and hence forth extract optical flow. “Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches” [4] a method for extracting depth information from stereo data and their respective patches. Similar to [4] “Depth and surface normal estimation from monocular images using regression on deep features and hierarchical {CRFs}” [5] uses different scale of image patches to extract depth information. 
</P>
</DIV>
<p><a href="https://cloud.githubusercontent.com/assets/11435669/20927710/c855abca-bb90-11e6-9dd1-3fe86007c398.png" target="_blank"><img src="https://cloud.githubusercontent.com/assets/11435669/20927710/c855abca-bb90-11e6-9dd1-3fe86007c398.png" alt="image" style="max-width:100%;"></a></p>
<P class="p15 ft6"> Multi Scale network </P>
<p><a href="https://cloud.githubusercontent.com/assets/11435669/20927750/f034fe20-bb90-11e6-9cb8-262d661d205a.png" target="_blank"><img src="https://cloud.githubusercontent.com/assets/11435669/20927750/f034fe20-bb90-11e6-9cb8-262d661d205a.png" alt="image" style="max-width:100%;"></a></p>
<p><a href="https://cloud.githubusercontent.com/assets/11435669/20927757/f4e5f3b6-bb90-11e6-91c3-ba2bf66dacb0.png" target="_blank"><img src="https://cloud.githubusercontent.com/assets/11435669/20927757/f4e5f3b6-bb90-11e6-91c3-ba2bf66dacb0.png" alt="image" style="max-width:100%;"></a></p>
<P class="p15 ft6"> FlowNet </P>
<DIV id="id_2_4">
<P class="p20 ft6"><H2> Methods </H2></P>
</DIV>

<DIV id="id_1_2">
<P class="p22 ft10"><SPAN class="ft10"><H3> A.&nbsp;&nbsp; Stereo ConvNet Architecture </H3></P>
<P class="p23 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The images and ground truth depth maps used for training, validation and testing are produced by varying orientations of the 3D model generated using the Blender software tool. As our first step, we use SteroConvNet [6] and the first half of the network is shown below. Second half of the network is the mirror image of the last convolution layer, replacing convolution with deconvolution and pooling with upscaling. Input Image, even though consists of concatenated left and right image pairs , the network takes it as two separate images. Here, the reference output label is the ground truth depth map generated using the Blender's "Mist" function. 
</P>
</DIV>
<p><a href="https://cloud.githubusercontent.com/assets/11435669/20928225/cc7f1b58-bb92-11e6-9217-fa0811db36bd.png" target="_blank"><img src="https://cloud.githubusercontent.com/assets/11435669/20928225/cc7f1b58-bb92-11e6-9217-fa0811db36bd.png" alt="image" style="max-width:100%;"></a></p>
<P class="p15 ft6"> Stereo ConvNet Architecture </P>
<DIV id="id_1_3">
<P class="p80 ft10"><SPAN class="ft10"><H3> B.&nbsp;&nbsp; Deeper Stereo ConvNet Architecture </H3></P>
<P class="p79 ft24">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In Deeper Stereo ConvNet, input remains constant but architecture is modified with an extra convolution and deconvolution layer. Also, depth of the filters is increased referring to [3] in order to capture more details.
</P>
</DIV>

<DIV id="id_1">
<P class="p80 ft10"><SPAN class="ft10"><H3> C.&nbsp;&nbsp; Patched Deeper Stereo ConvNet Architecture </H3></P>
<P class="p90 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Referring to [4] and [5], input stream has been increased to 6 for Patched Deeper Stereo ConvNet, by decomposing left image into 4 scaled parts. Thus, as in the referenced papers higher accuracy of the depth map is expected. 
</P>
</DIV>
<p><a href="https://cloud.githubusercontent.com/assets/11435669/20930304/f7fd45c2-bb9a-11e6-866e-be18af9e450b.png" target="_blank"><img src="https://cloud.githubusercontent.com/assets/11435669/20930304/f7fd45c2-bb9a-11e6-866e-be18af9e450b.png" alt="image" style="max-width:100%;"></a></p>
<P class="p15 ft6"> Patched Deeper Stereo ConvNet Architecture </P>
<DIV id="id_1">
<P class="p80 ft10"><SPAN class="ft10"><H2> Results </H2></P>
<P class="p117 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Stereo ConvNet Architecture
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ smooth without holes
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ coarse structure preserved
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Blurred at edges
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Sharp structures lost
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Fine objects smeared or lost.
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time to test = 20 s
</P>
</DIV>

<DIV id="id_1">
<P class="p117 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Deeper Stereo ConvNet Architecture
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ smooth without holes
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ coarse structure preserved
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Edges are sharper
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Still noise at the edges
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Fine details/objects smeared or lost.
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Note:The increased depth of the network learns more detail about the scene.
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time to test = 70 s
</P>
</DIV>

<DIV id="id_1">
<P class="p117 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Patched Deeper Stereo ConvNet Architecture
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ smooth without holes
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Fine structure preserved
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Image predicted with less noise. 
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Time to train and test increases.
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Note:The increased depth and increased data resolution of the network learns more <br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail about the scene.
<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time to test = 145 s
</P>
</DIV>

<P class="p15 ft6"><H3> Stereo ConvNet Architecture: </H3></P>
<p><a href="https://cloud.githubusercontent.com/assets/11435669/20932317/3d7a65d8-bba2-11e6-90d0-0589dc66ccee.png" target="_blank"><img src="https://cloud.githubusercontent.com/assets/11435669/20932317/3d7a65d8-bba2-11e6-90d0-0589dc66ccee.png" alt="image" style="max-width:100%;"></a></p>
<P class="p15 ft6"><H3> Deeper Stereo ConvNet Architecture: </H3></P>
<p><a href="https://cloud.githubusercontent.com/assets/11435669/20932408/8b2d6d0c-bba2-11e6-977c-10a82ce5e9aa.png" target="_blank"><img src="https://cloud.githubusercontent.com/assets/11435669/20932408/8b2d6d0c-bba2-11e6-977c-10a82ce5e9aa.png" alt="image" style="max-width:100%;"></a></p>
<P class="p15 ft6"><H3> Patched Deeper Stereo ConvNet Architecture: </H3></P>
<p><a href="https://cloud.githubusercontent.com/assets/11435669/20932444/a7e2b2ae-bba2-11e6-8bfa-d8b2e5dca250.png" target="_blank"><img src="https://cloud.githubusercontent.com/assets/11435669/20932444/a7e2b2ae-bba2-11e6-8bfa-d8b2e5dca250.png" alt="image" style="max-width:100%;"></a></p>

<P class="p15 ft6"> <H3> 3D modeling for Patched Deeper Stereo ConvNet Architecture: </H3> </P>

| Image | Expected output | Derived output |
| ------------- | ------------- | ------------- |
| ![1_s](https://cloud.githubusercontent.com/assets/11435669/20933414/f20f8d4a-bba5-11e6-98a3-849b483ea88f.PNG)  | ![2_s](https://cloud.githubusercontent.com/assets/11435669/20934331/bd2196ac-bba8-11e6-83c9-89f051d5d19f.gif)  | ![3_s](https://cloud.githubusercontent.com/assets/11435669/20934437/1e552ed4-bba9-11e6-932c-1a5c31ef9755.gif)  |
| ![4_s](https://cloud.githubusercontent.com/assets/11435669/20934176/396336e0-bba8-11e6-813f-490800551b6c.PNG)  | ![5_s](https://cloud.githubusercontent.com/assets/11435669/20934515/674ab582-bba9-11e6-998d-36f8ccffe2a2.gif) | ![6_s](https://cloud.githubusercontent.com/assets/11435669/20934002/c49369d4-bba7-11e6-9cbd-6be921976ee2.gif) |


<DIV id="id_1">
<P class="p80 ft10"><SPAN class="ft10"><H2> Conclusion </H2></P>
<P class="p131 ft9">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Data Driven Depth Estimation approaches would be effective if sufficiently large descriptive labelled dataset were avialable. Patched Deeper Stereo ConvNet predicts depth map very similar to the ground truth. Time to train the network is directly proportional to the depth and complexity of the CNN architecture. In further implementations, we plan to combine the architecture of our Patched Deeper StereoConvNet with Multi-Scale Deep Network and observe the results for real world images.
</P>
</DIV>

<DIV id="id_1_2">
<P class="p80 ft10"><SPAN class="ft10"><H2> References </H2></P>
<P class="p136 ft59"><SPAN class="ft29">[1]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</SPAN>“Deep3D: Fully Automatic 2D-to-3D Video Conversion with Deep Convolutional Neural Networks” Junyuan Xie, Ross Girshick, Ali Farhadi,University of Washington. </P>
<P class="p137 ft59"><SPAN class="ft29">[2]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</SPAN>“Depth Map Prediction from a Single Image using a Multi-Scale Deep Network” David Eigen, Christian Puhrsch, Rob Fergus Dept. of Computer Science, Courant Institute, New York University.</P>
<P class="p138 ft59"><SPAN class="ft29">[3]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</SPAN> “FlowNet: Learning Optical Flow with Convolutional Networks”, A. Dosovitskiy and P. Fischer, ICCV , 2015.</P>
<P class="p139 ft29"><SPAN class="ft60">[4]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</SPAN>“Depth and surface normal estimation from monocular images using regression on deep features and hierarchical CRFs” by Bo Li1, Chunhua Shen , Yuchao Dai , Anton van den Hengel, Mingyi He, IEEE Conference on Computer Vision and Pattern Recognition (CVPR'15). </P>
<P class="p136 ft63"><SPAN class="ft29">[5]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</SPAN>“Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches” by Jure Zbontar ,University of Ljubljana Vecna ,Yann LeCun, Journal of Machine Learning Research 17 (2016).</P>
<P class="p136 ft59"><SPAN class="ft29">[6]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</SPAN>https://github.com/LouisFoucard/StereoConvNet </P>
</DIV>
</BODY>
</HTML>
