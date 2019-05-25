# LR-License-Plate-Generation
## Goal
Here we propose the method based on <To learn image super-resolution, use a GAN to learn how to do image degradation first>, to generate the real-world LR license plate images for the SR problem.
## Network
We tried to use auto-encoder structure to generate the LR images, but the performance is not good.   
The Generator is ResNet-based and the Discriminator is VGG based. Only MSE and GAN-loss is included.
Due to the different value of the noise value, one HR images could have multiple LR images and influenced by the noise level.
We transfer the Method to the License Plate SR problem.
## DataSet
The data we used is paired and has multiple degradations. User can download their own dataset from the Web and make a directory called "training", sub-directories are named "HR" and "LR".   
Input:  
![image](/src/training-demo/I1_000_deblur.jpg)  
GT:  
![image](/src/training-demo/I1_000.png)  
## Result
HR:  
![Alt text](/src/GT-hr/I1_000_deblur.jpg)
![Alt text](/src/GT-hr/I1_001_deblur.jpg)  
![Alt text](/src/GT-hr/I1_002_deblur.jpg)
![Alt text](/src/GT-hr/I1_003_deblur.jpg)  
![Alt text](/src/GT-hr/I1_004_deblur.jpg)
![Alt text](/src/GT-hr/I1_005_deblur.jpg)   
![Alt text](/src/GT-hr/I1_006_deblur.jpg)
![Alt text](/src/GT-hr/I1_007_deblur.jpg)   
LR:   
![Alt text](/src/gen-lr/I1_000_deblur.jpg)
![Alt text](/src/gen-lr/I1_001_deblur.jpg)   
![Alt text](/src/gen-lr/I1_002_deblur.jpg)
![Alt text](/src/gen-lr/I1_003_deblur.jpg)   
![Alt text](/src/gen-lr/I1_004_deblur.jpg)
![Alt text](/src/gen-lr/I1_005_deblur.jpg)   
![Alt text](/src/gen-lr/I1_006_deblur.jpg)
![Alt text](/src/gen-lr/I1_007_deblur.jpg)   
## Ohter Results
![Alt text](/src/other/I1_000_deblur.jpg)
![Alt text](/src/other/I1_001_deblur.jpg)   
![Alt text](/src/other/I1_002_deblur.jpg)
![Alt text](/src/other/I1_003_deblur.jpg)   
![Alt text](/src/other/I1_004_deblur.jpg)
![Alt text](/src/other/I1_005_deblur.jpg)   
![Alt text](/src/other/I1_006_deblur.jpg)
![Alt text](/src/other/I1_007_deblur.jpg)   

