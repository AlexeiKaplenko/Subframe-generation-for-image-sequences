[![Watch the video]](https://www.youtube.com/watch?v=DuxXRWH5Ex0&feature=youtu.be)

# Subframe-generation-for-image-sequences

Draft ID  14653   Docket Number  FE2847  

*Abstract  
Provide a brief summary of the invention and its key points of novelty.    

We have prototyped a technology based on state-of-the-art deep neural networks allowing to synthesize nonexistent frames in-between the original frames for AutoSlice&View approach with easy extension to other methods such as SpinMill, micro CT, VolumeScope and TEM tomography. 
The developed method brings two benefits:
-physical cuts can have a larger thickness, therefore a user can make a fewer slices and thus save time (missing slices will be interpolated at the same time as AS&V job is running).
-synthesized frames improve a resolution in the Z direction. 

*Problem Description  
Describe the problems that motivated this invention. Indicate whether there has been a long-standing need for a solution to these problems. Describe any earlier attempts to solve these problems, whether successful or not.   

Auto slice and View (and in principle all tomography techniques) is very time demanding method. In current approach, image acquisition using SEM requires several times more time than cutting with FIB.
The current solution can be improved by faster autofunctions or sparse scanning and it will bring throughput improvement by 10-20%. We offer a solution that can bring yield improvement by a factor from 2 to 8 (already tested) or theoretically even more (depends on the sample structure and required slice thickness). Slice thickness is limited by physical cutting medium thickness (ion beam or diamond knife), proposed approach improves depth resolution by artificial sub frame generation.
  
*Detailed Description of the Invention  
Provide a detailed description of the invention and the manner in which it operates to solve the problems described above. Describe any unexpected favorable results achieved by the invention.   

State-of-the-art optical flow method, can serve as a strong baseline for frame interpolation. However, motion boundaries and severe occlusions are still challenging to existing flow methods, and thus the interpolated frames tend to have artifacts around boundaries of moving objects. It's challenging to define optical flow vectors on raw microscopic data due to noise, local contrast variations and artifacts produced by electron microscope itself (defocus, astigmatism etc.)
Our approach was inspired by neural networks creating slow motion effect with high fps video from video with low fps. Proposed method outperforms classical computer vision algorithms because on top of calculating optical flow in the image neural network learns hierarchical features to gather contextual information from neighboring pixels. Optical flow is not calculated on raw data but rather on encoded feature representations of the image (U-Net type architectures were used). 
Thus the proposed method take advantage of the optical flow and local interpolation kernels (best from both worlds) for synthesizing the output frames.
As input network gets two subsequent frames and outputs user defined number of interpolated subframes in-between original frames (network can interpolate a frame at any arbitrary time step between two frames). 
Since the quality of interpolation frames is often reduced due to large object motion into two successive frames (what is Ok for creation slow motion effect but our customers are interested in obtaining high quality independent frames), in some cases make sense to apply classical sharpening algorithms in order to improve sharpness. In case of large object motion and blurry object boundaries second neural network can be applied to perform boundary restoration and create visually more appealing results visually indistinguishable for the user from the original frames.
Proposed technique creates 1 intermediate frame in resolution 6k in 10 seconds (using NVIDIA GeForce GTX 1080 GPU). Typical AS&V image acquisition time is a few minutes.
Our approach is an advanced interpolation technique. Thus, in case there is a feature in-between two frames that is not visible in any of them the network will miss it. However, the quality of interpolation is much higher than non-deep learning techniques since we interpolate by a very complex model obtaining 40 millions parameters. 
