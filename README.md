Robust Saliency Map Comparison
=========

ss_robust_metric MATLAB function performs comparison of saliency video sequence with ground-truth saliency video
sequence using the method described in "Objective evaluation" section of

> Y. Gitman, M. Erofeev, D. Vatolin, A. Bolshakov, A. Fedorov. "Semiautomatic Visual-Attention Modeling and
> Its Application to Video Compression". 2014 IEEE International Conference on Image Processing (ICIP).
> Paris, France, pp. 1105-1109.

The comparison carried out by the function is almost invariant to all common transformation, thus it allows to carry out
fare comparisons of saliency models.

Features
---------

 - Metric is invariant to
   - Black and white level corrections
   - Gamma correction
   - Mixing method with Center Prior
 - Finds the best center prior model for ground-truth sequence

Usage
-----
```MATLAB
function [ ss,per_frame_ss ] = ss_robust_metric(src_avi, gt_avi,sampling_density,optimization_iterations)
```

Input arguments:
 - `src_avi` - path to AVI file with saliency sequence (uncompressed AVI recommended)
 - `gt_avi` - path to AVI file with ground-truth saliency sequence (uncompressed AVI recommended)
 - `sampling_density` - [DEFAULT VALUE  = 400] sets frame sampling frequency for parameter optimization. Lower value leads
   to more precise results but increases computation time
 - `optimization_iterations` - [DEFAULT VALUE  = 100] set how many times method will try to find global extrema. Higer value leads
   to more precise results but increases computation time

Output arguments:
 - ss - final metric value. Higher metric values stand for higher correlation with ground-truth
 - per_frame_ss - vector of per-frame metric values

> NOTE 1: First run of the the function with new ground-truth sequence takes longer than subsequent runs. On the first run method
> selects centre prior model for the ground-truth video. The selected ground-truth model will be cached for future used in *_cp.png
> file
>
> NOTE 2: The method works really slow (it is the cost of its robustness), please, be patient

Usage example
------------
File example.m contains usage example. It performs comparison of two saliency sequences and visualizes its results.


Misc
-----

You can get ground-truth saliency sequencies from the [project website][]. With any questions regarding this code usage you can contact
Mikhail Erofeev merofeev@graphics.cs.msu.ru

If you use this code in your research please cite the following paper:
> Y. Gitman, M. Erofeev, D. Vatolin, A. Bolshakov, A. Fedorov. "Semiautomatic Visual-Attention Modeling and
> Its Application to Video Compression". 2014 IEEE International Conference on Image Processing (ICIP).
> Paris, France, pp. 1105-1109.

 [project website]: http://compression.ru/video/savam/        "project website"
