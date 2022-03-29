# CSC2541_Project
In this rep., I share the codes I wrote for the Colab note.

There are two main parts: full_batch_experiments.py and mini_batch_experiments.py

* In both parts, I only considered a two-layer FC model with 128-dimensional hidden units, run on MNIST dataset.

* In full_batch_experiments.py, the hessian gradient-overlap and the eigenvectors-overlap is computed by using all training-data (i.e., both gradients and Hessians are computed by using all data, not minibatch). The hessian-gradient overlap is computed at the end of each epoch, whereas the eigenvectors-overlap is computed at the end of each five-epoch period (and the overlap is computed with repect to the previously calculated eigenvectors). I run a simple experiment with batch sizes 32, 128 and 1024. The plot is as follows:


* In mini_batch_experiments.py, only the hessian-gradient overlaps are computed. They are computed by using mini-batches and the overlap of an epoch is set to the mean of the overlaps computed during that epoch. Here as well, I run a simple experiment  with batch sizes 32, 128 and 1024. The plot is as follows:

<p align="center">
   <img src="fhessianoverlap.png" width="350" height="350">
  <img   src="mhessianoverlap.png" width="350" height="350">
</p>
