# CSC2541_Project
In this rep., I sahre the codes I wrote for the Colab note.

There are two main files: full_batch_experiments.py and mini_batch_experiments.py

In full_batch_experiments.py, the hessian gradient-overlap and the eigenvectors-overlap is computed by using all training-data. The hessian-gradient overlap is computed at the end of each epoch, whereas the eigenvectors-overlap is computed at the end of each five-epoch period (and the overlap is computed with repect to the previously calculated eigenvectors). I run a simple experiment with batch sizes 32, 128 and 1024. The plot is given as:
![Alt text](fhessianoverlap.png?raw=true "Title")
