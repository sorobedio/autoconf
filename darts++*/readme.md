#Instructions

In this folder we include the DARTS model searched on cifar 10.
W also included the best_model weights and the genotypes.
 To use the pretrained model 
<python test.py --auxiliary --model_path cifar10_model.pt>
expected results 97.52%

To search an architecture using the proposed training princple.
<python train_search.py --unrolled     # for conv cells on CIFAR-10>
