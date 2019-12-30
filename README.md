# Exploring Co-variate Shift

See [here](https://github.com/JRC1995/Causal-Inference/blob/master/CS_594_Causal_Inference_Final_Report.pdf) for details.

MNIST_1 contains codes for experimenting on Colored MNIST 1.0 and MNIST_2 contains codes for experimenting on Colored MNIST 2.0. First gen_color_mnist.py needs to be executed to download and generate the data in either of the folder. Then there are separate files for training separate models. Any of the train_XYZ.py files can be run for training and testing. 

# Credits

The codes for ICP, IRM, Colored MNIST 1.0 generation, and synthetic data generation is taken from here: 
https://github.com/facebookresearch/InvariantRiskMinimization.

Progress_Report/experiment_synthetic is mostly taken from: https://github.com/facebookresearch/InvariantRiskMinimization.
The codes in Progress_Report/Colored_MNIST/ is built upon: https://github.com/facebookresearch/InvariantRiskMinimization.
(The data generation code, and the standard IRM penalty function). The entropy penalty function is taken from here:
https://github.com/salesforce/corr_based_prediction

The starting VAE codes in MNIST_1/models and MNIST_2/models were based on: https://github.com/pytorch/examples/blob/master/vae/main.py (I modified it for my purpose - using convolutions and adding beta hyperparameter).

MNIST_1/gen_color_mnist.py is for generating Colored MNIST 1.0 data based on: https://github.com/facebookresearch/InvariantRiskMinimization.

MNIST_2/gen_color_mnist.py is for generating Colored MNIST 2.0 data based on: https://github.com/salesforce/corr_based_prediction.

The pretty print function from MNIST_1/common_functions.py and MNIST_2/common_functions.py is from: https://github.com/facebookresearch/InvariantRiskMinimization.

The ER_penalty_fn function in BVAE.py or Causal_BVAE_.py in MNIST_1/ or MNIST_2/ is built upon the codes from: https://github.com/salesforce/corr_based_prediction

The penalty_fn function in BVAE.py or Causal_BVAE_.py in MNIST_1/ or MNIST_2/ is built upon the codes from: https://github.com/facebookresearch/InvariantRiskMinimization.
