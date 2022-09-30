# Galaxy MNIST-convXpress
Implementation of convXpress CNN for [GalaxyMNIST Dataset](https://github.com/mwalmsley/galaxy_mnist).
## Train the model:
It is recommended to train the model using a Singularity/Apptainer container created from the included definition file ```gmnist.def```
Next download the GalaxyMNIST dataset files and place it in the ```data/``` directory.

Finally run the ```train_model.py``` file:
```
apptainer exec --nv gmnist.sif python train_model.py
```

## Evaluate the model:
Model evaluation is done in the Jupyter Notebook Files found in the ```notebooks``` driectory. Simply navigate there and open them in Jupyter Notebooks or Lab.
