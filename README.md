# Explore protein conformational space with variational autoencoder

## File explanation

### Main

1. align_trajs.py: align trajectories
2. maps_featurize.py: turn traj data into xyz features
3. training.py: train VAE/AE models
4. models.py: model architecture for encoder, decoder and VAE

### Utils

1. PDB_process.py: extract PDB template and write new coordinates
2. DOPE.py: calculate DOPE score of the given PDB
3. evaluation.py: evaluate the performance of VAE/AE models

## Dependencies

The dependencies are listed in `requirements.txt` file. 

For DOPE score calculation, `modeller` is required and can be downloaded [here](https://salilab.org/modeller/download_installation.html).  
