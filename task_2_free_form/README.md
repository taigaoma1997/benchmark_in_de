This is the repository for Task 2:  The free-form structure for trasmission filter inverse design. 

Please download the dataset, the trained models, the predicted structures (for diversity metrics) from the google drive [folder](https://drive.google.com/drive/folders/1VXDLD6ydglWOBs8TvXCm3-S5YbgiWuU4?usp=sharing), and put it under this folder.

# File introduction:



`./data`: Dataset used for training.

`./data_predicted`: The 1000 predicted structure for a randomly chosen transmission inverse design task. Used for analyze diversity. 

`./Net`: Saved 3 models (tandem, VAE, GAN).

`check_diversity.ipynb`: Examine the diversity of each model. 

`check_results_final.ipynb`: Examine the traininig process and color accuracy. 

`image_process.py`: Free-form structure construction

The training process for each model is provided as the name states. 
