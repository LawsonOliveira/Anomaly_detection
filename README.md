# Anomaly detection using an LSTM Variational Autoencoder

# Table of Contents
1. [What is an LSTM VAE ?](#introduction)
2. [Dataset](#Dataset)
3. [Metrics](#Metrics)
4. [Reconstruction](#second_approach)
5. [Future Works](#future)
6. [Requirements](#requirements)

# What is an LSTM Variational Autoencoder ? <a name="introduction"></a>
A variational autoencoder is divided into three parts, the encoder, the representation in latent space, and a decoder. The encoder is used to reduce the dimension of the input space so that it is possible to represent it in latent space with only a few parameters. This representation is done using a normal distribution with parameters $\mu$ and $\sigma$ to be estimated. With these parameters it is possible to sample and then use the decoder to try to recreate the input. This neural network is shown in the image below.

<img src="./Images/neural_net.png?raw=true " width="100%"> 


This concept can be expanded to detect anomalies, since if the network cannot get the same input and output, the input cannot be represented in the latent space. Different structures can be used to perform the encoding and decoding, in this project LSTM was used.




# Dataset <a name="Dataset"></a>
The dataset used was ECG5000, which has 5000 cardiograms classified into 5 classes. Each cardiogram has 140 measurements and the whole dataset corresponds to about 20 hours.

The representation of each class is shown below.

<img src="./Images/occurrences.png?raw=true " width="40%"> 


The next figure shows the mean cardiogram for each class and its variance.
<img src="./Images/mean_variance_data.png?raw=true " width="100%"> 






# Metrics <a name="Metrics"></a>
After the model had been trained, some tests were performed to determine the threshold and its behavior. The threshold corresponds to a 95% quantile in the validation set. After this, the model was run on the test set.

The figure below allows us to visualize the results obtained on the test set

<img src="./Images/results_test_set.png?raw=true " width="100%"> 



Accuracy: 0,989

Precision: 0,942

Recall: 0,972

F1 Score: 0,957




# Reconstruction <a name="Reconstruction"></a>

<img src="./Images/reconstruction.png?raw=true " width="100%"> 

# Future works <a name="future"></a>
- Compare with a Transformer-Based Variational Autoencoder

# Requirements  <a name="requirements"></a>
- Pytorch, SkLearn, NumPy, Tqdm
- Pandas, Seaborn, Matplotlib


# References
   - <a href="https://arxiv.org/pdf/1805.00794" target="_blank">ECG Heartbeat Classification: A Deep Transferable Representation</a>

   - <a href="https://openreview.net/pdf?id=r1cLblgCZ" target="_blank">Recurrent Auto-Encoder Model for Multidimensional Time Series Representation </a>

   - <a href="https://arxiv.org/pdf/1412.6581.pdf" target="_blank">Variational Recurrent Auto-Encoders </a>

   - <a href="https://www.mdpi.com/1424-8220/22/1/123/pdf?version=1641281672" target="_blank">Attention Autoencoder for Generative Latent Representational Learning in Anomaly Detection </a>

   - <a href="https://github.com/curiousily/Getting-Things-Done-with-Pytorch" target="_blank">Curiousily</a>

