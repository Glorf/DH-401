# DH-401: EPFL Digital Humanities - Digital Musicology semester project

## Predicting music popularity using DNNs

### Research Question  

It's common in the history of music that some genres and bands gain very high popularity and some aren't noticed at all. It's not really easy to find out which piece will gain an universal acclaim. Or is it? Maybe the popularity of the most known music is the intrinsic feature of that music itself?

In this project, we want to answer the question: **to what extent music popularity can be identified solely based on the musical features, and what are the features that make music popular?** It's interesting to see whether there are universal parameters and patterns which lead to the increased music popularity. Needless to say, that this kind of information is priceless for the whole music industry, and for the musicians themselves, as the music which is more popular might have a higher chance of selling better and being more plausible for the audience. As a result of this project, we expect to identify several technical characteristics of the music which result in its success.

### Concepts and Data

The project consists of two parts. First, we want to train an end-to-end deep learning model to predict popularity of the music. It's to be decided whether the task of the model will be regression or classification (predicting audience number vs quantized levels of popularity). Then, we want to perform the blackbox model analysis ML to try to see what features cause those specific pieces to be more likely to be popular than others. By modifying computationally the physical parameters of the predicted music (eg. pitch, tempo), mixing-in noise or adding different kinds of effects, we want to see how these changes affect the model prediction and thus their possibility to affect the composition positively and negatively. This will be a non-straight approach to creating interpretability of the model.

Creating an end-to-end deep learning model requires a vast amount of data for training and evaluation. For this reason we want to use the music dataset introduced in the paper “FMA: A Dataset for Music Analysis” by Michael Defferrard, Kirell Benzi, P. Vandergheynst and X. Bresson (2017) The data is publicly available here: [fma](https://github.com/mdeff/fma).
There are two datasets that could be used. A small one containing 8 genres, and a larger one with 22 genres.

The dataset consists both of popularity scores and MP3 files, therefore it should be applicable to the training with minimal or no pre-processing. Given the size of the dataset, we should make use only of the smallest subset (fma_small or fma_medium) due to the computational complexity of our task.

### Methods

We propose two methods of developing this project. First is the ambitious one: we would like to try to fine-tune wav2vec2.0, a novel CNN+Transformer architecture from Facebook AI, on a subset of the data (>1GB) to create "music piece embeddings". Then, we should be able to fine-tune a popularity prediction layer to predict popularity on top of that model. Then, by altering the input music we can try to conclude, which features of the piece are important for it to be predicted as more popular. The ultimate goal would be to prepare a list of parameters which increase the piece likelihood to be of high popularity, and the ones that decrease it. This solution is completely novel and it's also it's major drawback - since the wav2vec2 architecture was prepared with speech processing in mind, and was not tested for music yet, we cannot be perfectly sure that our solution will work well for music.

The model would predict popularity of a music, and we are interested in knowing what makes a music popular. To answer that, we could give some input to the model music, and tweak some of its features (pitch, …), see how the model reacts, and identify groups of key features like that. We would expect some features to be more important than others, and also we expect them to be correlated.

Therefore, there is also a fallback method: in case of failure, we want to train a shallow/linear model for such a prediction, based on the precomputed musical features (eg. features from librosa available in the FMA dataset). This kind of model is guaranteed to work to some extent, and is much more interpretable. However, it has no chances of performing as well as a deep network.

### Literature

* [wav2vec-2.0](https://www.semanticscholar.org/paper/wav2vec-2.0%3A-A-Framework-for-Self-Supervised-of-Baevski-Zhou/4f55740f5eaa67ec7f388c4e204154d8dc68fa06)  
The architecture is fresh and was originally used for speech-to-text processing, we want to pre-train it for music analysis tasks

* [Music Popularity: Metrics, Characteristics, and Audio-based Prediction](https://www.semanticscholar.org/paper/Music-Popularity%3A-Metrics%2C-Characteristics%2C-and-Lee-Lee/d1bb1d35ad3666f196e6292b1f782c10da6ba677)  
The paper states some hypothesis on what makes a song popular, and looks at how it has evolved in time, to try to understand how songs become popular

* [Predicting Music Popularity Patterns based on Musical Complexity and Early Stage Popularity](https://www.semanticscholar.org/paper/Predicting-Music-Popularity-Patterns-based-on-and-Lee-Lee/861d3d2c092d5394151f8fc9b1666e67fa0a69f5)  
The paper is trying to predict the popularity of a song based on some predefined features, we would only give as input to our model the raw audio (our fallback solution would be similar however).

* [FMA: A Dataset for Music Analysis](https://www.semanticscholar.org/paper/FMA%3A-A-Dataset-for-Music-Analysis-Defferrard-Benzi/9a82095be10926f0a52f8f9939deadfe39be2184)
We want to use the dataset in a novel way, and show its wider application to general purpose musicology.

### Feedback request

We'll be happy to see what are the promising non-disruptive ways of modifying recorded music and some software that can be used to do so, preferably as a python library
