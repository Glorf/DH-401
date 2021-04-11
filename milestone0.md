# DH-401: Digital Musicology semester project

## Predicting music popularity using DNNs

### Research Question

It's common in the history of music that some genres and bands gain very high popularity and some aren't noticed at all. It's not really easy to find out which piece will gain an universal acclaim. Or is it? Maybe the popularity of the most known music is the intrinsic feature of that music itself?

In this project, we want to answer the question: **to what extent music popularity can be identified solely based on the musical features, and what are the features that make music popular?** It's interesting to see whether there are parameters and patterns which lead to the increased music popularity. Needless to say, that this kind of information is priceless for the whole music industry, and for the musicians themselves, as the music which is more popular will, by definition, reach a broader audience and have a higher chance of selling better. This approach inscribes our work in the still recent field of *Hit Song Science* (HSS) whose goal is to "understand better the relation between intrinsic characteristics of songs and their popularity, regardless of the complex and poorly understood mechanisms of human appreciation and social pressure" ([Pachet, 2012](#pachet2012)). As a result of this project, we expect to identify several technical characteristics of the music which result in its success.

### Concept

The project consists of two different, comparable approaches to perform the same task: predicting popularity of a song. It's to be decided whether the task of the models will be regression or classification (predicting audience number vs quantized levels of popularity). The first method will be a traditional ML regression task, while the second one will rely on recent breakthrough in Deep Networks development (see [Methods](#Methods)) . The challenge (especially in the latter case) will be to identify musical features that drive popularity.

Previous HSS works have highlighted different musical features influencing music's popularity, defining parameters on which we will focus during our study in addition to data-driven outcomes. Music's pitch ([Jakubowski et al., 2017](#jakubowski); [Yang et. al, 2017](#yang); [Lee & Lee, 2018](#lee)), tempo ([Ni et al., 2011](#ni); [Jakubowski et al., 2017](#jakubowski); [Léveillé Gauvin, 2017](#gauvin)) and intrinsic loudness ([Ni et al., 2011](#ni); [Serra et al., 2012](#serra); [Gauvin, 2017](#gauvin); [Lee & Lee, 2018](#lee)) appear as recurrent features to discriminate hit songs from non-hits.

### Data

Creating an end-to-end deep learning model requires a vast amount of data for training and evaluation. For this reason we want to use the music dataset build by [M. Defferrard, K. Benzi, P. Vandergheynst and X. Bresson (2017)](#fma). The data is publicly available here: [fma](https://github.com/mdeff/fma). There are two datasets that could be used: a small one containing 8 genres, and a larger one with 22 genres.

The dataset consists both of popularity scores and MP3 files, therefore it should be applicable to the training with minimal or no pre-processing. Given the size of the dataset, we should make use only of the smallest subset (fma_small or fma_medium) due to the computational complexity of our task.

### Methods

We propose two methods of developing this project. First, we will train a shallow/linear model based on the precomputed musical features (available in the FMA dataset) to predict songs' popularity. While this highly interpretable model is guaranteed to work to some extent, we expect it to have limited performances. Indeed, previous works in the field of HSS using traditional machine learning methods have shown some limitations in their predictive power based solely on musical features ([Pachet & Roy, 2008](#pachet2008)). Thus, we intend to apply state-of-the-art machine learning algorithms to the task.

Therefore, we would like to develop a more ambitious method by fine-tuning wav2vec2.0 ([Baevski et al., 2020](#wav2vec)), a novel CNN+Transformer architecture from Facebook AI, on a subset of the data (>1GB) to create "music piece embeddings". Then, we should be able to fine-tune a popularity prediction layer to predict popularity on top of that model. This solution is completely novel and it's also it's major drawback - since the wav2vec2 architecture was prepared with speech processing in mind, and was not tested for music yet, we cannot be perfectly sure that our solution will work well for music.

The model would predict popularity of a music, and we are interested in knowing what makes a music popular. We can try to answer this question through a non-straight approach to create interpretability of the CNN. By modifying computationally the physical parameters of the predicted music, we want to see how these changes affect the model's prediction and thus their possibility to affect the composition positively or negatively. The ultimate goal would be to prepare a list of parameters which increase the piece likelihood to be of high popularity, and the ones that decrease it. We would expect some features to be more important than others, and also we expect them to be correlated.

Preliminary searches showed some python's module that could be used to tweak audio files. librosa ([McFee et al., 2015](#librosa)), Pydub ([Robert & Webbie, 2018](#pydub)), aupyom ([Rouanet & Rabault, 2019](#aupyom)), or [Audacity's Scripting](#audacity) could be helpful to modify pitch and temporal characteristics. An attempt to modify intrinsic loudness could be made using the code develope by [Steinmetz and Reiss (2021)](#pyloudnorm).

However failures of more traditional ML algorithms tend to show that "simple shallow models may not capture the rich acoustic and genre diversity exhibited in Western hits" ([Yang et al., 2017](#yang)). This might imply that modifying basic musical features in a meaningful way could be deceiving.

### Literature

- <a name="pachet2008">**[Pachet & Roy, 2008]**</a>: Pachet, F., & Roy, P. (2008). Hit Song Science Is Not Yet a Science. *ISMIR*. [Paper](https://www.cs.swarthmore.edu/~turnbull/cs97/f08/paper/pachet08.pdf).
  
- <a name="ni">**[Ni et al., 2011]**</a>: Ni, Y., Santos-Rodríguez, R., McVicar, M., & Bie, T.D. (2011). Hit Song Science Once Again a Science?. [Paper](https://www.semanticscholar.org/paper/Hit-Song-Science-Once-Again-a-Science-Ni-Santos-Rodr%C3%ADguez/c645b02ff9053c10151a09baf60be84d3e3ff12f), [website](http://scoreahit.com).
  
- <a name="pachet2012">**[Pachet, 2012]**</a>: Li, T., Ogihara, M., & Tzanetakis, G. (2011). Hit Song Science - by François Pachet. In *Music Data Mining* (1st ed., pp. 305–326). CRC Press. [Chapter](https://www.francoispachet.fr/wp-content/uploads/2021/01/pachet-11a.pdf).
  
- <a name="serra">**[Serrà et. al, 2012]**</a>: Serrà, J., Corral, Á., Boguñá, M., Martin, H. & Arcos, J.LI. (2012). Measuring the Evolution of Contemporary Western Popular Music. *Sci Rep* 2, 521. [Paper](https://doi.org/10.1038/srep00521).
  
- <a name="librosa">**[McFee et. al, 2015]**</a>: McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in python. In *Proceedings of the 14th python in science conference* (Vol. 8). [Paper](https://conference.scipy.org/proceedings/scipy2015/pdfs/brian_mcfee.pdf), [website](https://librosa.org), [GitHub page](https://github.com/librosa).
  
- <a name="fma">**[Defferrard et. al, 2017]**</a>: Defferrard, M., Benzi, K., Vandergheynst, P. & Bresson, X. (2017). FMA: A Dataset For Music Analysis. *ISMIR*. [Paper](https://arxiv.org/abs/1612.01840), [GitHub repository](https://github.com/mdeff/fma).
  
- <a name="jakubowski">**[Jakubowski et. al, 2017]**</a>: Jakubowski, K., Finkel, S., Stewart, L., & Müllensiefen, D. (2017). Dissecting an earworm: Melodic features and song popularity predict involuntary musical imagery. *Psychology of Aesthetics, Creativity, and the Arts, 11*(2), 122–135. [Paper](https://psycnet.apa.org/doi/10.1037/aca0000090).
  
- <a name="gauvin">**[Léveillé Gauvin, 2017]**</a>: Léveillé Gauvin, H. (2017). Drawing listener attention in popular music: Testing five musical features arising from the theory of attention economy. *Musicae Scientiae*, *22*(3), 291–304. [Paper](https://journals.sagepub.com/doi/abs/10.1177/1029864917698010?journalCode=msxa).
  
- <a name="yang">**[Yang et. al, 2017]**</a>: Yang, L., Chou, S., Liu, J., Yang, Y., & Chen, Y. (2017). Revisiting the problem of audio-based hit song prediction using convolutional neural networks. *2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 621-625. [Paper](https://ieeexplore.ieee.org/document/7952230).
  
- <a name="lee">**[Lee & Lee, 2018]**</a>: Lee, J., & Lee, J. S. (2018). Music Popularity: Metrics, Characteristics, and Audio-Based Prediction. *IEEE Transactions on Multimedia*, *20*(11), 3173–3182. [Paper](https://doi.org/10.1109/tmm.2018.2820903).
  
- <a name="pydub">**[Robert & Webbie, 2018]**</a>: Robert, J., Webbie, M., et al. (2018). *Pydub*. GitHub. [Website](http://pydub.com/), [GitHub repository](https://github.com/jiaaro/pydub/).
  
- <a name="aupyom">**[Rouanet & Rabault, 2019]**</a>: Rouanet, P. & Rabault, N. (2019). *aupyom*. GitHub. [GitHub repository](https://github.com/pierre-rouanet/aupyom).
  
- <a name="wav2vec">**[Baevski et. al, 2020]**</a>: Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. [Paper](https://arxiv.org/pdf/2006.11477.pdf).
  
- <a name="audacity">**[Audacity, 2021]**</a>: *Scripting - Audacity Manual*. (2021, March 8). Audacity. [Webpage](https://alphamanual.audacityteam.org/man/Scripting).
  
- <a name="pyloudnorm">**[Steinmetz & Reiss, 2021]**</a>: Steinmetz, C.J. & Reiss, D.J. (2021). pyloudnorm: A simple yet flexible loudness meter in Python. Under review. [Paper](https://csteinmetz1.github.io/pyloudnorm-eval/paper/pyloudnorm_preprint.pdf), [GitHub repository](https://github.com/csteinmetz1/pyloudnorm).
