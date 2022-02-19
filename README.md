# DeepRNA - Easy &amp; Pretrained SOTA Deep Learning for RNA predictions
<img src=https://i.ibb.co/TmJ2k5S/RNABody-Model.png width="600">
Implemented with Tensorflow 2.X with Keras API

### **Features**

* SOTA architecture inspired by winning models in [mRNA OpenVaccine competition (2020)](https://www.kaggle.com/c/stanford-covid-vaccine) 
* Pretrained DeepRNA models, easy to use, Keras-style
* Applicable to general RNA prediction problems
* Built-in Dataset with can handle RNA of mixed lengths simultaneously (powered with little upgrade by [Spektral's Graph Loader](https://github.com/danielegrattarola/spektral))
* Built-in Self-Supervised AutoEncoder pretraining
* Companion feature extractions from basic to advanced of any RNA dataset with minimum assumptions (See full tutorial below) 
* Advanced techniques like pseudo-labeling and uncertainty-handling can be done relatively easily (See the 2nd tutorial below)


### **Step-by-Step Tutorials on Kaggle**

Why Kaggle? Because Kaggle is almost like a "free" Colab "Pro" with an extra plus of free permanent storage which can easily transfer to any working notebook.
Kaggle working environment is amazing!

* A tutorial on "Preprocessing RNA strings for Deep Learning Models in a General "Graph" Setting [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/ratthachat/preprocessing-deep-learning-input-from-rna-string)

* Quick and advanced tutorials on "Finetune Pretrained State-of-the-Art DeepRNA Model to General Prediction Problems Made Easy"[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/ratthachat/tutorial-pretrained-sota-deeprna-model-made-easy)

<img src=https://i.ibb.co/8mkQ1vh/RNA-data-preprocessing.png width="600">

### Acknowledgement

* My friends [Akensert](https://github.com/akensert/) and [Raman](https://github.com/SamusRam) who inspired me about this project.
* Amazing data scientists at Kaggle who contributed to the mRNA modeling and be a backbone to this project : [Gilles Vandewiele](https://www.kaggle.com/group16), [tito](https://www.kaggle.com/its7171), [Mrkmakr](https://www.kaggle.com/mrkmakr), [Jiayang Gao](https://www.kaggle.com/nullrecurrent) and [xhlulu](https://www.kaggle.com/xhlulu)
* [Spektral Project](https://github.com/danielegrattarola/spektral) whom I borrow and little modify their wonderful graph loader to handle RNA of arbitrary-length sequences.
