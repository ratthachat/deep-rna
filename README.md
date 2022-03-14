# DeepRNA - Easy &amp; Pretrained Deep Learning for RNA predictions
<img src=https://i.ibb.co/TmJ2k5S/RNABody-Model.png width="600">
Implemented with Tensorflow 2.X with Keras API 

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/ratthachat/tutorial-pretrained-sota-deeprna-model-made-easy)

## **Features**

* Architecture inspired by winning models in [mRNA OpenVaccine competition (2020)](https://www.kaggle.com/c/stanford-covid-vaccine) 
* Pretrained DeepRNA models, easy to use, Keras-style
* Applicable to general RNA prediction problems
* Built-in Dataset with can handle RNA of mixed lengths simultaneously (powered by [Spektral's Graph Loader](https://github.com/danielegrattarola/spektral) with little upgrade ;)
* Built-in Self-Supervised AutoEncoder pretraining
* Companion feature extractions from basic to advanced of any RNA dataset with minimum assumptions (See full tutorial below) 
* Advanced techniques like pseudo-labeling and uncertainty-handling can be done relatively easily (See the 2nd tutorial below)

## **Step-by-Step Tutorials on Kaggle**

Why Kaggle? Because Kaggle is almost like a "free" Colab "Pro" with an extra plus of free permanent storage which can easily transfer to any working notebook.
Kaggle working environment is amazing!

* A tutorial on "Preprocessing RNA strings for Deep Learning Models in a General "Graph" Setting [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/ratthachat/preprocessing-deep-learning-input-from-rna-string)

* Quick and advanced tutorials on "Finetune Pretrained State-of-the-Art DeepRNA Model to General Prediction Problems Made Easy"[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/ratthachat/tutorial-pretrained-sota-deeprna-model-made-easy)

<img src=https://i.ibb.co/8mkQ1vh/RNA-data-preprocessing.png width="600">


## Benchmark Results
Prediction provided by our tutorial is, as of Feb 2022, provided the highest scores in OpenVaccine's public notebooks (see [Benchmark in this page](https://www.kaggle.com/c/stanford-covid-vaccine/code?competitionId=22111&sortBy=scoreAscending) ). Note that in Kaggle benchmark, techniques such as multi-model and kfolds ensemble are standard)

## Quick Start in 4 Steps
**Step 1.** Clone the repo to your working directory
```
git clone https://ratthachat@github.com/ratthachat/deep-rna.git
cp -rf ./deep-rna/deep_rna ./
```

**Step 2.** Prepare your RNA dataset using the default option as suggested in [this tutorial](https://www.kaggle.com/ratthachat/preprocessing-deep-learning-input-from-rna-string).
After this step, you will have a list of RNA ids `rna_id_list`, and directories containing
RNA node and edge features i.e. `NODE_DIR` and `EDGE_DIR`.

**Step 3.** Make and load your dataset
```
from deep_rna.dataset import RNADataset
from deep_rna.spektral.data import BatchLoader

rna_dataset = RNADataset(rna_id_list,
                          node_dir=NODE_DIR,
                          edge_dir=EDGE_DIR,
                          manhattan_edge_feature=True)

batch_loader = BatchLoader(rna_dataset, batch_size=128, mask=True, shuffle=True, epochs=1) # set epochs=None to load indefinitly
```

**Step 4.** Load the pretrained model and get the RNA embedding vector so that you can add them into your ML pipeline!
```
from deep_rna.models import RNAPretrainedModel
model = RNAPretrainedModel(weights='openvaccine', include_top=False)

for x in batch_loader.load():
    embed = model.predict(x)
```

### Acknowledgement

* My friends [Akensert](https://github.com/akensert/) and [Raman](https://github.com/SamusRam) who inspired me about this project.
* Amazing data scientists at Kaggle who contributed to the mRNA modeling and be a backbone to this project : [Gilles Vandewiele](https://www.kaggle.com/group16), [tito](https://www.kaggle.com/its7171), [Mrkmakr](https://www.kaggle.com/mrkmakr), [Jiayang Gao](https://www.kaggle.com/nullrecurrent) and [xhlulu](https://www.kaggle.com/xhlulu)
* [Spektral Project](https://github.com/danielegrattarola/spektral) whom I borrow and little modify their wonderful graph loader to handle RNA of arbitrary-length sequences.
