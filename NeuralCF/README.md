# Music Recommendation System

An simple scratch on recommendation system. Servers as final project of Machine Learning course @ UVa

## Data Description

### Raw Data

For the raw data of Amazon Review Dataset (https://nijianmo.github.io/amazon/index.html).        

Here we only use the Digital Music subset of the whole dataset.     

These are some statistics which are important for us to have a knowledge on the data:

#### Original Data

* Total items: 456992
* Total users: 840372
* Total interactions: 1584082

It is clearly that this dataset is extremely sparse. The density of this dataset is only about 0.0004125% which is significantly less than the density of some other commonly used dataset such as: movielens (4.468%). Thus, there are some additional operations need to be done to preprocess the data.