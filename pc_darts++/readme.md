
#Instructions

In this folder we include the PC-DARTS model searched PC-DARTS++ on cifar 10 AND imageNet

Searching for Cifar-10
To search an architecture using the proposed training princple.

to train on cifar10 or imagenet you have to set the approapriate genotype in genotype.py file


 
```python
python train_search.py
```
train cifar 10
```python
python train.py --auxiliary  --cutout
```

Search imagenet model 
```python
python train_search_imagenet.py   --tmp_data_dir ImageNetFolder   --save log_path
```


Train ImageNet Model
 ```python
python train_imagenet.py  --tmp_data_dir ImageNetFolder   --save log_path      --auxiliary 
```

