# AD4CD
### Quick start：  
you can run **train.py** to train the model, run **test.py** to
inference  
Run command format:
```
train.py 'dataset' 'k_fold' 'baseCDM' 'add_or_not' 'baseAD'
```
1. 'dataset' mean  what dataset you choose. you should put dataset into *data* folders
2. 'k_fold' mean which fold to choose as the validation set. you should choose 0-4
3. 'baseCDM' mean which CDM model to choose as the benchmark model. you should put CDM in *baseCDM* folders
4. 'add_or_not' mean whether to add **AD4CD** as an additional framework. you should choose 'add' or 'noadd'
5. 'baseAD' mean which AD algorithm to choose. You can implement the algorithm yourself or introduce the **pyod** library  


### A simple example:  
```
train.py ASSIST09 0 NCD add ECOD
```

It will generate model file in *model* folders, then run **test.py** to inference
