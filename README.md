
*** REPO IN CONSTRUCTION

This repo implements a language model based on the word level. 

This code is meant to work with heavy datasets (>500 Go of text), so I optimized the data loading and processing. See `Lazy contiguous dataset`

Through the code you will see some checks 
```
if os.name=='nt': 
  ...
 ```
 Which is a statement that allows me to check if I run this code on a local windows machine for debugging or on a Linux cluster.
 
 The small dataset used to debug/optimize this project can be found at : https://www.kaggle.com/c/asap-sas/data
 
 
 

