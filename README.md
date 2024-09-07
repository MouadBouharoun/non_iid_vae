# Detecting Model Inconsistency Attacks Against Federated Learning Systems


**Requirements**:

* python3 / jupyter
  * TensorFlow2
  * keras 
  * numpy
  * matplotlib
  * tqdm
  * pandas
  * scikit-learn
 
⚠️ **Important Note:** We cannot definitively determine that the instance is not malicious because the probability distribution of the global models is unknown, 
        given our assumption that the data is not IID (independently and identically distributed).
  
**Usage**:
```
python3 main.py -s <settings file> -n <number of clients>
```
