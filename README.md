# VGG-Face-Logistic-Regression
Classifying images of faces using the same technique as the controversial "Gaydar" that could guess the sexual orientation used: https://psyarxiv.com/hv28a/


## Requirements
To install: ```pip3 install -r requirements.txt```
* Python 3.*
* Tensorflow
* Sklearn
* CV2
* Keras
* Keras VGG Face
* tqdm
* Numpy
* Six


## Adjusting parameters
Before you create your dataset file and train your model it's adviseable that you adjust the parameters first. This you can do by editing the file ```config.py```. If you don't understand a parameter it's best to leave it alone.

When running a script you can always pass the parameters directly and override the ```config.py``` parameters. To get to know what the parameters are you can type ```python3 script_name.py -h```.


## Dataset
There's a dataset in the ```data``` folder with images of faces from men and women.

To use your own dataset you can look at the file ```data/data.json``` and copy the same format.


## Training
It's as easy as running the script ```python3 training.py```


## Testing
It's as easy as running the script ```python3 test_model.py --img_path=image.jpg```


### Todos
* Other ML techniques instead of Logistic Regression.


### Other
Made by Oliver Edholm, 15 years old.
