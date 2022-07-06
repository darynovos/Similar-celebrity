# Similar-celebrity

Python framework that predicts celebrity similar to uploaded photo.

# Requirements
- `bing_image_downloader`
- `Pillow`
- `Shutil`
- `Pandas`
- `NumPy`
- `face_recognition`
- `Scikit-Learn`
- `PyYAML`
- `logging`

# Modules

1) load_data.py

Downloads images from bing.com and resizes them according to n_size parameter.

2) process_data.py

Finds faces on images and processes them to embeddings. Forms 3 files: dict_actors - dictionary with celebrity's name and ID, embendings - embendings of images, target - target variable with celebrity's ID.

3) train_module.py

Trains prediction model based on logistic regression.

4) Check_your_photo.py

Predicts celebrity similar to uploaded photo.

# Configuration file
File with parameters needed for modules functioning.

**train:**
- key_load - if True launches the data downloading from Bing.com
- key_process - if True process downloaded pictures and forms data for the model. Note, if key_load is True, key process is True y default.
- path_model - path to the file with the model
- test_size - the size of test data 

**load_data:**
- limit_loads - number of images to dowwnload
- list_names - list of names to dowload
- path - path to save downloaded pictures

**convert:**
- n_size - size of image convertion
- new_path - path to save data for the model

**predict:**
- file_path - path to photo for prediction


# How to use

1) Update config file

- if you don't have available images, set key_load as True
- if you want just process data (e.g. you manually added images) - set key_process as True
2) start train_model.py
3) start check_your_photo.py to check similar celebrity for your photo.

