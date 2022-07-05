import os
import logging 
import shutil
from bing_image_downloader.downloader import download
from PIL import Image


logging.basicConfig(format = '%(levelname)s:%(message)s', level=logging.INFO)


def request_download(path:str, list_names:list, limit_loads:int):

    """Create requests and download pictures from Bing.com
       :param path: path to dataset
       :param list_names: the list with actors' names to download
       :param limit_load: number of downloaded images
       :return: None """


    if not os.path.exists(path):
        os.makedirs(path)      


    #delete folder with the dataset
    logging.info('Clean the folder')
    shutil.rmtree(path)

    logging.info('Download images')
    
    for name in list_names:
        req = f'face {name}'
        download(req,
                limit = limit_loads,
                output_dir = path,
                adult_filter_off = True,
                force_replace = False,
                timeout = 60,
                verbose = True)

    #Rename folders (delete the word 'face')
        new_name = ' '.join(req.split()[1:])
        os.rename(path+ '/'+ req, path+ '/'+ new_name)
    return req


def resize_photo(image:Image, n_size:int):
    """Resize one image
       :param image: image 
       :n_size: new size of the image 
       :return: resized image
       """
    
    size = image.size
    coef = n_size/size[0]

    resized_image = image.resize((int(size[0]*coef), int(size[1]*coef)))
    resized_image = resized_image.convert('RGB')
    return resized_image


def convert_im(list_names:list, n_size:int, path:str):
    """Convert images
    :param list_names: the list with actors' names 
    :n_size: new size of the image 
    :param path: path to dataset
    """
    logging.info('Formatting Images')

    for name in list_names:
        files = os.listdir(f'{path}/{name}')

        for i in files:
            try:
                image = Image.open(f'{path}/{name}/{i}')
                resized_image = resize_photo(image, n_size)
                resized_image.save(f'{path}/{name}/{i}')
                
            except:
                os.remove(f'{path}/{name}/{i}')
