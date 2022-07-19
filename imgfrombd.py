# code source détécter des articles et recherche d'autres similaires (k=3)
# code entrainé par deepfashion2 (juste partie validation)
# model_final.pth est le modèle entrainé 





import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

from PIL import Image
import numpy as np
import pickle


from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import libraries
import os
import numpy as np
import cv2
import random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog












#liée la BD
import psycopg2
DB_Host='localhost'
DB_Name='chatbot'
DB_User='postgres'
DB_pass='hadil24'
connection = psycopg2.connect(dbname=DB_Name,user=DB_User,password=DB_pass,host=DB_Host)
cursor=connection.cursor()



from detectron2.data.datasets import register_coco_instances
#register_coco_instances("deepfashion_train", {}, "/content/DeepFashion2/deepfashion2_train.json", "/content/DeepFashion2/train/image")
register_coco_instances("deepfashion_val", {}, "/home/hadil/final/deepfashion2_validation.json", "/home/hadil/final/validation/image")




cfg = get_cfg() #la configuration par défaut de détéctron
cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")) #builtin config file.
cfg.DATASETS.TRAIN = ("deepfashion_val",)
cfg.DATASETS.TEST = ()

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo 
                                                                                                  #pre-trained weights
                                                                                                  #URL to the model trained using the given config

"""_BASE_: "../Base-RCNN-FPN.yaml"
MODEL:
  WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
  MASK_ON: False
  RESNETS:
    DEPTH: 101
SOLVER:
  STEPS: (210000, 250000)
  MAX_ITER: 270000
  """                                                                                                  



#definir les hyperparametres du modèle
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500  #nombre d'itération
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13 #nombre de classe

cfg.TEST.EVAL_PERIOD = 500



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) # create directory output_dir
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False) 



cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/home/hadil/final/model_final.pth")  #trained model path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55   # seuil utilisé pour filtrer low-scored bounding boxes predicted by the Fast R-CNN during the test
                                               
cfg.DATASETS.TEST = ("deepfashion_val", )
predictor = DefaultPredictor(cfg)



model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3)) #pretrained resnet for extract image features
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

lr = 0.1
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy']) #optmisation de resnet 




def listimg():

    filenames=[]

    print("\nDisplaying Entire Table product ")
    cursor.execute("SELECT * FROM attachment;")
    output=cursor.fetchall()
    for item in output:
        print(item[1])
        filenames.append(item[1])
    def extract_features(img_path,model):  #extract features of all images in the dataset
        img = image.load_img(img_path,target_size=(224,224)) #charger une img
        img_array = image.img_to_array(img) #convertir en array
        expanded_img_array = np.expand_dims(img_array, axis=0) #(1,224,224,3)
        preprocessed_img = preprocess_input(expanded_img_array) #The images are converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)

        return normalized_result
    feature_list = []

    for file in tqdm(filenames):
        feature_list.append(extract_features(file,model))
    pickle.dump(feature_list,open('embedding.pkl','wb')) #créer un fichier pkl pour charger les vecteurs de caractéristiques
    pickle.dump(filenames,open('filenames.pkl','wb'))    #créer un fichier pkl pour charger les paths des images du dataset
    feature_list=pickle.load(open('embedding.pkl','rb')) #load features list
    filenames=pickle.load(open('filenames.pkl','rb')) #load image's paths list
    
    return  filenames,feature_list




def model_img(path,filenames,feature_list):
   ######## Instance 0
    im = cv2.imread(path)
    outputs = predictor(im) #output is a dictionary contain all the predicted instances
    #Metadata is a key-value used to interpret what’s in the dataset( names of classes)..
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))  #moving outputs["instances"] to cpu then apply draw_instance_predictions

    import uuid #helps in generating random objects of 128 bits as ids
    
    boxes = {}
    for coordinates in outputs["instances"][0].to("cpu").pred_boxes:   #detecter une seule instance
     coordinates_array = []   #les coordonnées de chaque instance
     for k in coordinates:
        coordinates_array.append(int(k))  
    
     boxes[uuid.uuid4().hex[:].upper()] = coordinates_array   #charger toutes les instances dans un dict 
     #crop chaque instance et l'afficher
    for k,v in boxes.items():
      crop_img = im[v[1]:v[3], v[0]:v[2], :]
    
     
      cv2.imwrite('part1.jpg', crop_img)
   
    


    #feature extract input image + applique knn
    img = image.load_img('part1.jpg',target_size=(224,224)) #charger une img
    img_array = image.img_to_array(img) #convertir en array
    expanded_img_array = np.expand_dims(img_array, axis=0) #(1,224,224,3)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    neighbors= NearestNeighbors(n_neighbors=3,algorithm='brute', metric='euclidean') #Number of neighbors to use=5, ‘brute’ will use a brute-force search, euclidean: distance entre 2 points
    neighbors.fit(feature_list)
    distances,indices= neighbors.kneighbors([normalized_result]) 
    print(indices) #liste des indices de les 3 images similaires
    print(distances) #liste des distances de les 3 images similaires
    list_sim=[] 
    for file in indices[0]:
     
     list_sim.append(filenames[file])

    return list_sim 




    
    
















  
   




