# pfe-visual-chatbot
Project: Visual Chatbot 
Visual recognition based chatbot using deep learning techniques. 
Find similar clothing items 
To search for similar items, we divide the task into two parts: Detection and research 
Detection of clothing items To do this we used the Faster-RCNN pretrained network, one of the forms of CNN. 
Dependencies 
• Detectron2: is a powerful object detection and image segmentation framework 
powered by Facebook AI research, contains several pre-trained object detection networks. https://github.com/facebookresearch/detectron2 DeepFashion2 dataset: contains 491K diverse images of 13 clothing categories (each item is labelled). https://github.com/switchablenorms/DeepFashion2 
Cuda 
• Detectron2 uses PyTorch 

Installation 
Dataset DeepFashion2: https://drive.google.com/drive/folders/125F48fsMBZ2EFOCpqk6aaHet5VH3990k unzip the files: ! unzip -p "2019Deepfashion2**" /content/drive/MyDrive/DeepFashion2Dataset/validation.zip -d /content/DeepFashion2/ 

Install the dependencies: 
!pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
!pip install cython pyyaml==5.1 !pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' import torch, torchvision print(torch. _version__, torch.cuda.is_available()). !gcc --version !install opencv 

Install detectron2 
!pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html 
search for similar items 
• We used Nearest Neighbors algorithm to calculate the nearest points [neighbors= 
NearestNeighbors (n_neighbors=5, algorithm='brute', metric='euclidean')] For extract image features we used ResNet50 network trained by ImageNet dataset. [model = ResNet50(weights='imagenet', include_top=False, input_shape= (224, 224,3))] 
Install Tensorflow: 
pip install -- upgrade tensorflow 

Code source 
imgfrombd.py & automatic.py After the training our model which is (model_final.pth], it is used directly to apply the faster techniques -RCNN and KNN in [imgfrombd.py] which is used in [automatic.py] to create the chatbot. 
training_deepFashion2.ipynb To train the model [model_final.pth] from scratch by modifying the dataset or its size [training_deepFashion2.ipynb] details the entire process. 
Removing mannequin skin To make the similarity search more accurate, we can remove the skin of the mannequin using the api used in the following code [remove.ipynb] using a trained CNN model [remove_skin.h5] 
Data extraction 
For data extraction from images we used Tesseract tool developped by google one of the OCR techniques ( optical character recognition) able to read multiple languages. This tool is based on LSTM one of the RNN network. 
Installation 
!pip install pytesseract ! sudo apt install tesseract-ocr !sudo apt-get install tesseract-ocr-por -y 
Read Arabic letters: 
!sudo apt-get install -y tesseract-ocr-script-arab 
To apply this tool the images must be binary, clear and filtered. [ret, th=cv2.threshold(resized, 175, 255, CV2. THRESH_BINARY)] this process is detailed in [extract.py]. 
