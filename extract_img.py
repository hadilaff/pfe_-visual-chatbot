#chatbot exraire des données depuis des images de preuve de paiement bancaire en utilisant pytesseract.
# enregistrer les données extraites automatiquement dans la base de données.
#lire les données en arabe et en francais




import cv2
import numpy as npc
import pytesseract
from pytesseract import Output
import psycopg2

from pickle import TRUE
from flask import Flask, request
import requests
import json
import wget 


DB_Host='localhost'
DB_Name='chatbot'
DB_User='postgres'
DB_pass='hadil24'

connection = psycopg2.connect(dbname=DB_Name,user=DB_User,password=DB_pass,host=DB_Host)
cursor=connection.cursor()


app = Flask(__name__)


PAGE_ACCESS_TOKEN = "EAAJHQuCEJ0oBAGZC4mvTD5FL96XkbDHFXTOmgUUCQiCpLIAZBit5Q1yZCaHWPa7sAKZBmFFq4ZCm0AbxwaniTgMVZAMZC9ZCh89G6ZC1M45WYZAoOCVzwlkXhcdr6wpmXlFV6XLxyoZAySIBtjYaf1dhYLmMd7jsPIzLM5ncrGsbuL7ZByOSVlJpKUBbe8UhxITNiBxVNTyRsIqXUIfS8g5D81P0"
# Function to access the Sender API

def callSendAPI(senderPsid, response):
    payload = {
        'recipient': {'id': senderPsid},
        'message': response,
        'messaging_type': 'RESPONSE',
        
    }
    headers = {'content-type': 'application/json'}

    url = 'https://graph.facebook.com/v10.0/me/messages?access_token={}'.format(PAGE_ACCESS_TOKEN)
    r = requests.post(url, json=payload, headers=headers)
    print(r.text)






def handleMessage(senderPsid, receivedMessage):
    # check if received message contains text
    print('We entered the HANDLE MESSAGE FUNCTION')


    #if 'id' in receivedMessage:
     #id=  receivedMessage['attachments']
     #id_s=id['id']
    if 'attachments' in receivedMessage:
        att = receivedMessage['attachments']
       
        
        a=att[0]['payload']['url']
        wget.download(a, out='/home/hadil/final/attachment/'+a[96:194]+'.jpg')
        path = '/home/hadil/final/attachment/'+a[96:194]+'.jpg'

        img=cv2.imread(path)
        dim=(2128, 1560)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR) #resized image
        ret,th=cv2.threshold(resized,175,255,cv2.THRESH_BINARY) #binarisation de l'img
        

        config="--psm 3" #full page
        text=pytesseract.image_to_string(th,config=config,lang="Arabic+fra")#afficher tout le texte extrait
        print(text)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

        im_data= pytesseract.image_to_data(th,lang="Arabic+fra",output_type=Output.DICT)
        print(im_data['text'])
        for i,word in enumerate(im_data['text']):
            if len(word) >=10: 
                
                if all(chr.isdigit() for chr in im_data['text'][i]):
                  compte=im_data['text'][i]
                  print(compte)
                  cursor.execute(f"INSERT INTO transaction_banking ( image_link,sm_id, rib) VALUES ('{path}','682496387526842','{compte}');")
                  connection.commit()
        
        for i,word in enumerate(im_data['text']):
            if word =="dinar(s)" or word =="dinars" or word =="dinar":
                montant=im_data['text'][i-2]+' '+im_data['text'][i-1] 
                print(montant)
                cursor.execute(f"UPDATE transaction_banking SET amount= '{montant}';")
                connection.commit()

        
            cursor.execute(f"UPDATE transaction_banking SET fbinsta_user_id= '{senderPsid}';")
            connection.commit()
 
                  

        response = {"text": "Bien reçu"}
            

        callSendAPI(senderPsid, response)  



    else:
        response = {"text": 'This chatbot only accepts image'}
        callSendAPI(senderPsid, response)            
        




"""
img=cv2.imread("/home/hadil/final/extract/stb.png")

dim=(2128, 1560)
resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR) #resized image
#resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
ret,th=cv2.threshold(resized,175,255,cv2.THRESH_BINARY) #binarisation de l'img

#cv2.imshow('im',th)
#cv2.waitKey(0)

config="--psm 3" #full page
text=pytesseract.image_to_string(th,config=config,lang="Arabic+fra")#afficher tout le texte extrait
print(text)
cv2.waitKey(0)
cv2.destroyAllWindows()


im_data= pytesseract.image_to_data(th,lang="Arabic+fra",output_type=Output.DICT)
print(im_data['text'])



### afficher le nom de la banque
for i,word in enumerate(im_data['text']):  
  if word =="banque" or word =="Banque" or word =="Banque:":
    print(im_data['text'][i+1])

#afficher montant
for i,word in enumerate(im_data['text']):
  for j,word in enumerate(im_data['text'][i]):
    if word==",":
      #print(im_data['text'][i])
      if any(chr.isdigit() for chr in im_data['text'][i]) and len(im_data['text'][i])>=4:
         print(im_data['text'][i])    



# RIB ou num de compte
for i,word in enumerate(im_data['text']):
  if len(word) >=10: 
    
    if all(chr.isdigit() for chr in im_data['text'][i]):
      compte=im_data['text'][i]
      print(compte)
      cursor.execute(f"INSERT INTO transaction_banking ( rib) VALUES ('{compte}');")
      connection.commit()


#afficher montant
for i,word in enumerate(im_data['text']):
  if word =="dinar(s)" or word =="dinars" or word =="dinar":
     montant=im_data['text'][i-2]+' '+im_data['text'][i-1] 
    
     cursor.execute(f"INSERT INTO transaction_banking (amount) VALUES ('{montant}') WHERE transaction_id={7};")
     connection.commit()
   # print(im_data['text'][i-2]+' '+im_data['text'][i-1] )         

"""





@app.route('/', methods=["GET", "POST"])
def home():
    return 'HOME'


@app.route('/webhook', methods=["GET", "POST"])
def index():
    if request.method == 'GET':
        # do something.....
        VERIFY_TOKEN = "hadil-2022"

        if 'hub.mode' in request.args:
            mode = request.args.get('hub.mode')
            print(mode)
        if 'hub.verify_token' in request.args:
            token = request.args.get('hub.verify_token')
            print(token)
        if 'hub.challenge' in request.args:
            challenge = request.args.get('hub.challenge')
            print(challenge)

        if 'hub.mode' in request.args and 'hub.verify_token' in request.args:
            mode = request.args.get('hub.mode')
            token = request.args.get('hub.verify_token')

            if mode == 'subscribe' and token == VERIFY_TOKEN:
                print('WEBHOOK VERIFIED')

                challenge = request.args.get('hub.challenge')

                return challenge, 200
            else:
                return 'ERROR', 403

        return 'SOMETHING', 200

    if request.method == 'POST':
        # do something.....
        VERIFY_TOKEN = "hadil-2022"

        if 'hub.mode' in request.args:
            mode = request.args.get('hub.mode')
            print(mode)
        if 'hub.verify_token' in request.args:
            token = request.args.get('hub.verify_token')
            print(token)
        if 'hub.challenge' in request.args:
            challenge = request.args.get('hub.challenge')
            print(challenge)

        if 'hub.mode' in request.args and 'hub.verify_token' in request.args:
            mode = request.args.get('hub.mode')
            token = request.args.get('hub.verify_token')

            if mode == 'subscribe' and token == VERIFY_TOKEN:
                print('WEBHOOK VERIFIED')

                challenge = request.args.get('hub.challenge')

                return challenge, 200
            else:
                return 'ERROR', 403

        # do something else
        data = request.data
        body = json.loads(data.decode('utf-8'))

        if 'object' in body and body['object'] == 'page':
            entries = body['entry']
            for entry in entries:
                webhookEvent = entry['messaging'][0]
                print(webhookEvent)

                senderPsid = webhookEvent['sender']['id']
                print('Sender PSID: {}'.format(senderPsid))

                if 'message' in webhookEvent:
                    handleMessage(senderPsid, webhookEvent['message'])
                return 'EVENT_RECEIVED', 200
        else:
            return 'ERROR', 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



