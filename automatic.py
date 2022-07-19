from pickle import TRUE
from flask import Flask, request
import requests
import json
import wget 
from imgfrombd import listimg, model_img 
import requests

from imgurpython import ImgurClient

#client_id = 'c4dde22710aeda0'
#client_secret = 'c50d8e42b976613a1be5e1a42a56245fbda1e220'

app = Flask(__name__)


filenames,feature_list=listimg()
#liée la BD
import psycopg2
DB_Host='localhost'
DB_Name='chatbot'
DB_User='postgres'
DB_pass='hadil24'

connection = psycopg2.connect(dbname=DB_Name,user=DB_User,password=DB_pass,host=DB_Host)
cursor=connection.cursor()



        

















PAGE_ACCESS_TOKEN = "EAAJHQuCEJ0oBAPDWMM3aeYaBclvhUq1D7nbI80dLT137haUZC0qzwS7ByBBCa3inqXscGuwP8Cz2q8EByg7mYju22bcl5vN56tQm6j0DzZB70ar24CeWGZAjWvxioTGu5QWqG0L9QySQzpmAZCS8MGeSs5nymrImltYdHCSCtWovPdje8XT196Ogrlt3nJEUXXdcqnkfJAZDZD"
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


def callSendAPI2(senderPsid,img_url):
    payload={ 
        "recipient":{"id":senderPsid},
        "message":{
            "attachment":{
                "type":"image", 
                "payload":{
                  "url":img_url,
                  "is_reusable":True
                }
            }
        }
    }
    print("-----------------------AAAAAAAAAAAAAA-----------------------")
    print(senderPsid)
    print(img_url)
    url = 'https://graph.facebook.com/v10.0/me/messages?access_token={}'.format(PAGE_ACCESS_TOKEN)
    r = requests.post(url, json=payload, headers={'content-type': 'application/json'})

# Function for handling a message from MESSENGER
def handleMessage(senderPsid, receivedMessage):
    # check if received message contains text
    print('We entered the HANDLE MESSAGE FUNCTION')
    
    if 'attachments' in receivedMessage:
        att = receivedMessage['attachments']
       
        
        a=att[0]['payload']['url']
        wget.download(a, out='/home/hadil/final/attachment/'+a[96:194]+'.jpg')
        path = '/home/hadil/final/attachment/'+a[96:194]+'.jpg'
        filenames,feature_list=listimg()
        
        sim_list = model_img(path,filenames,feature_list)
        
        print("LEN OF SIM LIST ")
        print(len(sim_list))
        for sim in sim_list:
            id=cursor.execute(f"SELECT product_id FROM attachment WHERE attachment_path = '{sim}';")
            id = cursor.fetchone()
            print(id[0])
            x=id[0]
            cursor.execute(f"SELECT * FROM product WHERE product_id = {x};")
            output=cursor.fetchall()

            print("LEN OF OUTPUT LIST ")
            print(len(output))

            #client = ImgurClient(client_id, client_secret)
            
            #img = client.upload_from_path(sim, config=None, anon=True)
            #urlim = img['link']
            #print(urlim)
            
            files = {
                    'file': open(f"{sim}", 'rb'),
                }

            response = requests.post('https://0x0.st', files=files)
            link = response.text
            print(link)


            for item in output:

                print("in for loop")

                model_response = ("Le prix de "+item[1]+ " est " +item[2]+", les tailles disponibles sont " +item[3]+ " , les couleurs "+item[4]+", son status: "+item[5] )
                
                print(model_response)
                

                response = {"text": model_response}
                
                print("1OUTPUT: ")
                callSendAPI(senderPsid, response) 
                print("OUTPUT: 2")   
                callSendAPI2(senderPsid,link)
                print("OUTPUT: 3")





    else:
        response = {"text": 'This chatbot only accepts image'}
        callSendAPI(senderPsid, response)


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

