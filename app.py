from dotenv import load_dotenv

load_dotenv() ## load all the environment variables
import time
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
from flask import Flask , request , render_template , jsonify
from gradio_client import Client
import requests
import json

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def format_response(response):
    # Replace newline characters with <br> tags for line breaks
    formatted_response = response.replace('\n', '<br>')
    formatted_response = formatted_response.replace('**', '<b>').replace('**' , '</b>')

    
    # Bold headings and italicize content within sections
    
    return formatted_response



def input_image_setup(uploaded_file):
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.content_type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts

def get_gemini_repsonse(input,image,prompt):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([input,image[0],prompt])
    return response.text


input_prompt="""
You are an expert in nutritionist and a medical image classifier  where you need to analyze the given image 
               and decide whether the image is food or not ,
               if it is not food image, just say "scan the report" and donot say anything else and 
               if the image is food image then provide the details of every food item in the image with calories intake
               as in the below format

               FOOD ITEMS AND CALORIES:

                1. Item 1 - XXX calories
                2. Item 2 - XXX calories
                3. Item 3 - XXX calories

                TOTAL CALORIES:
                Your total caloric intake from this meal is XXX calories.

                NUTRITIONAL ANALYSIS:
                Based on my assessment, this meal contains the following ratios of macronutrients:

                - Carbohydrates: XX%
                - Protein: XX%
                - Fat: XX%

                RECOMMENDATION:
                [Your food is healthy/Your food is not healthy] because [detailed explanation of why it is or is not healthy].
                I would recommend [specific suggestions to improve meal healthiness] to help optimize your nutritional intake.


                if it is not food image, just say "scan the report" and donot say anything else
"""







app = Flask(__name__)
client = Client("https://addai-breast-cancer-detection-with-deep-transfer-5a8b408.hf.space/")


def predict_cancer(image):
        result = client.predict(
                       image ,	# str representing input in 'img' Image component
                        api_name="/predict"
        )
        return result


@app.route('/' , methods = ['GET' , 'POST'])
def home():
    
    if request.method == 'POST':
     
       images_file = request.files['image']
       file_path = os.path.join('./static', 'output.jpeg')
       images_file.save(file_path)


       byte_image = input_image_setup(images_file)
       response = get_gemini_repsonse(input_prompt, byte_image, "")
       response=format_response(response)

       if "scan" in response.lower():
           response = predict_cancer(file_path)
           print (response)
               

       return render_template('index.html' , response=response )
    else : 
        return render_template('index.html')


if __name__ == '__main__':
     
     app.run(host = '0.0.0.0' , debug=True)

       