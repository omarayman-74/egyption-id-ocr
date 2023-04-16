from flask import Flask, render_template, request, session
import os
from werkzeug.utils import secure_filename
import cv2
from rembg import remove
from PIL import Image
from easyocr import easyocr
import numpy as np
import pytesseract

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# The default folder name for static files should be "static"
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'You can write anything, is just a test'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=("POST", "GET"))
def upload_file():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        # Upload file to database (defined uploaded folder in static path)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)

        # Perform image processing and text extraction on the uploaded file
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'moraaa.png')
        input = Image.open(input_path)
        output = remove(input)
        output.save(output_path)

        img = cv2.imread(output_path)

        blurred = cv2.blur(img, (5,5))

        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)

        canny = cv2.Canny(sharpened, 50, 200)

        pts = np.argwhere(canny>0)
        y1,x1 = pts.min(axis=0)
        y2,x2 = pts.max(axis=0)

        cropped = img[y1:y2, x1:x2]

        w,h,c=cropped.shape
        o=int(w/2)
        i=int(h/2.5)
        n=int(h/6)
        cr=cropped[n-10:i+15,o:]
        cropped_img=cropped[i+10:,o+15:]
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "newimg.png"),cropped_img)

        text=pytesseract.image_to_string(cr,lang='ara',config='--psm 11 --oem 3')
        splited=text.split('\n')

        state=0

        if len(text.split('\n'))==8:
            state=1
            print(state)
            firstname=splited[0]
            secondname=splited[2]
            adress=splited[4]+" "+splited[6]

            data=[firstname,secondname,adress]
            for i in data:
                if i==None:
                    print("error! reenter img")
                    break
                else:
                    imgs = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'newimg.png'),0)
                    gauss = cv2.GaussianBlur(imgs, (7,7), 0)
                    unsharp_image = cv2.addWeighted(imgs, 2, gauss, -1, 0)

                    cv2.imshow("unsharp",unsharp_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()    

                    s=easyocr.Reader(['ar','ar'])
                    o=s.readtext(unsharp_image, detail = 0,text_threshold = 0.2
                    ,width_ths = 0.8,low_text= .17)
                    m=0

                    data.append(o)
                    print(data)
                    break


        elif state==0:
            state=2
            
            imgs = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'newimg.png'),0)
            gauss = cv2.GaussianBlur(imgs, (7,7), 0)
                   
            unsharp_image = cv2.addWeighted(imgs, 2, gauss, -1, 0)
            s=easyocr.Reader(['ar','ar'])

            d=s.readtext(cr, detail = 0,text_threshold = 0.27
            ,width_ths = 0.9,low_text= 0.17)
            
            o=s.readtext(unsharp_image, detail = 0,text_threshold = 0.27
            ,width_ths = 0.9,low_text= 0.17)
            print(state)
            
            cv2.imshow("id",unsharp_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()    
                
            
            if o==None or d==None:
                state=4
            else:
                print(o)
                print(d)
        elif state==4:
            print("reenter img")

    return render_template('uploaded_image.html')

@app.route('/show_image')
def display_image():
    # Retrieving uploaded file
    img_file_path = session.get('uploaded_img_file_path', None)
    # Display image in Flask application web page
    return render_template('show_image.html', user_image=img_file_path)

if __name__ == '__main__':
    app.run(debug=False)

