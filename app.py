from flask import Flask, render_template, request, send_file
import urllib.request
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sticker import get_sticker_img
import cv2

app = Flask(__name__)
# localhost:8000/aisticker
@app.route('/aisticker', methods=['POST','GET'])
def sticker():
    print("여기로 전송")
    myfile = request.form['myfile']
    url = 'http://localhost:8181/demo/resources/upload/'+myfile
    save_url = 'static/image/'+myfile
    print(myfile)
    try:
        img = urllib.request.urlopen(url).read()
        with open(save_url,"wb") as f:
            f.write(img)
            print("저장 완료")
    except urllib.error.HTTPError:
        print("접근할 수 없는 url입니다.")

    errorcode = get_sticker_img(save_url,'static/image/sticker_bat.jpg')
    if errorcode == "roierror":
        return render_template('error.html')
    else :
        return render_template('result.html',myfile='result_image/'+myfile)

if __name__ == '__main__':
    app.run(host='0.0.0.0')