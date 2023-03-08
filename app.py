import base64

from flask import Blueprint, request
import numpy as np
import matplotlib.pyplot as plt
import cv2
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['file']
    # f.save('/var/www/uploads/uploaded_file.txt')
    print(f)
    count, image = caculate(f.read())
    # # 将bytes转成base64
    # data
    # data = base64.b64encode(data).decode()
    base= base64.b64encode(cv2.imencode(".jpg", image)[1].tostring())
    return {"count":count, "pic": str(base, 'utf-8')}
    print(count)

def show(img):
    plt.imshow(img)
    plt.show()

def caculate(buf):
    kernel = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="float32")
    img_array = np.fromstring(buf, np.uint8)
    # src = cv2.imread(file)
    src = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # show(src)
    src = cv2.filter2D(src, -1, kernel)


    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    # gray= cv2.Canny(gray, 100, 100)
    # show(gray)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 40, param1=20, param2=20, minRadius=20, maxRadius=30)

    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=38, param2=56, minRadius=45, maxRadius=100)
    count=0
    for circle in circles[0]:
        # 圆的基本信息
        #         print(circle[2])
        # 坐标行列－圆心坐标
        x = int(circle[0])
        y = int(circle[1])
        # 半径
        r = int(circle[2])
        # 在原图用指定颜色标记出圆的边界
        cv2.circle(src, (x, y), r, (0, 0, 255), 3)
        # 画出圆的圆心
        #         cv2.circle(src, (x, y),5, (0, 255, 0), -1)
        count+=1
        cv2.putText(src, str(count), (x-int(r/2), y+ int(r/2)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2) #在米粒左上角写上编号

    print(count)
    show(src)
    return count, src
    # show(src)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)