from flask import Flask, request, jsonify, render_template,flash, redirect, url_for
from werkzeug.utils import secure_filename

import torch
from PIL import Image
import io
import cv2
import numpy as np
from ultralytics import YOLO
import os
###GPLv3###
from fitz import open as fitzopen
###GPLv3###
app = Flask(__name__)
app.secret_key = 'your_secret_key'

ALLOWED_EXTENSIONS = {'pdf'}
UPLOAD_FOLDER = 'uploads'

# 加載 YOLOv5 預訓練模型
# model = YOLOv5("yolov5s.pt")  # 可使用 'yolov5s.pt' 預訓練模型，其他模型可以從 YOLOv5 GitHub 下載
# Load a COCO-pretrained YOLO11n model
# model = YOLO("yolo11n.pt")
model = YOLO("best.pt")
# Directory to store result images
RESULT_FOLDER = 'static/results'
os.makedirs(RESULT_FOLDER, exist_ok=True)
# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pdf_to_pictures(pdf_path, output_folder, fileName):
    pdf_document = fitzopen(pdf_path)
    print("under pdf_document")
    print(len(pdf_document))
    # return
    # Iterate over each page in the PDF
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        
        # Extract images from the page
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_filename = f"{output_folder}/{fileName}_page_{page_number + 1}_image_{image_index + 1}.jpeg"
            
            # Save the image
            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)
            print(f"Saved image {image_filename}")

    # Close the PDF document
    pdf_document.close()


@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML template

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # 讀取圖片
    img = Image.open(file.stream)

    # 轉換圖片為 NumPy array
    # img_np = np.array(img)

    # 使用 YOLO 模型進行物體檢測
    # results = model("https://ultralytics.com/images/bus.jpg")

    results = model(img)  # 預測結果
    # results.pandas().xywh   # 輸出檢測結果（用 pandas）

    # Save the result image to disk as result2.jpg
    count = 0
    for result in results:
        result_image_name = f'{count}.jpg'
        result_image_path = os.path.join(RESULT_FOLDER, result_image_name)
        result.save(filename=result_image_path)
        count += 1
        # result.save(filename=result_image_path)  # Save to disk

        # Save the result image
    # result_image_name = 'result2.jpg'
    # result_image_path = os.path.join(RESULT_FOLDER, result_image_name)
    # results.save(filename=result_image_path)
    return render_template('result.html', image_url=f'/static/results/{result_image_name}')

    # Return the result image path
    return jsonify({'result_image': 'results/result2.jpg'})
    # 我們可以提取檢測結果，並返回簡單的 JSON 格式
    # result_data = results.pandas().xywh[0].to_dict(orient="records")
    # return jsonify({'predictions': result_data})


    # Run batched inference on a list of images
    # results = model(img_np)  # return a list of Results objects

    # Process results list
    count = 0
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        # result.show()  # display to screen
        result.save(filename=f"result{count}.jpg")  # save to disk
    return 'finished'
@app.route('/pdfUpload', methods=['GET', 'POST'])
def pdf_upload():
    if request.method == 'POST':
        # flash("TEST")
        if 'pdfFile' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['pdfFile']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # 存pdf檔案至 uploads 資料夾
        if file and allowed_file(file.filename):
            print('in 113 line')
            filename = secure_filename(file.filename)
            print(os.path.join(UPLOAD_FOLDER, filename))
            pdf_abs_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
            file.save(pdf_abs_path)
            flash(f'File uploaded successfully: {filename}')
            # return redirect(url_for('pdf_upload'))



            # 處理 PDF 檔案，抽取其中的圖片，存在static/檔案名稱.pdf資料夾中
            # mkdir for pdf to store images
            os.makedirs(os.path.join(app.static_folder, filename), exist_ok=True)
            pdf_to_pictures(pdf_abs_path, f'static/{filename}', filename.split('.')[0])
            # 返回上傳的 PDF 內存的圖片們
            return redirect(url_for('show_pdf_pictures', filename=filename))
        else:
            flash('Invalid file type. Only PDFs are allowed.')
            return redirect(request.url)
    return render_template('pdf_upload.html')
@app.route('/show_pdf_pictures/<filename>')
def show_pdf_pictures(filename):
    # images = ['1.jpg', '2.jpg', '3.jpg']  # 假設這是從 PDF 中提取的圖片列表
    # 這裡可以添加代碼來顯示上傳的 PDF 檔案的圖片
    # 例如，使用 pdf2image 將 PDF 轉換為圖片並顯示
    # 這裡僅作為範例，實際實現可能需要根據需求進行調整
    image_folder = os.path.join(app.static_folder, filename)# 等一下images要替換成pdf的檔名
    images = [img for img in os.listdir(image_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]
    print(images)
    newimages = [filename+'/' + img for img in images]
    # for image in images:
        # image = filename + '/' + image
    print(newimages)
    return render_template('show_pdf_pictures.html', filename=filename, images=newimages)
# 展示出pdf內含圖片們經過yolo model predict後的結果
@app.route('/predictNew/<filename>')
def predict_pdf(filename):
    # 讀取static/pdf檔名/pdf內含的圖片
    image_folder = os.path.join(app.static_folder, filename)
    images = [image_folder + '/'+ img for img in os.listdir(image_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]
    print(images)
    count = 0

    # 創建資料夾供儲存預測結果
    os.makedirs(os.path.join('static', 'predict', filename), exist_ok=True)
    os.makedirs(os.path.join('static', 'predict', filename, 'crops'), exist_ok=True)
    os.makedirs(os.path.join('static', 'predict', filename, 'possibility'), exist_ok=True)    
    ## 使用 YOLO 模型進行物體檢測
    for image in images:
        img = Image.open(image)
        results = model(img)
        imagecv2 = cv2.imread(image)
        # for result in results:
        #     result_image_name = f'{count}.jpg'
        #     result_image_path = os.path.join('static','predict', filename, result_image_name)
        #     result.save(filename=result_image_path)
        #     count += 1
        for i, result in enumerate(results):
            result_image_name = f'{count}.jpg'
            result_image_path = os.path.join('static','predict', filename,'possibility', result_image_name)
            result.save(filename=result_image_path)
            
            # 上面是原本功能正常的code
            # 以下為測試，提取boundingbox圖
            boxes = result.boxes.xyxy.cpu().numpy()  # xyxy format (x1, y1, x2, y2)
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                crop = imagecv2[y1:y2, x1:x2]
                cv2.imwrite(f"static\\predict\\{filename}\\crops\\{filename}_crop_{count}_{j}.jpg", crop)
            count += 1

    """
    count = 0
    for image in images:
        img = Image.open(image)
        results = model(img)

        for result in results:
            result_image_name = f'{count}.jpg'
            result_image_path = os.path.join('static/predict', result_image_name)
            result.save(filename=result_image_path)
            count += 1
    """
    predict_images_folder = os.path.join('static','predict', filename,'possibility')
    predict_images = ['/'+'predict' +'/'+filename+'/'+'possibility'+'/'+img for img in os.listdir(predict_images_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]
    print(predict_images)
    return render_template('show_predict_pictures.html',filename=filename,images=predict_images)


    # 讀取圖片
    img = Image.open(file.stream)

    # 轉換圖片為 NumPy array
    # img_np = np.array(img)

    # 使用 YOLO 模型進行物體檢測
    # results = model("https://ultralytics.com/images/bus.jpg")

    results = model(img)  # 預測結果
    # results.pandas().xywh   # 輸出檢測結果（用 pandas）

    # Save the result image to disk as result2.jpg
    count = 0
    for result in results:
        result_image_name = f'{count}.jpg'
        result_image_path = os.path.join('static/predict', result_image_name)
        result.save(filename=result_image_path)
        count += 1
        # result.save(filename=result_image_path)  # Save to disk

        # Save the result image
    # result_image_name = 'result2.jpg'
    # result_image_path = os.path.join(RESULT_FOLDER, result_image_name)
    # results.save(filename=result_image_path)
    return render_template('result.html', image_url=f'/static/predict/{result_image_name}')

if __name__ == '__main__':
    app.run(debug=True)

