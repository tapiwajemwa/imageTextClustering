# from warnings import filterwarnings
from flask import Flask, flash, request, redirect, render_template
# from werkzeug.utils import secure_filename
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import joblib
import base64
from PIL import Image
from io import BytesIO

UPLOAD_FOLDER = './static/images/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'super secret key'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            if 'file' in request.files:
                print('tapinda')
                # model = joblib.load('models/VGG16_model.pkl')
                kmeans = joblib.load('models/image_clustering_model.pkl')
                model = load_model("models/VGG16_model.h5")
                # img = Image.open(request.files['file']).convert('L')
                test_image = Image.open(request.files['file']).convert('L')
                test_image = test_image.resize((224, 224))
                test_image = test_image.convert('L')
                test_image = np.array(test_image) / 255.0
                test_image = np.expand_dims(test_image, axis=-1)
                test_image = np.repeat(test_image, 3, axis=-1)
                test_feature = model.predict(
                    np.expand_dims(test_image, axis=0)).flatten()
                test_label = kmeans.predict(
                    np.array([test_feature.astype(float)]))[0]
                cluster = generate_cluster_image(test_label, file)
                # buffered = BytesIO()
                # cluster.save(buffered, format="JPEG")
                img_str = base64.b64encode(cluster.getvalue())
                image_data = base64.b64encode(
                    cluster.getvalue()).decode('utf-8')
                print(test_label)

                return render_template('main.html', image=image_data)
            else:
                print('hello')
    return render_template('main.html')


def generate_cluster_image(test_label, img):
    labels = np.load('static/labels.npy')
    IMAGE_DIR = 'static/images/Products/'
    # image_filenames = os.listdir(IMAGE_DIR)
    images = joblib.load('static/images/imagesFile')
    # for filename in image_filenames:
    #     image_path = os.path.join(IMAGE_DIR, filename)
    #     image = Image.open(image_path)
    #     images.append(image)
    similar_images = [image for image, label in zip(
        images, labels) if label == test_label]
    w = 10
    h = 10
    fig = plt.figure(figsize=(15, 15), frameon=True)
    fig.suptitle('Similar Items', fontsize=18)
    columns = 6
    rows = 5
    fig.add_subplot(rows, columns, 1)
    plt.imshow(plt.imread(img))
    plt.axis('off')
    plt.title("YOUR INPUT")
    for i in range(1, len(similar_images)):
        if i == 30:
            break
        # img = plt.imread(similar_images[i])
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(similar_images[i])
        plt.axis('off')
        plt.title(f"k = {i}")
    # plt.show()
    fig.savefig('static/images/result.png')
    img_buf = BytesIO()
    fig.savefig(img_buf, format='png')

    # im = Image.open(img_buf)
    # im.show(title="My Image")

    # img_buf.close()
    return img_buf


if __name__ == '__main__':
    app.run()
