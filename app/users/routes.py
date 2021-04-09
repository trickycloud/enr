from flask import jsonify, request, Blueprint, current_app, send_file, make_response
import tensorflow as tf
import numpy as np
from app.users import utils
import cv2
from app.models import User, Data, Predictions, Coordinates
from app import db
from werkzeug.security import generate_password_hash, check_password_hash
import jwt 

users = Blueprint('users', __name__)


@users.route('/')
def home():
    return jsonify({"message": "Welcome"})


@users.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}
    uid = request.values.get("id")
    user = User.query.filter_by(unique_id=uid).first()

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image") and user:
            # read the image in PIL format
            image = request.files["image"].read()
            # original_image = Image.open(io.BytesIO(image))

            image = np.fromstring(image, np.uint8)
            original_image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            # cv2.imwrite('test.jpg', original_image)

            # preprocess the image and prepare it for classification
            image = utils.prepare_image(original_image, input_size=416)

            interpreter = utils.model.load_interpreter()
            input_details = utils.model.input_details()
            output_details = utils.model.output_details()

            # classify the input image and then initialize the list
            # of predictions to return to the client
            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            boxes, pred_conf = utils.filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                  input_shape=tf.constant([utils.input_size, utils.input_size]))

            # preds = utils.model.predict(image)
            # results = imagenet_utils.decode_predictions(preds)
            # data["predictions"] = []

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=utils.iou,
                score_threshold=utils.score
            )

            original_h, original_w, _ = original_image.shape
            bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
            class_names = utils.model.read_labels()
            allowed_classes = list(class_names.values())
            counted_classes = utils.count_objects(pred_bbox, by_class=True, allowed_classes=allowed_classes)
            final_image = utils.draw_bbox(original_image, pred_bbox, counted_classes, allowed_classes=allowed_classes)
            # final_image = Image.fromarray(final_image.astype(np.uint8))
            # final_image = cv2.cvtColor(np.array(final_image), cv2.COLOR_BGR2RGB)
            _, db_image = cv2.imencode('.jpg', final_image)

            d = Data(image=db_image, user_id=user.id)
            db.session.add(d)
            db.session.commit()
            predictions = []
            for i in range(valid_detections.numpy()[0]):
                prediction = dict()
                prediction['class_id'] = int(classes.numpy()[0][i])
                prediction['name'] = class_names[int(classes.numpy()[0][i])]
                prediction['coordinates'] = {}
                prediction['coordinates']['xmin'] = str(bboxes[i][0])
                prediction['coordinates']['ymin'] = str(bboxes[i][1])
                prediction['coordinates']['xmax'] = str(bboxes[i][2])
                prediction['coordinates']['ymax'] = str(bboxes[i][3])
                prediction['confidence'] = str(round(scores.numpy()[0][i], 2))
                predictions.append(prediction)

                p = Predictions(belong_to_class=class_names[int(classes.numpy()[0][i])],
                                confidence=float(scores.numpy()[0][i]),
                                count=counted_classes[class_names[int(classes.numpy()[0][i])]],
                                data_id=d.id)
                db.session.add(p)
                db.session.commit()

                c = Coordinates(x_min=float(bboxes[i][0]), y_min=float(bboxes[i][1]), x_max=float(bboxes[i][2]), y_max=float(bboxes[i][3]),
                                prediction_id=p.id)

                db.session.add(c)
                db.session.commit()

            data["predictions"] = predictions
            data["counts"] = counted_classes

            # indicate that the request was a success
            data["success"] = True
            data["id"] = request.values['id']
    # return the data dictionary as a JSON response
    return jsonify(data)


@users.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if request.method == 'POST':
        password_hash = generate_password_hash(data["password"])
        user = User(username=data["username"],
                    password_hash=password_hash,
                    phone_no=data["phone_no"],
                    unique_id=data["unique_id"]
                    )
        if User.query.filter_by(username=data["username"]).first():
            return jsonify({"message": "This username is taken! Try Using other username."}), 401
        if User.query.filter_by(phone_no=data["phone_no"]).first():
            return jsonify({"message": "Phone number already in use !"}), 401
        if User.query.filter_by(unique_id=data["unique_id"]).first():
            return jsonify({"message": "Unique ID already in use !"}), 401

        db.session.add(user)
        db.session.commit()

        token = jwt.encode({"public_id": user.id}, current_app.config["SECRET_KEY"])
        return jsonify({"message": "User Created Successfully", "token": token.decode("utf-8")}), 201


@users.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if request.method == 'POST':
        if not data.get('username'):
            return jsonify({"message": "Username is missing !"}), 401
        if not data.get('password'):
            return jsonify({"message": "Password is missing !"}), 401
        user = User.query.filter_by(username=data['username']).first()
        if not user:
            return jsonify({"message": "Incorrect Username or Password !"}), 404
        if not check_password_hash(user.password_hash, data['password']):
            return jsonify({"message": "Incorrect Username or Password !"}), 404
        token = jwt.encode({"public_id": user.id}, current_app.config["SECRET_KEY"])
        return jsonify({"message": "Logged in successfully", "token": token.decode("utf-8")})


@users.route('/profile')
@utils.token_required
def get_profile(current_user):
    data = {
        "Username": current_user.username,
        "Phone No": current_user.phone_no,
        "Unique ID": current_user.unique_id,
    }
    return jsonify(data)


@users.route('/get_image')
@utils.token_required
def get_image(current_user):
    data = Data.query.filter_by(user=current_user).order_by(Data.timestamp.desc()).first()
    image = data.image
    response = make_response(image)
    response.headers.set('Content-Type', 'image/jpeg')
    # response.headers.set()
    return response


@users.route('/get_data')
@utils.token_required
def get_data(current_user):
    predictions = Data.query.filter_by(user=current_user).order_by(Data.timestamp.desc()).first().prediction
    res = []
    for prediction in predictions:
        data = {
            'id': prediction.id,
            'class': prediction.belong_to_class,
            'confidence': prediction.confidence,
            'count': prediction.count
        }
        res.append(data)
    return jsonify(res)


@users.route('/get_coordinates/<id>')
@utils.token_required
def get_coordinates(current_user, id):
    coordinates = Coordinates.query.filter_by(prediction_id=id).first()
    data = {
        'x_min': coordinates.x_min,
        'x_max': coordinates.x_max,
        'y_min': coordinates.y_min,
        'y_max': coordinates.y_max,
    }
    return jsonify(data)


