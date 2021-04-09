from app import db
from datetime import datetime


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    # email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    phone_no = db.Column(db.String(22), nullable=False, unique=True)
    unique_id = db.Column(db.String(60), nullable=False, unique=True)
    data = db.relationship('Data', backref='user', lazy='dynamic')

    def __repr__(self):
        return f"User {self.username} {self.unique_id}"


class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.LargeBinary)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    # prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction = db.relationship('Predictions', backref='data', lazy='dynamic')

    def __repr__(self):
        return f'Data({self.id}, {self.timestamp}, User_id:{self.user_id})'


class Predictions(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    belong_to_class = db.Column(db.String(40), nullable=False)
    confidence = db.Column(db.Float)
    count = db.Column(db.Integer)
    # coordinates = db.Column(db.String(100))
    coordinates = db.relationship('Coordinates', backref='predictions', lazy='dynamic')
    # data = db.relationship('Data', backref='predictions', lazy='dynamic')
    data_id = db.Column(db.Integer, db.ForeignKey('data.id'), nullable=False)


class Coordinates(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    x_min = db.Column(db.Float)
    y_min = db.Column(db.Float)
    x_max = db.Column(db.Float)
    y_max = db.Column(db.Float)
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=False)