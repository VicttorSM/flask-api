from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()


class Person(db.Model):
    __tablename__ = 'people'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(60), nullable=False)
    cpf = db.Column(db.String(11), nullable=False, unique=True)

    training_session_id = db.Column(db.Integer, db.ForeignKey('training_sessions.id'), unique=True)
    training_session = db.relationship('TrainingSession', back_populates='person')

    def __init__(self, name, cpf, training_session=None):
        self.name = name
        self.cpf = cpf
        self.training_session = training_session


class TrainingSession(db.Model):
    __tablename__ = 'training_sessions'
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False)
    trained_at = db.Column(db.DateTime)
    started_training = db.Column(db.Boolean, nullable=False, default=False)
    qtd_real_images_uploaded = db.Column(db.Integer, nullable=False)
    qtd_false_images_uploaded = db.Column(db.Integer, nullable=False)
    directory = db.Column(db.String(200))
    real_images_path = db.Column(db.String(200))
    false_images_path = db.Column(db.String(200))
    model_path = db.Column(db.String(200))

    person = db.relationship('Person', uselist=False, back_populates='training_session')
    metric_id = db.Column(db.Integer, db.ForeignKey('metrics.id'), unique=True)
    metric = db.relationship('Metric', back_populates='training_session')

    def __init__(self,
                 qtd_real_images_uploaded,
                 qtd_false_images_uploaded,
                 real_images_path,
                 false_images_path,
                 directory):
        self.qtd_real_images_uploaded = qtd_real_images_uploaded
        self.qtd_false_images_uploaded = qtd_false_images_uploaded
        self.real_images_path = real_images_path
        self.false_images_path = false_images_path
        self.directory = directory

        self.created_at = datetime.now()
        self.started_training = False


class Metric(db.Model):
    __tablename__ = 'metrics'
    id = db.Column(db.Integer, primary_key=True)
    accuracy = db.Column(db.Float)
    loss = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)

    training_session = db.relationship('TrainingSession', uselist=False, back_populates='metric')

    def __init__(self, accuracy, loss, precision, recall, f1_score):
        self.accuracy = accuracy
        self.loss = loss
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
