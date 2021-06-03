from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()


class Person(db.Model):
    __tablename__ = 'people'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(60), nullable=False)
    cpf = db.Column(db.String(11), nullable=False, unique=True)

    # user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    # user = db.relationship('User', back_populates='registered_people')
    training_session_id = db.Column(db.Integer, db.ForeignKey('training_sessions.id'))
    training_session = db.relationship('TrainingSession', back_populates='person')

    def __init__(self, name, cpf, training_session=None):
        self.name = name
        self.cpf = cpf
        self.training_session = training_session


# class User(db.Model):
#     __tablename__ = 'users'
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(60), nullable=False)
#     username = db.Column(db.String(50), nullable=False)
#     hash_password = db.Column(db.String(200), nullable=False)
#     email = db.Column(db.String(100))
#
#     registered_people = db.relationship("Person", back_populates='user')
#
#     def __init__(self, name, username, hash_password, email, registered_people):
#         self.name = name
#         self.username = username
#         self.hash_password = hash_password
#         self.email = email
#         self.registered_people = [Person(person['name']) for person in registered_people]


# class Signature(db.Model):
#     __tablename__ = 'signatures'
#     id = db.Column(db.Integer, primary_key=True)
#     identification_name = db.Column(db.String(20), nullable=False)
#
#     person_id = db.Column(db.Integer, db.ForeignKey('people.id'))
#     person = db.relationship('Person', back_populates='signatures')
#     training_sessions = db.relationship('TraioningSession', back_populates='signature')
#
#     def __init__(self, identification_name):
#         self.identification_name = identification_name


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

    # person_id = db.Column(db.Integer, db.ForeignKey('people.id'), nullable=False)
    person = db.relationship('Person', uselist=False, back_populates='training_session')
    metric_id = db.Column(db.Integer, db.ForeignKey('metrics.id'))
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
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)

    # training_session_id = db.Column(db.Integer, db.ForeignKey('training_sessions.id'), nullable=False)
    training_session = db.relationship('TrainingSession', uselist=False, back_populates='metric')

    def __init__(self, accuracy, loss, recall, f1_score):
        self.accuracy = accuracy
        self.loss = loss
        self.recall = recall
        self.f1_score = f1_score
