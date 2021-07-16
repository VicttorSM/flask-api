import os

from time import time
from flask import Flask, jsonify, request, make_response, redirect, render_template
from flask_sqlalchemy import SQLAlchemy
from pathlib import Path
from models import db
from models import Person
from models import TrainingSession
from neural_network import run_neural_network
from neural_network import verify_signature_authenticity


app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:pg123@localhost/signature_verification_db'
app.debug = True

db.init_app(app)


def find_person_by_cpf(cpf):
    person = Person.query.filter_by(cpf=cpf).first()
    return person


def create_object(new_object):
    db.session.add(new_object)
    db.session.commit()
    return True


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":

        if request.files:
            # Creates the person
            name = request.form["name"]
            cpf = request.form["cpf"]
            create_object(Person(name=name, cpf=cpf))

            person = find_person_by_cpf(cpf)

            # Creates directory for this person
            image_directory = 'C:/Users/Public/postgresql-files/signature_recognition/images/{}'.format(person.id)
            original_file_directory = image_directory + '/Original'
            false_file_directory = image_directory + '/Forgery'

            Path(original_file_directory).mkdir(parents=True, exist_ok=True)
            Path(false_file_directory).mkdir(parents=True, exist_ok=True)

            # Retrieves the signatures
            original_signatures = request.files.getlist("original_signatures")
            false_signatures = request.files.getlist("false_signatures")

            # Creates the training session
            training_session = TrainingSession(qtd_real_images_uploaded=len(original_signatures),
                                               qtd_false_images_uploaded=len(false_signatures),
                                               real_images_path=original_file_directory,
                                               false_images_path=false_file_directory,
                                               directory=image_directory)
            training_session.person = person

            create_object(training_session)

            # Saves the signatures in their respective folders
            for original_signature in original_signatures:
                original_signature.save(os.path.join(original_file_directory, original_signature.filename))

            for false_signature in false_signatures:
                false_signature.save(os.path.join(false_file_directory, false_signature.filename))

            start_time = time()
            training_session = run_neural_network(training_session)
            end_time = time()
            delta_time = end_time - start_time
            print('Treinamento durou {:.2f} segundos'.format(delta_time))

            print('Trying to update session')
            db.session.merge(training_session)
            db.session.commit()
            print('updated session')

            return redirect(request.url)

    return render_template("public/upload_image.html")


@app.route("/verify-signature", methods=["GET", "POST"])
def verify_signature():
    if request.method == "POST":

        if request.files:
            # Creates the person
            cpf = request.form["cpf"]
            person = find_person_by_cpf(cpf)

            if person is None:
                print('CPF nao cadastrado')
                return redirect(request.url)

            if person.training_session is None:
                print('This person has no training session yet')
                return redirect(request.url)

            if not person.training_session.started_training:
                print('This person`s training session has not started yet')
                return redirect(request.url)

            if person.training_session.model_path is None:
                print('This person`s training session has not finished yet')
                return redirect(request.url)

            signature = request.files["signature"]

            Path(person.training_session.directory).mkdir(parents=True, exist_ok=True)

            signature.save(os.path.join(person.training_session.directory, signature.filename))

            start_time = time()
            label, probability = verify_signature_authenticity(person.training_session,
                                          os.path.join(person.training_session.directory, signature.filename))
            end_time = time()
            delta_time = end_time - start_time
            print('Tempo para verificar a assinatura: {:.2f} segundos'.format(delta_time))

            print('Label: {}'.format(label))
            print('Probability: {}'.format(probability))
            if label == 0:
                label = 'Falso'
            else:
                label = 'Verdadeiro'
            return render_template("public/verify_signature.html",
                                   label=label,
                                   string='{} ({:.2f}%)'.format(label, probability * 100))

    return render_template("public/verify_signature.html")


@app.route("/")
def home():
    return redirect("http://127.0.0.1:5000/upload-image", code=302)


@app.route('/person', methods=['GET'])
def get_people():
    all_people = Person.query.all()
    output = []
    for person in all_people:
        curr_person = {'id': person.id, 'name': person.name, 'user_id': person.user_id}
        output.append(curr_person)
    return jsonify(output)


@app.route('/person/<id>', methods=['GET'])
def get_person(id):
    person = Person.query.filter_by(id=id).first()
    if person is None:
        return custom_response("This person was not found", 404)
    output = {'id': person.id, 'name': person.name, 'user_id': person.user_id}
    return jsonify(output)


@app.route('/person', methods=['POST'])
def post_people():
    person_data = request.get_json()
    create_object(Person(name=person_data['name'], cpf=person_data['cpf']))
    return custom_response('Person added successfully', 200)


def custom_response(message, status_code):
    dictionary = {"message": message}
    return make_response(jsonify(dictionary), status_code)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with app.app_context():
        db.drop_all()
        db.create_all()
    print('Ended successfully')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
