import os
from imutils import paths
from random import shuffle
import cv2
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from datetime import datetime
from pathlib import Path

from models import TrainingSession
from models import Metric


def run_neural_network(training_session: TrainingSession, qtd_batches=8, qtd_epochs=200):
    directory = training_session.directory
    file_directory = 'C:/Users/Public/postgresql-files/signature_recognition/files/{}'.format(
        training_session.person.id)

    file_model = 'model.h5'
    file_model_full_path = file_directory + "/" + file_model
    training_session.model_path = file_model_full_path
    # ---------------------------------------------------------------------------------------------------------

    Path(file_directory).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------------------------------------

    print("Carregando labels...")
    label_dict = {}
    class_name_dict = {}

    for i, d in enumerate(sorted(os.listdir(directory))):
        label_dict[d] = i
        class_name_dict[i] = d

    print(label_dict)
    num_classes = len(label_dict)

    print("Carregando imagens...")
    data = []
    labels = []
    images_path = sorted(list(paths.list_images(directory)))
    shuffle(images_path)
    print("caminho: ", images_path)

    # ---------------------------------------------------------------------------------------------------------
    for image_path in tqdm(images_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        data.append(image)

        label = label_dict[image_path.split(os.path.sep)[-2]]
        labels.append(label)

    data = np.array(data, dtype="int")  # 0 - 255
    # print(data)
    labels = np.array(labels)

    # particionar os dados entre teste e treinamento
    (train_x, test_x, train_y, test_y_non_categorical) = train_test_split(
        data, labels, test_size=0.10, random_state=42)

    train_y = to_categorical(train_y, num_classes=num_classes)
    test_y = to_categorical(test_y_non_categorical, num_classes=num_classes)

    # ---------------------------------------------------------------------------------------------------------
    mobile = MobileNet()

    # ---------------------------------------------------------------------------------------------------------
    mobile.summary()

    # ---------------------------------------------------------------------------------------------------------
    x = mobile.layers[-6].output
    output = Dense(units=2, activation='softmax')(x)

    # ---------------------------------------------------------------------------------------------------------
    model = Model(inputs=mobile.input, outputs=output)

    # ---------------------------------------------------------------------------------------------------------
    model.summary()

    # ---------------------------------------------------------------------------------------------------------
    aug = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=False,
        brightness_range=[0.9, 1.1],
        fill_mode="nearest"
    )

    # ---------------------------------------------------------------------------------------------------------
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    batch = qtd_batches
    epochs = qtd_epochs

    # ---------------------------------------------------------------------------------------------------------
    print("Treinamento do modelo...")
    training_session.started_training = True
    h = model.fit(
        aug.flow(train_x,
                 train_y,
                 batch_size=batch
                 ),
        epochs=epochs
    )

    training_session.trained_at = datetime.now()

    # ---------------------------------------------------------------------------------------------------------
    score = model.evaluate(test_x, test_y)
    loss = score[0]
    accuracy = score[1]
    print('Erro na base de teste: ', score[0])

    print('Acurácia na base de teste: ', score[1])

    # ---------------------------------------------------------------------------------------------------------
    predictions = model.predict(test_x)
    predicted_answers = np.argmax(predictions, axis=1)

    f1 = f1_score(test_y_non_categorical, predicted_answers)
    recall = recall_score(test_y_non_categorical, predicted_answers)
    precision = precision_score(test_y_non_categorical, predicted_answers)

    training_session.metric = Metric(accuracy=accuracy, loss=loss, precision=precision, recall=recall, f1_score=f1)
    # ---------------------------------------------------------------------------------------------------------
    for i, pred in enumerate(predictions):
        print('Falso: {:.2f} Original: {:.2f} ({})'.format(pred[0],
                                                           pred[1],
                                                           class_name_dict[test_y_non_categorical[i]]))

    model.save(file_model_full_path)

    return training_session


def verify_signature_authenticity(training_session: TrainingSession, image_path):
    start_image = cv2.imread(image_path)
    model = load_model(training_session.model_path)
    image = start_image.copy()
    os.remove(image_path)

    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype="int")

    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    print(prediction)
    probability = np.max(prediction)
    label = np.argmax(prediction)

    return label, probability
