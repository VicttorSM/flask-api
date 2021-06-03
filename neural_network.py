import os
from imutils import paths
from random import shuffle
import cv2
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
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
    # directory = 'C:/Users/victt/OneDrive/Documents/Trabalhos/TCC 2/Imagens UTSig/{}'.format(person)

    # file_directory = "C:/Users/victt/OneDrive/Documents/Trabalhos/TCC 2/Logs UTSig/{}".format(person)
    # directory = 'C:/Users/Public/postgresql-files/signature_recognition/images/{}'.format(person)
    directory = training_session.directory
    file_directory = 'C:/Users/Public/postgresql-files/signature_recognition/files/{}'.format(
        training_session.person.id)
    # file_name = "log_{}.txt".format(test_number)
    # file_full_path = file_directory + "/" + file_name
    #
    # file_final_name = "log_final_{}.txt".format(test_number)
    # file_final_full_path = file_directory + "/" + file_final_name

    # file_test_x = 'testX_{}.npy'.format(test_number)
    # file_test_x_full_path = file_directory + "/" + file_test_x
    # file_test_y = 'testY_{}.npy'.format(test_number)
    # file_test_y_full_path = file_directory + "/" + file_test_y

    file_model = 'model.h5'
    file_model_full_path = file_directory + "/" + file_model
    training_session.model_path = file_model_full_path
    # ---------------------------------------------------------------------------------------------------------

    Path(file_directory).mkdir(parents=True, exist_ok=True)

    # if os.path.exists(file_final_full_path):
    #     print('Pulou person {}'.format(person))
    #     continue
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
        # plt.imshow(image)
        # plt.show()
        image = cv2.resize(image, (224, 224))
        # plt.imshow(image)
        # plt.show()
        # break
        data.append(image)

        label = label_dict[image_path.split(os.path.sep)[-2]]
        labels.append(label)

    data = np.array(data, dtype="int")  # 0 - 255
    # print(data)
    labels = np.array(labels)

    # particionar os dados entre teste e treinamento
    (train_x, test_x, train_y, test_y_non_categorical) = train_test_split(
        data, labels, test_size=0.10, random_state=42)

    # print('test_x')
    # np.save(file_test_x_full_path, test_x)
    # print('test_y')
    # np.save(file_test_y_full_path, test_y_non_categorical)

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
    # plt.plot(h.history['loss'], label='erro', c="red")
    # plt.xlabel('episodios')
    # plt.legend(loc='upper right')
    # plt.title("Erro na base de treinamento")
    # plt.show()
    # plt.plot(h.history['categorical_accuracy'], label='acuracia', c="blue")
    # plt.xlabel('episodios')
    # plt.legend(loc='lower left')
    # plt.title("Acurácia na base de treinamento")
    # plt.show()

    # ---------------------------------------------------------------------------------------------------------
    score = model.evaluate(test_x, test_y)
    loss = score[0]
    accuracy = score[1]
    print('Erro na base de teste: ', score[0])

    print('Acurácia na base de teste: ', score[1])

    # ---------------------------------------------------------------------------------------------------------
    # image = cv2.imread(directory+'/Original/original_{}_1.png'.format(person))
    # original = image.copy()

    # image = cv2.resize(image, (224,224))
    # image = np.array(image, dtype="int")

    # image = np.expand_dims(image, axis=0)
    # pred = model.predict(image)
    # print(pred)
    # proba = np.max(pred)
    # label = np.argmax(pred)
    # label = "{} : {:.2f}%".format(label, proba*100)

    # plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    # plt.title(label)

    # ---------------------------------------------------------------------------------------------------------
    predictions = model.predict(test_x)
    # predicted_classes = np.argmax(predictions)

    # f1 = f1_score(test_y_non_categorical, predicted_classes)
    # print('f1_score ' + f1)
    #
    # recall = recall_score(test_y_non_categorical, predicted_classes)
    # print('recall_score ' + recall)

    training_session.metric = Metric(accuracy=accuracy, loss=loss, recall=None, f1_score=None)
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


if __name__ == '__main__':
    pass
