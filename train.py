import os
import sys
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

### DL requirements ###
from layers import DenseLayer
from activations import SigmoidLayer
from losses import MSE
from models import Model

lr = 0.001
epochs = 100
batch_size = 32
model_path = 'nn.weights.hieu'
checkpoint_step = 5

mean_losses = []
val_losses = []
mean_acc = []
val_accs = []

def generate_dataset():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(-1, train_images.shape[1] * train_images.shape[2])
    test_images = test_images.reshape(-1, test_images.shape[1] * test_images.shape[2])

    ### Normalizing inputs ###
    train_images = (train_images - 127.5) / 127.5
    test_images  = (test_images - 127.5) / 127.5

    train_labels = train_labels.reshape(-1, 1)
    test_labels = test_labels.reshape(-1, 1)
    train_labels = OneHotEncoder().fit_transform(train_labels).toarray()
    test_labels = OneHotEncoder().fit_transform(test_labels).toarray()

    return train_images, train_labels, test_images, test_labels

def evaluate(model, loss_func, x, y):
    print('[INFO] Evaluating ... ')
    outputs = model.forward(x)
    loss, grad = loss_func(y, outputs)

    predictions = np.argmax(outputs, axis=1)
    labels = np.argmax(y, axis=1)
    accuracy = accuracy_score(predictions, labels)

    return loss, accuracy

def generate_gif(files):
    with imageio.get_writer('images/output.gif', mode='I') as writer:
        for file in files:
            image = imageio.imread(file)
            writer.append_data(image)

def make_sample_prediction(data, model):
    n_samples = 16
    sample_images = []
    data_len = data.shape[0]

    for i in range(n_samples):
        rand_id = np.random.randint(0, data_len)
        image = data[rand_id]
        sample_images.append(image)

    sample_images = np.array(sample_images)
    outputs = model.forward(sample_images)
    predictions = np.argmax(outputs, axis=1)

    fig, ax = plt.subplots(4,4, figsize=(8,8))
    for i in range(n_samples):
        row = i // 4
        col = i %  4
        ax[row][col].xaxis.set_ticklabels([])
        ax[row][col].yaxis.set_ticklabels([])
        ax[row][col].imshow(sample_images[i].reshape(28, 28, 1))
        ax[row][col].set_title(predictions[i])

    plt.show()


X_train, Y_train, X_test, Y_test = generate_dataset()

model = Model(lr=lr)
dense1  = DenseLayer(X_train.shape, n_out=16)
dense2  = DenseLayer((None, 16), n_out=16)
dense3  = DenseLayer((None, 16), n_out=Y_train.shape[1])
sigmoid = SigmoidLayer()

model.add([dense1, dense2, dense3, sigmoid])
mse = MSE()
fig, ax = plt.subplots(1,2, figsize=(10, 5))

if(os.path.exists(model_path)):
    print('[INFO] Loading model from %s' % model_path)
    model.load_model(model_path)

steps_per_epochs = X_train.shape[0] // batch_size
filenames = []
for i in range(epochs):
    losses = []
    acc = []
    with tqdm(total=steps_per_epochs, file=sys.stdout) as pbar:
        for j in range(steps_per_epochs):
            batch_start = j * batch_size
            batch_end   = (j + 1) * batch_size
            batchX = X_train[batch_start: batch_end]
            batchY = Y_train[batch_start: batch_end]
 
            outputs = model.forward(batchX)
            predictions = np.argmax(outputs, axis=1)
            labels = np.argmax(batchY, axis=1)
            accuracy = accuracy_score(predictions, labels)

            loss, grad = mse(batchY, outputs)
            model.backward(grad)

            model.step()
            losses.append(loss)
            acc.append(accuracy)

            pbar.update(1)

    ### Evaluating ###
    val_loss, val_acc = evaluate(model, mse, X_test, Y_test)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    if((i + 1) % checkpoint_step == 0):
        print('[INFO] Checkpointing to %s' % model_path)
        model.save_model(model_path)

    print('[INFO] Epoch #%d, Loss = %.6f, Accuracy = %.6f, Val_Loss = %.6f, Val_acc = %.6f \n' % 
        (i + 1, np.array(losses).mean(), np.array(acc).mean(), val_loss, val_acc))
    mean_losses.append(np.array(losses).mean())
    mean_acc.append(np.array(acc).mean())

    ax[0].clear()
    ax[0].plot(mean_losses, color='blue', label='Training Loss')
    ax[0].plot(val_losses, color='orange', label='Testing Loss')

    ax[1].clear()
    ax[1].plot(mean_acc, color='blue', label='Training Accuracy')
    ax[1].plot(val_accs, color='orange', label='Testing Accuracy')

    ax[0].set_title('Model loss (Train and Val)')
    ax[1].set_title('Model accuracy (Train and Val)')

    ax[0].legend()
    ax[1].legend()

    if(i % 2 == 0):
        filename = 'images/{}.png'.format(i)
        filenames.append(filename)
        fig.savefig(filename)

predictions = np.argmax(model.forward(X_test), axis=1)
labels = np.argmax(Y_test, axis=1)

accuracy = accuracy_score(predictions, labels)
print('[INFO] Final test accuracy : %.4f' % accuracy)

generate_gif(filenames)
print('[INFO] Gif image generated ...')

make_sample_prediction(X_test, model)
print('[INFO] Visualizing sample predictions ...')

ax[0].clear()
ax[1].clear()

ax[0].plot(mean_losses, color='blue', label='Training Loss')
ax[0].plot(val_losses, color='orange', label='Testing Loss')
ax[0].legend()

ax[1].plot(mean_acc, color='blue', label='Training Accuracy')
ax[1].plot(val_accs, color='orange', label='Testing Accuracy')
ax[1].legend()
plt.show()
