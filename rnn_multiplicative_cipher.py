# Commented out IPython magic to ensure Python compatibility.
import random
import numpy as np
# %matplotlib inline
import keras 
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN, Input
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Model

key = 13
vocab = [char for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']

def encrypt(text):
    indexes = [vocab.index(char) for char in text]
    encrypted_indexes = [ (idx + key) % len(vocab) for idx in indexes]
    encrypted_chars = [vocab[idx] for idx in encrypted_indexes]
    encrypted = ''.join(encrypted_chars)
    return encrypted

def encrypt_mul(text):
   key = [7,0]
   ASC_A = 64
   WIDTH = 27
   encrypted = []
   for ch in text:
    offset = ord(ch) - ASC_A
    a=chr(((key[0] * (offset + key[1])) % WIDTH) + ASC_A)
    encrypted.append(a)
   encrypted = ''.join(encrypted)
   return encrypted

num_examples = 128
message_length = 64

encrypt_mul('IMPWFTUDGIFGKFWDIPKVOTKHYHOSRXCGTJUVPJLORLMXGVGSENHWQQAWDGRALOKQ')

def dataset(num_examples):
    inp = np.zeros([num_examples,64])
    out = np.zeros([num_examples,64,27])
    for x in range(0, num_examples):
        ex_out = ''.join([random.choice(vocab) for x in range(message_length)])
        #ex_out = "IMPWFTUDGIFGKFWDIPKVOTKHYHOSRXCGTJUVPJLORLMXGVGSENHWQQ-WDGRALOKQ"
        ex_in = encrypt_mul(''.join(ex_out))
        ex_in = [vocab.index(x) for x in ex_in]
        ex_in = np.asarray(ex_in)
        ex_out = [vocab.index(x) for x in ex_out]
        ex_out = np.asarray(ex_out)
        ex_out = to_categorical(ex_out,num_classes=27)
        #dataset.append([(ex_in), (ex_out)])
        inp[x] = ex_in
        out[x] = ex_out
        #print(ex_out)
    return inp, out

inp,out = dataset(500)
print(inp.shape)
print(out.shape)

inp1,out1 = dataset(200)

embedding_dim = 5
hidden_dim = 10
vocab_size = len(vocab)
print(vocab_size)

main_input = Input(shape=(64,), dtype='int32', name='main_input')
x = Embedding(output_dim=5, input_dim=27, input_length=64)(main_input)
x = SimpleRNN(units = 10, activation='tanh', use_bias=True, return_sequences = True)(x)
x = Dense(27, activation='softmax')(x)
model = Model(inputs=main_input, outputs=x)
model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Fitting the RNN to the Training set
history = model.fit([inp], [out], epochs = 100, batch_size = 32, validation_data=([inp1],[out1]))

def dataset(num_examples):
    inp = np.zeros([num_examples,64])
    out = np.zeros([num_examples,64,27])
    for x in range(0, num_examples):
        ex_out = ''.join([random.choice(vocab) for x in range(message_length)])
        #ex_out = "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKL"
        ex_in = encrypt_mul(''.join(ex_out))
        print(ex_in)
        print(ex_out)
        ex_in = [vocab.index(x) for x in ex_in]
        ex_in = np.asarray(ex_in)
        ex_out = [vocab.index(x) for x in ex_out]
        ex_out = np.asarray(ex_out)
        ex_in = to_categorical(ex_in,num_classes=27)
        #dataset.append([(ex_in), (ex_out)])
        out[x] = ex_in
        inp[x] = ex_out
        #print(ex_out)
    return inp, out

inp,out = dataset(1)
print(inp.shape)
print(out.shape)

ex_out = "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKL"
print(len(ex_out))

model.save_weights("b.hdf5")

inp,out = dataset(1)
#print(out)
word = ''
for i in range(64):
  #print(out[0][i])
  a=model.predict([inp])[0][i]
  a=np.asarray(a)
  #a1=to_categorical(a,num_classes=1)
  #word = word.join(vocab[np.where(a == np.amax(a))[0][0]])
  print(vocab[np.where(a == np.amax(a))[0][0]],sep='', end='')

encrypt_mul('IYJKXZCPAIXAQXKPIJQGFZQESEFVROLAZMCGJMUFRUYOAGAVTBEKNNDKPARDUFQN')

model.load_weights("weights.hdf5")

from keras import backend as K

get_2nd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_2nd_layer_output([inp])[0]
print(layer_output)

model.layers[3].get_weights()[0].shape

fileName = 'embed_weights.txt'
fileHandler = open(fileName, "w")

row = model.layers[1].get_weights()[0].shape[0]
col = model.layers[1].get_weights()[0].shape[1]

for i in range(row):
  for j in range(col):
     fileHandler.write(str(model.layers[1].get_weights()[0][i][j])) 
     fileHandler.write('\n')
fileHandler.close()

fileName = 'ih_weights.txt'
fileHandler = open(fileName, "w")

row = model.layers[2].get_weights()[0].shape[0]
col = model.layers[2].get_weights()[0].shape[1]
for j in range(col):
  for i in range(row):
     fileHandler.write(str(model.layers[2].get_weights()[0][i][j])) 
     fileHandler.write('\n')
fileHandler.close()

fileName = 'hh_weights.txt'
fileHandler = open(fileName, "w")

row = model.layers[2].get_weights()[1].shape[0]
col = model.layers[2].get_weights()[1].shape[1]

for j in range(col):
  for i in range(row):
     fileHandler.write(str(model.layers[2].get_weights()[1][i][j])) 
     fileHandler.write('\n')
fileHandler.close()

fileName = 'ih_bias.txt'
fileHandler = open(fileName, "w")

row = model.layers[2].get_weights()[2].shape[0]

for i in range(row):
     fileHandler.write(str(model.layers[2].get_weights()[2][i])) 
     fileHandler.write('\n')
fileHandler.close()

fileName = 'linear_weights.txt'
fileHandler = open(fileName, "w")

row = model.layers[3].get_weights()[0].shape[0]
col = model.layers[3].get_weights()[0].shape[1]
for j in range(col):
  for i in range(row):
     fileHandler.write(str(model.layers[3].get_weights()[0][i][j])) 
     fileHandler.write('\n')
fileHandler.close()

fileName = 'linear_bias.txt'
fileHandler = open(fileName, "w")

row = model.layers[3].get_weights()[1].shape[0]

for i in range(row):
  fileHandler.write(str(model.layers[3].get_weights()[1][i]) )
  fileHandler.write('\n')
fileHandler.close()

plt.plot(loss_list, label="Training Loss")
#plt.plot(accuracies, label="accuracy on validation data set")
plt.grid()
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot.png")

plt.plot(accuracies, label="Accuracy on Val. Data")
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("acc.png")

