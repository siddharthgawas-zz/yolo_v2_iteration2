"""
Created on Sun Feb 24 12:56:30 2019

@author: siddh
"""
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPool2D,BatchNormalization,LeakyReLU, Lambda
from config import Configuration
from keras import backend as K
import keras.applications.vgg16 as vgg16
import tensorflow as tf
import numpy as np
from dataset import Dataset

GRID_SIZE = Configuration.GRID_SIZE
ANCHORS = Configuration.ANCHORS
N_ANCHORS = Configuration.ANCHORS.shape[0]
CLASS_LABELS = Configuration.CLASS_LABELS
N_CLASS_LABELS = len(CLASS_LABELS.keys())
LEAKY_RELU_ALPHA = Configuration.LEAKY_RELU_ALPHA
PREDICTIONS_PER_CELL = N_ANCHORS*(5+N_CLASS_LABELS)
GRID_CELL_LOCATIONS = Configuration.GRID_CELL_LOCATIONS
BATCH_SIZE = Configuration.BATCH_SIZE

C_COLUMN_INDICES = []
CENTER_COORD_COLUMN_INDICES = []
DIM_COLUMN_INDICES = []
LABEL_INDICES = []
j = 4
for i in range(N_ANCHORS):
    C_COLUMN_INDICES.append(j)
    CENTER_COORD_COLUMN_INDICES.append(j-4)
    CENTER_COORD_COLUMN_INDICES.append(j-3)
    DIM_COLUMN_INDICES.append(j-2)
    DIM_COLUMN_INDICES.append(j-1)
    for k in range(N_CLASS_LABELS):
        LABEL_INDICES.append(j+k+1)
        
    j = j+N_CLASS_LABELS+5

del j


IMAGE_W = Configuration.IMAGE_WIDTH
IMAGE_H = Configuration.IMAGE_HEIGHT


def output_layer(x):
    import tensorflow as tf
    x = tf.cast(x,dtype=tf.float32)

    c_tensor = tf.gather(x,C_COLUMN_INDICES,axis=3)
    center_coord_tensor = tf.gather(x,CENTER_COORD_COLUMN_INDICES,axis=3)
    dim_tensor = tf.gather(x,DIM_COLUMN_INDICES,axis=3)
    label_tensor = tf.gather(x,LABEL_INDICES,axis=3)
    
    
    center_coord_tensor = tf.reshape(center_coord_tensor,(-1,GRID_SIZE,GRID_SIZE,N_ANCHORS,2))
    dim_tensor = tf.reshape(dim_tensor,(-1,GRID_SIZE,GRID_SIZE,N_ANCHORS,2))
    label_tensor = tf.reshape(label_tensor,(-1,GRID_SIZE,GRID_SIZE,N_ANCHORS,N_CLASS_LABELS))
    
    center_offset = tf.reshape(tf.constant(GRID_CELL_LOCATIONS),(-1,GRID_SIZE,GRID_SIZE,2))
    #BATCH_SIZE if the issue
    center_offset = tf.tile(center_offset,[BATCH_SIZE,1,1,N_ANCHORS])
    center_offset = tf.reshape(center_offset,[-1,GRID_SIZE,GRID_SIZE,N_ANCHORS,2])
    center_offset = tf.cast(center_offset,tf.float32)
    
    dim_offset = tf.reshape(tf.constant(ANCHORS),(-1,N_ANCHORS*2))
    #BATCH SIZE is the issue because it has to be to dynamic
    dim_offset = tf.tile(dim_offset,[BATCH_SIZE,GRID_SIZE*GRID_SIZE])
    dim_offset = tf.reshape(dim_offset,(-1,GRID_SIZE,GRID_SIZE,N_ANCHORS,2))
    dim_offset = tf.cast(dim_offset,tf.float32)
    
    c_tensor = tf.sigmoid(c_tensor)
    center_coord_tensor = tf.sigmoid(center_coord_tensor) + center_offset
    dim_tensor = dim_offset*tf.exp(dim_tensor)
    label_tensor = tf.math.softmax(label_tensor, axis=4)
#    label_tensor = tf.nn.sigmoid(label_tensor)
    
    center_coord_tensor = tf.reshape(center_coord_tensor,(-1,GRID_SIZE,GRID_SIZE,N_ANCHORS*2))
    dim_tensor = tf.reshape(dim_tensor,(-1,GRID_SIZE,GRID_SIZE,N_ANCHORS*2))
    label_tensor = tf.reshape(label_tensor,(-1,GRID_SIZE,GRID_SIZE,N_ANCHORS*N_CLASS_LABELS))
    
    out1 = tf.transpose(c_tensor)
    out2 = tf.transpose(center_coord_tensor)
    out3 = tf.transpose(dim_tensor)
    out4 = tf.transpose(label_tensor)
    
    out1_indices = tf.reshape(tf.constant(C_COLUMN_INDICES,dtype=tf.int32),(out1.shape[0],1))
    out2_indices = tf.reshape(tf.constant(CENTER_COORD_COLUMN_INDICES,dtype=tf.int32),(out2.shape[0],1))
    out3_indices = tf.reshape(tf.constant(DIM_COLUMN_INDICES,dtype=tf.int32),(out3.shape[0],1))
    out4_indices = tf.reshape(tf.constant(LABEL_INDICES,dtype=tf.int32),(out4.shape[0],1))
    
    out1 = tf.scatter_nd(out1_indices,out1,[x.shape[3],GRID_SIZE,GRID_SIZE,BATCH_SIZE])
    out2 = tf.scatter_nd(out2_indices,out2,[x.shape[3],GRID_SIZE,GRID_SIZE,BATCH_SIZE])
    out3 = tf.scatter_nd(out3_indices,out3,[x.shape[3],GRID_SIZE,GRID_SIZE,BATCH_SIZE])
    out4 = tf.scatter_nd(out4_indices,out4,[x.shape[3],GRID_SIZE,GRID_SIZE,BATCH_SIZE])
    
    out1 = tf.transpose(out1)
    out2 = tf.transpose(out2)
    out3 = tf.transpose(out3)
    out4 = tf.transpose(out4)
    
    return  out1+out2+out3+out4

def output_layer_shape(input_shape):
    return input_shape

def get_custom_model():
    input = Input(shape=(IMAGE_H,IMAGE_W,3),dtype=tf.float32)
    layer1 = Conv2D(256,(7,7),strides=(2,2),padding='valid',activation=LeakyReLU(LEAKY_RELU_ALPHA))(input)
    layer1 = BatchNormalization()(layer1)
    
    layer2 = MaxPool2D(pool_size=(3,3),strides=(2,2))(layer1)
    
    layer3 = Conv2D(256,(5,5),strides=(2,2),padding='valid',activation=LeakyReLU(LEAKY_RELU_ALPHA))(layer2)
    layer3 = BatchNormalization()(layer3)
    
    layer4 = MaxPool2D(pool_size=(3,3),strides=(2,2))(layer3)
    
    layer5 = Conv2D(512,(3,3),strides=(1,1),padding='valid',activation=LeakyReLU(LEAKY_RELU_ALPHA))(layer4)
    layer5 = BatchNormalization()(layer5)
    
    layer6 = Conv2D(512,(3,3),strides=(1,1),padding='valid',activation=LeakyReLU(LEAKY_RELU_ALPHA))(layer5)
    layer6 = BatchNormalization()(layer6)
    
    layer7 = Conv2D(1024,(3,3),strides=(1,1),padding='valid',activation=LeakyReLU(LEAKY_RELU_ALPHA))(layer6)
    layer7 = BatchNormalization()(layer7)
    
    layer8 = Conv2D(PREDICTIONS_PER_CELL,(1,1),strides=(1,1),padding='same')(layer7)
    layer8 = BatchNormalization()(layer8)
    
    output = Lambda(output_layer,output_layer_shape)(layer8)   
    model  = Model(inputs=input,outputs=output)
    return model

def get_pre_trained_model():
    input = Input(shape=(IMAGE_H,IMAGE_W,3))
    model = vgg16.VGG16(include_top=False,input_shape=(IMAGE_H,IMAGE_W,3),input_tensor=input)
    model.layers.pop()
    layer = model.layers[-1].output
    layer = Conv2D(256,(3,3),strides=(1,1),padding='valid',activation='relu')(layer)
    layer = Conv2D(256,(3,3),strides=(1,1),padding='valid',activation='relu')(layer)
    layer = Conv2D(256,(3,3),strides=(1,1),padding='valid',activation='relu')(layer)
    layer = Conv2D(256,(3,3),strides=(1,1),padding='valid',activation='relu')(layer)
    layer = Conv2D(256,(1,1),strides=(1,1),padding='valid',activation='relu')(layer)
    layer = Conv2D(PREDICTIONS_PER_CELL,(1,1),strides=(1,1),padding='valid')(layer)
    output = Lambda(output_layer,output_layer_shape)(layer)
    model = Model(inputs=input,outputs=output)
    return model


model = get_pre_trained_model()
model.summary()

#init = tf.global_variables_initializer()
#x_placeholder = tf.placeholder(dtype=tf.float32,shape=(1,3,3,40))
#x = np.random.uniform(size=(1,3,3,40))
#
#dataset = Dataset('data/train','data/test')
#X, y = dataset.next_train_batch()
#y1 = np.reshape(y,(-1,3, 3,40))
#
##indices = tf.constant([[1],[0],[2]])
##updates = tf.linspace(1.0,18.0,18)
##input_t = tf.reshape(updates,(2,3,3))
##updates = tf.transpose(input_t)
##shape = tf.constant([3,3,2])
##scatter = tf.scatter_nd(indices, updates, shape)
##out = tf.transpose(scatter)
##with tf.Session() as sess:
##    input_t,updates,out = sess.run([input_t,updates,out])
#    
#
#with tf.Session() as sess:
#    _, out = sess.run([init,output_layer(x_placeholder)],feed_dict={x_placeholder:y1})
#
#out = np.reshape(out,(1,3*3,40))
#y = np.reshape(y,(1,9,40))
#grid_cell_loc = np.reshape(np.array(GRID_CELL_LOCATIONS),(9,2))

#model = get_pre_trained_model()
#model.summary()
