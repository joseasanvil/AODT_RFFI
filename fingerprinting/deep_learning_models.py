import os
import random
import numpy as np
import tensorflow as tf

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, ReLU, Add, Dense, Conv2D, Flatten, AveragePooling2D
from tensorflow.keras import initializers

# Set the seed value for reproducibility
seed_value = 42

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)

# 5. Force TensorFlow to use deterministic algorithms
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def resblock(x, kernelsize, filters, first_layer=False, seed=None):
    kernel_init = initializers.glorot_uniform(seed=seed)
    if first_layer:
        fx = Conv2D(filters, kernelsize, padding='same', kernel_initializer=kernel_init)(x)
        fx = ReLU()(fx)
        fx = Conv2D(filters, kernelsize, padding='same', kernel_initializer=kernel_init)(fx)
        
        x = Conv2D(filters, 1, padding='same', kernel_initializer=kernel_init)(x)
        
        out = Add()([x, fx])
        out = ReLU()(out)
    else:
        fx = Conv2D(filters, kernelsize, padding='same', kernel_initializer=kernel_init)(x)
        fx = ReLU()(fx)
        fx = Conv2D(filters, kernelsize, padding='same', kernel_initializer=kernel_init)(fx)
              
        out = Add()([x, fx])
        out = ReLU()(out)

    return out 

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)           

class TripletNet():
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        
    def create_net(self, embedding_net, alpha):
        self.alpha = alpha
        
        input_shape = [self.datashape[1], self.datashape[2], self.datashape[3]]
        input_1 = Input(input_shape)
        input_2 = Input(input_shape)
        input_3 = Input(input_shape)
        
        A = embedding_net(input_1)
        P = embedding_net(input_2)
        N = embedding_net(input_3)
   
        loss = Lambda(self.triplet_loss)([A, P, N]) 
        model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
        return model
      
    def triplet_loss(self, x):
        # Triplet Loss function.
        anchor, positive, negative = x
        # Distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor - positive), axis=1)
        # Distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor - negative), axis=1)

        basic_loss = pos_dist - neg_dist + self.alpha
        loss = K.maximum(basic_loss, 0.0)
        return loss

    def feature_extractor(self, datashape):
        self.datashape = datashape
        input_shape = [self.datashape[1], self.datashape[2], self.datashape[3]]
        inputs = Input(shape=input_shape)
        
        kernel_init = initializers.glorot_uniform(seed=seed_value)
        x = Conv2D(32, 7, strides=2, activation='relu', padding='same', kernel_initializer=kernel_init)(inputs)
        
        x = resblock(x, 3, 32, seed=seed_value)
        x = resblock(x, 3, 32, seed=seed_value)

        x = resblock(x, 3, 64, first_layer=True, seed=seed_value)
        x = resblock(x, 3, 64, seed=seed_value)

        x = AveragePooling2D(pool_size=2)(x)
        
        x = Flatten()(x)
    
        x = Dense(512, kernel_initializer=kernel_init)(x)
  
        outputs = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model             

    def get_triplet(self):
        a_label = self.rng.choice(self.dev_range)
        n_label = a_label
        while n_label == a_label:
            n_label = self.rng.choice(self.dev_range)
        a = self.call_sample(a_label)
        p = self.call_sample(a_label)
        n = self.call_sample(n_label)
        return a, p, n

    def call_sample(self, label_name):
        indices = np.where(self.label == label_name)[0]
        idx = self.rng.choice(indices)
        return self.data[idx]

    def create_generator(self, batchsize, dev_range, data, label):
        """Generate a triplets generator for training."""
        self.data = data
        self.label = label
        self.dev_range = dev_range
        
        while True:
            list_a = []
            list_p = []
            list_n = []

            for _ in range(batchsize):
                a, p, n = self.get_triplet()
                list_a.append(a)
                list_p.append(p)
                list_n.append(n)
            
            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N = np.array(list_n, dtype='float32')
            
            # A "dummy" label which will come into our identity loss
            # function below as y_true. We'll ignore it.
            label = np.ones(batchsize)

            yield [A, P, N], label  

class QuadrupletNet():
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        
    def create_net(self, embedding_net, alpha1, alpha2):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        input_shape = [self.datashape[1], self.datashape[2], self.datashape[3]]
        input_1 = Input(input_shape)
        input_2 = Input(input_shape)
        input_3 = Input(input_shape)
        input_4 = Input(input_shape)
        
        A = embedding_net(input_1)
        P = embedding_net(input_2)
        N1 = embedding_net(input_3)
        N2 = embedding_net(input_4)
   
        loss = Lambda(self.quadruplet_loss)([A, P, N1, N2]) 
        model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=loss)
        return model

    def quadruplet_loss(self, x):
        anchor, positive, negative1, negative2 = x

        # Calculate distances
        ap_dist = K.sum(K.square(anchor - positive), axis=1)
        an1_dist = K.sum(K.square(anchor - negative1), axis=1)
        an2_dist = K.sum(K.square(anchor - negative2), axis=1)
        n1n2_dist = K.sum(K.square(negative1 - negative2), axis=1)

        # Calculate loss
        loss1 = K.maximum(ap_dist - an1_dist + self.alpha1, 0)
        loss2 = K.maximum(ap_dist - an2_dist + self.alpha1, 0)
        loss3 = K.maximum(ap_dist - n1n2_dist + self.alpha2, 0)

        return K.mean(loss1 + loss2 + loss3)

    def feature_extractor(self, datashape):
        self.datashape = datashape
        input_shape = [self.datashape[1], self.datashape[2], self.datashape[3]]
        inputs = Input(shape=input_shape)
        
        kernel_init = initializers.glorot_uniform(seed=seed_value)
        x = Conv2D(32, 7, strides=2, activation='relu', padding='same', kernel_initializer=kernel_init)(inputs)
        
        x = resblock(x, 3, 32, seed=seed_value)
        x = resblock(x, 3, 32, seed=seed_value)

        x = resblock(x, 3, 64, first_layer=True, seed=seed_value)
        x = resblock(x, 3, 64, seed=seed_value)

        x = AveragePooling2D(pool_size=2)(x)
        
        x = Flatten()(x)
    
        x = Dense(512, kernel_initializer=kernel_init)(x)
  
        outputs = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model             

    def get_quadruplet(self):
        """Choose a quadruplet (anchor, positive, negative1, negative2) of images
        such that anchor and positive have the same label and
        negatives have different labels from the anchor."""
        a_label = self.rng.choice(self.dev_range)
        n1_label = a_label
        n2_label = a_label

        while n1_label == a_label:
            n1_label = self.rng.choice(self.dev_range)
        while n2_label == a_label or n2_label == n1_label:
            n2_label = self.rng.choice(self.dev_range)

        a = self.call_sample(a_label)
        p = self.call_sample(a_label)
        n1 = self.call_sample(n1_label)
        n2 = self.call_sample(n2_label)

        return a, p, n1, n2
          
    def call_sample(self, label_name):
        """Choose an image from our training or test data with the
        given label."""
        indices = np.where(self.label == label_name)[0]
        idx = self.rng.choice(indices)
        return self.data[idx]

    def create_generator(self, batchsize, dev_range, data, label):
        """Generate a quadruplets generator for training."""
        self.data = data
        self.label = label
        self.dev_range = dev_range
        
        while True:
            list_a = []
            list_p = []
            list_n1 = []
            list_n2 = []

            for _ in range(batchsize):
                a, p, n1, n2 = self.get_quadruplet()
                list_a.append(a)
                list_p.append(p)
                list_n1.append(n1)
                list_n2.append(n2)
            
            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N1 = np.array(list_n1, dtype='float32')
            N2 = np.array(list_n2, dtype='float32')
            
            # A "dummy" label which will come into our identity loss
            # function below as y_true. We'll ignore it.
            label = np.ones(batchsize)

            yield [A, P, N1, N2], label