import os
import logging as log
import matplotlib.pyplot as plt
import tensorflow as tf
from functools import reduce
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img


log.basicConfig(filename="./exception_log.log",filemode='w')
logger = log.getLogger()
logger.setLevel(log.INFO)

'''File Managment'''

try:
    train_dir = "./Drug Addicted or Not People - DANP/train"
    test_dir = "./Drug Addicted or Not People - DANP/test"
    
    
    
    train_adicts_dir = os.path.join(train_dir,'Addicted')
    train_not_addicts_dir = os.path.join(train_dir,'Not Addicted')
    
    validation_adicts_dir = os.path.join(test_dir,'Addicted')
    validation_not_adicts_dir = os.path.join(test_dir,'Not Addicted')
    
    logger.info("File management successful")

except Exception as ex:
    logger.error("Exception occured ",exc_info=True)



def plotSample(sampledir,index = 0):
    print("Sample image: ")
    plt.imshow(load_img(f"{os.path.join(sampledir,os.listdir(sampledir)[index])}"))
    plt.show()
    
    
try:
    plotSample(train_adicts_dir,index=2)
    
except Exception as ex:
    print(ex)
    

sample_img = load_img(f'{os.path.join(train_adicts_dir,os.listdir(train_adicts_dir)[0])}')
sample_array = img_to_array(sample_img)
print(f'Each image has shape: {sample_array.shape}')

#Image processing

def train_val_generators(training_dir,test_dir):
    train_datagen = ImageDataGenerator(
        rescale= 1/255.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )
    
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        batch_size=32,
        class_mode='binary',
        target_size=(160,160)
        )
    
    test_datagen = ImageDataGenerator(rescale=1/255.)
    
    test_generator = test_datagen.flow_from_directory(
        
        directory=test_dir,
        batch_size=32,
        class_mode='binary',
        target_size=(160,160)
        
        )
    
    return train_generator,test_generator

train_generator,validation_generator = train_val_generators(train_dir,test_dir)


#Callback class
class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>0.950):
            logger.info("Reached %95 accuracy so cancell fitting")
            self.model.stop_training = True




'''Pre trained model setup'''
from tensorflow.keras.applications.inception_v3 import InceptionV3
def create_pre_trained_model():
    pre_trained_model = InceptionV3(input_shape=(160,160,3),
                                    include_top= False, #Leave out the last flly layer
                                    weights = 'imagenet'
                                    )
    #All the layers in the pre_trained model non-trainable
    for layer in pre_trained_model.layers:
        layer.trainable = False
    
    return pre_trained_model
    
def output_of_last_layer(pre_trained_model):
    last_desired_layer = pre_trained_model.get_layer('mixed7')
    print('Last layer output shape : ',last_desired_layer.output_shape)
    last_output = last_desired_layer.output
    print('last layer output :',last_output)
    return last_output

,
'''Create final model'''
def create_model(pre_trained_model,last_output):
    x = layers.Flatten()(last_output)
    x = layers.Dense(1024,activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1,activation='sigmoid')(x)
    
    model = Model(inputs=pre_trained_model.input,outputs=x)
    model.compile(optimizer = RMSprop(learning_rate = 0.001),
                  loss='binary_crossentropy',
                  metrics = ['accuracy']
                  )
    return model

pre_trained_model = create_pre_trained_model()

print(pre_trained_model.summary())

last_output = output_of_last_layer(pre_trained_model)

model = create_model(pre_trained_model, last_output)

logger.info(f"There are {model.count_params()} total parameters in this model")
logger.info(f"There are {sum([w.shape.num_elements() for w in model.trainable_weights])} trainable parameters in this model")


'''Train the model'''

callbacks = myCallBack()

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=100,
    verbose=2,
    callbacks = callbacks
    )



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

#visualize result
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()