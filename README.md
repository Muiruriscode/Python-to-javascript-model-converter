# Mobilenet model
This code create a mobilenet image classifier using tensorflow with keras. The model is then converted to to a tensorflowjs model to be used in javascript. This prodess produces model.json and binary files. The binary files are converted to chunks to enable them to be easily loaded on the browser.

## dependencies
```py
import os
import tensorflow as tf
```

## Create model using Tensorflow and keras
```py
# use mobile net version 2 pass in the input shape the weights and activation function
model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    input_shape = (224, 224, 3), weights='imagenet',
    classifier_activation='softmax'
)
```

## Save the Model
Save the model to a tmp directory
```py
from tensorflow.python.saved_model.save import save

save_dir = os.path.join('/tmp/', 'mobilenetv2/saved_model.h5')
model.save(save_dir)
```

## Install tensorflowjs
```
!pip3 install tensorflowjs
```
## Convert the model
The model is converted to model.json and chunks of binary files for easier loading into the browser
```
!cd /tmp/mobilenetv2/
!tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model /tmp/mobilenetv2/saved_model.h5 /tmp/mobilenetv2
```
## Compress the model
```
!zip -r /tmp/mobilenetv2/mobiledata.zip /tmp/mobilenetv2
```
