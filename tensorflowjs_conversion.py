import os
import tensorflow as tf

model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    input_shape = (224, 224, 3), weights='imagenet',
    classifier_activation='softmax'
)

from tensorflow.python.saved_model.save import save

save_dir = os.path.join('/tmp/', 'mobilenetv2/saved_model.h5')
model.save(save_dir)

!pip3 install tensorflowjs

!cd /tmp/mobilenetv2/
!tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model /tmp/mobilenetv2/saved_model.h5 /tmp/mobilenetv2

!zip -r /tmp/mobilenetv2/mobiledata.zip /tmp/mobilenetv2