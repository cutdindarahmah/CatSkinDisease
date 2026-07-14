import tensorflow as tf
import numpy as np
from pathlib import Path

models = {
    'EfficientNetB1': Path('models/efficientnetb1_cat_skin_disease_final.keras'),
    'MobileNetV2': Path('models/mobilenetv2_cat_skin_disease.h5')
}

for name, path in models.items():
    print('---', name)
    if not path.exists():
        print('MISSING', path)
        continue
    model = tf.keras.models.load_model(str(path))
    print('input_shape', model.input_shape)
    print('output_shape', model.output_shape)
    print('input_dtype', model.input_dtype)
    try:
        last = model.layers[-1]
        print('last_layer', last.name, getattr(last, 'activation', None))
    except Exception as e:
        print('last_layer error', e)
    zeros = np.zeros((1,) + tuple(model.input_shape[1:]), dtype='float32')
    rand = np.random.rand(1, *tuple(model.input_shape[1:])).astype('float32')
    p0 = model.predict(zeros, verbose=0)
    p1 = model.predict(rand, verbose=0)
    print('pred_zero', p0[0], 'argmax', np.argmax(p0[0]))
    print('pred_rand', p1[0], 'argmax', np.argmax(p1[0]))
    print('same output?', np.allclose(p0, p1))

