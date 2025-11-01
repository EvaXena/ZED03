#继承layer
import hls4ml
import tensorflow as tf
import keras
import numpy as np
import os

from tensorflow.keras import layers, initializers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

# =================================================================
#           Keras层定义 
# =================================================================
class Tanhlu(layers.Layer):
    def __init__(self, alpha_initializer='ones', beta_initializer='zeros', lambd_initializer='ones', **kwargs):
        super(Tanhlu,self).__init__(**kwargs)
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.lambd_initializer = initializers.get(lambd_initializer)
    def build(self,input_shape):
        self.alpha = self.add_weight(shape=(1,), name='alpha', initializer=self.alpha_initializer, trainable=True)
        self.beta = self.add_weight(shape=(1,), name='beta', initializer=self.beta_initializer, trainable=True)
        self.lambd = self.add_weight(shape=(1,), name='lambd', initializer=self.lambd_initializer, trainable=True)
        super(Tanhlu,self).build(input_shape)
    def get_config(self):
        config = {'alpha_initializer': initializers.serialize(self.alpha_initializer), 'beta_initializer': initializers.serialize(self.beta_initializer), 'lambd_initializer': initializers.serialize(self.lambd_initializer)}
        base_config = super(Tanhlu,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def call(self,inputs):
        return self.alpha * tf.math.tanh(self.lambd * inputs) + self.beta * inputs

# =================================================================
#           hls4ml层实现
# =================================================================

# 1. 继承自最基础的 Layer
class HTanhlu(hls4ml.model.layers.Layer):

    _expected_attributes = [
        hls4ml.model.layers.WeightAttribute('alpha'),
        hls4ml.model.layers.WeightAttribute('beta'),
        hls4ml.model.layers.WeightAttribute('lambd'),
    ]


    def initialize(self):

        inp = self.get_input_variable()
        self.add_output_variable(inp.shape, inp.dim_names)

        alpha_data = self.get_attr('alpha')
        self.add_weights_variable(name='alpha', var_name='alpha{index}', data=alpha_data)
        
        beta_data = self.get_attr('beta')
        self.add_weights_variable(name='beta', var_name='beta{index}', data=beta_data)
        
        lambd_data = self.get_attr('lambd')
        self.add_weights_variable(name='lambd', var_name='lambd{index}', data=lambd_data)


def parse_tanhlu_layer(keras_layer, input_names, input_shapes, data_reader):
    layer = {}
    layer['class_name'] = 'HTanhlu'
    layer['name'] = keras_layer['config']['name']
    

    shape = input_shapes[0][1:]
    if len(shape) == 3: 
        layer['height'] = shape[0]
        layer['width'] = shape[1]
        layer['n_in'] = shape[2] # n_in 是通道数
    else: # 全连接层或其他1D数据
        layer['height'] = 1
        layer['width'] = 1
        layer['n_in'] = np.prod(shape)

    if input_names is not None:
        layer['inputs'] = input_names
    
    # 提取权重
    weights = data_reader.get_weights_data(layer['name'],'alpha')
    layer['alpha'] = weights[0]
    weights = data_reader.get_weights_data(layer['name'],'beta')
    layer['beta'] = weights[0]
    weights = data_reader.get_weights_data(layer['name'],'lambd')
    layer['lambd'] = weights[0]

    return layer, input_shapes

# =================================================================
#           C++ 模板 
# =================================================================

# 3. 恢复需要 height 和 width 的 config 模板
tanhlu_config_template = """struct config{index} : nnet::tanhlu_config {{
    static const unsigned n_in = {n_in};
    static const unsigned height = {height};
    static const unsigned width = {width};
}};\n"""

tanhlu_function_template = 'nnet::tanhlu<{input_t},{result_t},{config}>({input},{output},{alpha},{beta},{lambd});'
tanhlu_include_list = ['nnet_utils/nnet_tanhlu_stream.h']

# =================================================================
#           注册逻辑 
# =================================================================

class HTanhluConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(HTanhlu)
        self.template = tanhlu_config_template
    def format(self, node):
        return self.template.format(**self._default_config_params(node))

class HTanhluFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(HTanhlu, include_header=tanhlu_include_list)
        self.template = tanhlu_function_template
    def format(self, node):
        params = self._default_function_params(node)
        # 4. 确保权重数值被正确提取
        params['alpha'] = node.weights['alpha'].data
        params['beta'] = node.weights['beta'].data
        params['lambd'] = node.weights['lambd'].data
        params['result_t'] = node.get_output_variable().type.name
        return self.template.format(**params)


hls4ml.converters.register_keras_layer_handler('Tanhlu',parse_tanhlu_layer)
hls4ml.model.layers.register_layer('HTanhlu', HTanhlu)
for backend_id in ['Vitis','Vivado']:
    backend = hls4ml.backends.get_backend(backend_id)
    backend.register_template(HTanhluConfigTemplate)
    backend.register_template(HTanhluFunctionTemplate)
    path = os.path.dirname(os.path.abspath(__file__))
    backend.register_source(f"{path}/nnet_utils/nnet_tanhlu_stream.h")

# =================================================================
#           主程序 
# =================================================================
if __name__ == '__main__':

    with custom_object_scope({'Tanhlu': Tanhlu}):
        model = load_model('model/final_model.h5')
    x_batched = np.random.randint(-5, 5, (1, 50, 50, 3), dtype='int32')
    res = model(x_batched)
    hmodel = hls4ml.converters.convert_from_keras_model(model, output_dir='../hls4mlprj_tanhlu_rf_2_zu15eg_v2', backend='Vitis', io_type='io_stream', hls_config={'Model': {'Precision': 'ap_fixed<16,6>', 'ReuseFactor': 2}},clock_period =10,part = 'xczu15eg-ffvb1156-2-e')
    hmodel.compile()
    hres = hmodel.predict(x_batched.astype('float32'))
    print("Keras result shape:", res.shape)
    print("hls4ml result shape:", hres.shape)
    res_flat = res.numpy().flatten()
    hres_flat = hres.flatten()
    print("\nComparing Keras (float) vs hls4ml (fixed-point):")
    print("Keras head:", res_flat[:10])
    print("hls4ml head:", hres_flat[:10])
    np.testing.assert_allclose(res_flat, hres_flat, rtol=1e-2, atol=1e-2)
    print("\nVerification successful (within tolerance)!")
    hmodel.build(
        csim=False,
        synth=True
    )