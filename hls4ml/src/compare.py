import hls4ml.model
import tensorflow as tf
import hls4ml
import keras
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras import layers, initializers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

# ==============================================================================
#                             --- 配置区 ---
# ==============================================================================
KERAS_MODEL_WEIGHTS_PATH = 'model/final_model.h5'
#INPUT_IMAGE_PATH = '../test_img/low_light_image.jpg'
INPUT_DIR_PATH = '../test_img'
TRUTH_DIR_PATH = '../truth'
HLS_OUTPUT_DIR = '../hls4mlprj_tanhlu_enhance'
MODEL_INPUT_SIZE_H = 50
MODEL_INPUT_SIZE_W = 50

# =================================================================
#           Keras层定义 和 hls4ml自定义层实现 
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

class HTanhlu(hls4ml.model.layers.Layer):
    _expected_attributes = [ hls4ml.model.layers.WeightAttribute('alpha'), hls4ml.model.layers.WeightAttribute('beta'), hls4ml.model.layers.WeightAttribute('lambd'), ]
    def initialize(self):
        inp = self.get_input_variable()
        self.add_output_variable(inp.shape, inp.dim_names)
        alpha_data = self.get_attr('alpha'); self.add_weights_variable(name='alpha', var_name='alpha{index}', data=alpha_data)
        beta_data = self.get_attr('beta'); self.add_weights_variable(name='beta', var_name='beta{index}', data=beta_data)
        lambd_data = self.get_attr('lambd'); self.add_weights_variable(name='lambd', var_name='lambd{index}', data=lambd_data)

def parse_tanhlu_layer(keras_layer, input_names, input_shapes, data_reader):
    layer = {}
    layer['class_name'] = 'HTanhlu'
    layer['name'] = keras_layer['config']['name']
    shape = input_shapes[0][1:]
    if len(shape) == 3: layer['height'], layer['width'], layer['n_in'] = shape[0], shape[1], shape[2]
    else: layer['height'], layer['width'], layer['n_in'] = 1, 1, np.prod(shape)
    if input_names is not None: layer['inputs'] = input_names
    layer['alpha'] = data_reader.get_weights_data(layer['name'],'alpha')[0]
    layer['beta'] = data_reader.get_weights_data(layer['name'],'beta')[0]
    layer['lambd'] = data_reader.get_weights_data(layer['name'],'lambd')[0]
    return layer, input_shapes

tanhlu_config_template = "struct config{index} : nnet::tanhlu_config {{ static const unsigned n_in = {n_in}; static const unsigned height = {height}; static const unsigned width = {width}; }};\n"
tanhlu_function_template = 'nnet::tanhlu<{input_t},{result_t},{config}>({input},{output},{alpha},{beta},{lambd});'
tanhlu_include_list = ['nnet_utils/nnet_tanhlu_stream.h']

class HTanhluConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self): super().__init__(HTanhlu); self.template = tanhlu_config_template
    def format(self, node): return self.template.format(**self._default_config_params(node))

class HTanhluFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self): super().__init__(HTanhlu, include_header=tanhlu_include_list); self.template = tanhlu_function_template
    def format(self, node):
        params = self._default_function_params(node)
        params.update({ 'alpha': node.weights['alpha'].data, 'beta': node.weights['beta'].data, 'lambd': node.weights['lambd'].data, 'result_t': node.get_output_variable().type.name })
        return self.template.format(**params)

def register_hls4ml_tanhlu():
    hls4ml.converters.register_keras_layer_handler('Tanhlu',parse_tanhlu_layer)
    hls4ml.model.layers.register_layer('HTanhlu', HTanhlu)
    for backend_id in ['Vitis','Vivado']:
        backend = hls4ml.backends.get_backend(backend_id)
        backend.register_template(HTanhluConfigTemplate)
        backend.register_template(HTanhluFunctionTemplate)
        path = os.path.dirname(os.path.abspath(__file__))
        backend.register_source(f"{path}/nnet_utils/nnet_tanhlu_stream.h")

def dsc_block(input_tensor, filters, name_prefix):
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False, name=f"{name_prefix}_dw")(input_tensor)
    x = layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
    x = layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f"{name_prefix}_pw")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_pw_bn")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu")(x)
    return x

def dsc_block_tanh(input_tensor, filters, name_prefix):
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False, name=f"{name_prefix}_dw")(input_tensor)
    x = layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
    x = layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, name=f"{name_prefix}_pw")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_pw_bn")(x)
    tanhlu = Tanhlu(beta_initializer=initializers.Constant(0.01), name=f'{name_prefix}_tanhlu_2')
    x = tanhlu(x)
    return x

def Model(input_shape=(50, 50, 3), number_f=16):
    model_input = keras.Input(shape=input_shape)
    concat_layer1 = layers.Concatenate(name='Concat_1',axis=-1)
    concat_layer2 = layers.Concatenate(name='Concat_2',axis=-1)
    concat_layer3 = layers.Concatenate(name='Concat_3',axis=-1)
    x1 = dsc_block(model_input, number_f, name_prefix='x1')
    x2 = dsc_block(x1, number_f, name_prefix='x2')
    x3 = dsc_block(x2, number_f, name_prefix='x3')
    x4 = dsc_block(x3, number_f, name_prefix='x4')
    concat_1 = concat_layer1([x3,x4])
    x5 = dsc_block(concat_1, number_f, name_prefix='x5')
    concat_2 = concat_layer2([x2,x5])
    x6 = dsc_block(concat_2, number_f, name_prefix='x6')
    concat_3 = concat_layer3([x1,x6])
    x7 = dsc_block_tanh(concat_3, 3, name_prefix='x7')
    model = keras.Model(inputs=model_input, outputs=x7, name='Keras_DCE_Net')
    return model

def enhance(raw_image, curve_tensor):
    enhanced_image = raw_image
    for _ in range(8):
        enhanced_image = enhanced_image + curve_tensor * (tf.square(enhanced_image) - enhanced_image)
    return enhanced_image

# =================================================================
#           *** 新增的辅助函数 ***
# =================================================================
def print_hls_model_details(hls_model):
    """
    遍历编译后的hls4ml模型，打印每一层的输出和权重的精度。
    """
    print("=" * 60)
    print(" hls4ml Model Layer-wise Precision Details")
    print(" Precision format: ap_fixed<Total Bits, Integer Bits>")
    print("=" * 60)
    layers = hls_model.get_layers()
    for layer in layers:
        layer_name = layer.name
        layer_class = layer.__class__.__name__
        print(f"\n- Layer: {layer_name} (Class: {layer_class})")
        output_var = layer.get_output_variable()
        output_precision = output_var.type.name
        print(f"  - Output Precision : {output_precision}")
        if layer.weights:
            print("  - Weight Precisions:")
            for weight_name, weight_variable in layer.weights.items():
                weight_precision = weight_variable.type.name
                print(f"    - {weight_name:<10}: {weight_precision}")
        else:
            print("  - No trainable weights in this layer.")
    print("\n" + "=" * 60)


def load_and_progress_imgs(INPUT_DIR_PATH,MODEL_INPUT_SIZE_H,MODEL_INPUT_SIZE_W):
    imgs = []
    print(f"\n--- 步骤 4: 加载并预处理图像 '{INPUT_DIR_PATH}' ---")
    if not os.path.exists(INPUT_DIR_PATH):
        raise FileNotFoundError(f"错误: 找不到输入图像 '{INPUT_DIR_PATH}'")
    for root,dirs,files in os.walk(INPUT_DIR_PATH):
        for filename in files:
            img = []
            filepath = os.path.join(root,filename)
            print(filepath)
            truthname = filename[:-4] + '_SeaErra.jpg'
            truth_path =  os.path.join('../truth/',truthname)
            print(truth_path)

            truth_raw = tf.io.read_file(truth_path)
            truth_image = tf.io.decode_image(truth_raw, channels=3, expand_animations=False)
            truth_image.set_shape([None, None, 3])
            img_raw = tf.io.read_file(filepath)
            original_image = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
            original_image.set_shape([None, None, 3])
            image_for_model = tf.image.resize(original_image, [MODEL_INPUT_SIZE_H, MODEL_INPUT_SIZE_W])
            image_for_model = tf.cast(image_for_model, tf.float32) / 255.0
            original_shape = tf.shape(original_image)[0:2]
            input_tensor_tf = tf.expand_dims(image_for_model, axis=0)
            print(f"预处理完成。送入模型的张量形状: {input_tensor_tf.shape}")
            img.append(input_tensor_tf)
            img.append(original_image)
            img.append(truth_image)
            imgs.append(img)
            
    return imgs

# =================================================================
#           主程序 (已更新)
# =================================================================
if __name__ == '__main__':
    # 步骤 0: 注册
    register_hls4ml_tanhlu()

    # 步骤 1: 加载Keras模型
    print("--- 步骤 1: 加载预训练的 Keras 模型 ---")
    if not os.path.exists(KERAS_MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"错误: 找不到Keras模型权重 '{KERAS_MODEL_WEIGHTS_PATH}'")
    with custom_object_scope({'Tanhlu': Tanhlu}):
        keras_model = load_model(KERAS_MODEL_WEIGHTS_PATH)
    print("Keras 模型加载成功。")

    # 步骤 2: 转换Keras模型为hls4ml模型
    print("\n--- 步骤 2: 将 Keras 模型转换为 hls4ml 模型 ---")
    hls_config = {'Model': {'Precision': 'ap_fixed<16,6>', 'ReuseFactor': 128}}
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model, output_dir=HLS_OUTPUT_DIR, backend='Vitis',
        io_type='io_stream', hls_config=hls_config, clock_period=10,
        part='xcu250-figd2104-2L-e'
    )
    print("hls4ml 模型转换完成。正在编译...")
    hls_model.compile()
    print("hls4ml 模型编译成功。")

    # *** 新增步骤 ***: 打印hls4ml模型精度详情
    print("\n--- 步骤 3: 检查 hls4ml 模型各层精度 ---")
    print_hls_model_details(hls_model)


##加载处理单张图片
##-----------------------------------------------------------------------------------------------------
    # # 步骤 4: 加载并预处理图像
    # print(f"\n--- 步骤 4: 加载并预处理图像 '{INPUT_IMAGE_PATH}' ---")
    # # ... (后续代码与之前版本相同)
    # if not os.path.exists(INPUT_IMAGE_PATH):
    #     raise FileNotFoundError(f"错误: 找不到输入图像 '{INPUT_IMAGE_PATH}'")
    # img_raw = tf.io.read_file(INPUT_IMAGE_PATH)
    # original_image = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
    # original_image.set_shape([None, None, 3])
    # image_for_model = tf.image.resize(original_image, [MODEL_INPUT_SIZE_H, MODEL_INPUT_SIZE_W])
    # image_for_model = tf.cast(image_for_model, tf.float32) / 255.0
    # original_shape = tf.shape(original_image)[0:2]
    # input_tensor_tf = tf.expand_dims(image_for_model, axis=0)
    # print(f"预处理完成。送入模型的张量形状: {input_tensor_tf.shape}")
##-----------------------------------------------------------------------------------------------------
##加载处理多张图片
##-----------------------------------------------------------------------------------------------------
    imgs = load_and_progress_imgs(INPUT_DIR_PATH,MODEL_INPUT_SIZE_H,MODEL_INPUT_SIZE_W)
##-----------------------------------------------------------------------------------------------------

    
##单张图像预测
##-----------------------------------------------------------------------------------------------------
    # 步骤 5: 预测
    # print("\n--- 步骤 5a: 使用 Keras 模型(软件)预测曲线图 ---")
    # curve_map_keras_tf = keras_model.predict(input_tensor_tf)
    # print("\n--- 步骤 5b: 使用 hls4ml 模型(硬件)预测曲线图 ---")
    # input_tensor_np = input_tensor_tf.numpy()
    # curve_map_hls_np = hls_model.predict(input_tensor_np)

    # # 步骤 6: 增强
    # original_image_normalized_tf = tf.cast(original_image, tf.float32) / 255.0
    # original_image_normalized_tf = tf.expand_dims(original_image_normalized_tf, axis=0)
    # print("\n--- 步骤 6a: 应用 enhance 函数 (Keras 结果) ---")
    # curve_map_keras_resized_tf = tf.image.resize(curve_map_keras_tf, original_shape, method=tf.image.ResizeMethod.BILINEAR)
    # enhanced_image_keras_tensor = enhance(original_image_normalized_tf, curve_map_keras_resized_tf)
    # print("Keras 结果增强完成。")
    # print("\n--- 步骤 6b: 应用 enhance 函数 (hls4ml 结果) ---")
    # curve_map_hls_np_reshaped = curve_map_hls_np.reshape((1, MODEL_INPUT_SIZE_H, MODEL_INPUT_SIZE_W, 3))
    # curve_map_hls_tf = tf.convert_to_tensor(curve_map_hls_np_reshaped, dtype=tf.float32)
    # curve_map_hls_resized_tf = tf.image.resize(curve_map_hls_tf, original_shape, method=tf.image.ResizeMethod.BILINEAR)
    # enhanced_image_hls_tensor = enhance(original_image_normalized_tf, curve_map_hls_resized_tf)
    # print("hls4ml 结果增强完成。")

    # # 步骤 7: 可视化
    # print("\n--- 步骤 7: 生成并显示最终对比结果 ---")
    # enhanced_image_keras_np = np.squeeze(enhanced_image_keras_tensor.numpy(), axis=0)
    # enhanced_image_keras_clipped = np.clip(enhanced_image_keras_np, 0, 1)
    # enhanced_image_hls_np = np.squeeze(enhanced_image_hls_tensor.numpy(), axis=0)
    # enhanced_image_hls_clipped = np.clip(enhanced_image_hls_np, 0, 1)


    # #保存结果
    # os.makedirs("../result_img",exist_ok=True)
    # plt.imsave("../result_img/original_img.jpg",original_image.numpy())
    # plt.imsave("../result_img/keras_img.jpg",enhanced_image_keras_clipped)
    # plt.imsave("../result_img/hls_img.jpg",enhanced_image_hls_clipped)



    # plt.figure(figsize=(18, 6))
    # plt.subplot(1, 3, 1); plt.imshow(original_image); plt.title('Original Image'); plt.axis('off')
    # plt.subplot(1, 3, 2); plt.imshow(enhanced_image_keras_clipped); plt.title('Enhanced (Keras Model)'); plt.axis('off')
    # plt.subplot(1, 3, 3); plt.imshow(enhanced_image_hls_clipped); plt.title('Enhanced (hls4ml Model)'); plt.axis('off')
    # plt.tight_layout(); plt.show()
##-----------------------------------------------------------------------------------------------------


##多张图像预测
##-----------------------------------------------------------------------------------------------------
    filename = 0
    for img in imgs:
        input_tensor_tf = img[0]
        original_image = img[1]
        truth_image = img[2]
        
        original_shape = tf.shape(original_image)[0:2]
        print("\n--- 步骤 5a: 使用 Keras 模型(软件)预测曲线图 ---")
        curve_map_keras_tf = keras_model.predict(input_tensor_tf)
        print("\n--- 步骤 5b: 使用 hls4ml 模型(硬件)预测曲线图 ---")
        input_tensor_np = input_tensor_tf.numpy()
        curve_map_hls_np = hls_model.predict(input_tensor_np)

        # 步骤 6: 增强
        original_image_normalized_tf = tf.cast(original_image, tf.float32) / 255.0
        original_image_normalized_tf = tf.expand_dims(original_image_normalized_tf, axis=0)
        print("\n--- 步骤 6a: 应用 enhance 函数 (Keras 结果) ---")
        curve_map_keras_resized_tf = tf.image.resize(curve_map_keras_tf, original_shape, method=tf.image.ResizeMethod.BILINEAR)
        enhanced_image_keras_tensor = enhance(original_image_normalized_tf, curve_map_keras_resized_tf)
        print("Keras 结果增强完成。")
        print("\n--- 步骤 6b: 应用 enhance 函数 (hls4ml 结果) ---")
        curve_map_hls_np_reshaped = curve_map_hls_np.reshape((1, MODEL_INPUT_SIZE_H, MODEL_INPUT_SIZE_W, 3))
        curve_map_hls_tf = tf.convert_to_tensor(curve_map_hls_np_reshaped, dtype=tf.float32)
        curve_map_hls_resized_tf = tf.image.resize(curve_map_hls_tf, original_shape, method=tf.image.ResizeMethod.BILINEAR)
        enhanced_image_hls_tensor = enhance(original_image_normalized_tf, curve_map_hls_resized_tf)
        print("hls4ml 结果增强完成。")

        # 步骤 7: 可视化
        print("\n--- 步骤 7: 生成并显示最终对比结果 ---")
        enhanced_image_keras_np = np.squeeze(enhanced_image_keras_tensor.numpy(), axis=0)
        enhanced_image_keras_clipped = np.clip(enhanced_image_keras_np, 0, 1)




        enhanced_image_hls_np = np.squeeze(enhanced_image_hls_tensor.numpy(), axis=0)
        enhanced_image_hls_clipped = np.clip(enhanced_image_hls_np, 0, 1)


        #保存结果
        os.makedirs("../result_img",exist_ok=True)
        plt.imsave(f"../result_img/{str(filename)}_img.jpg",original_image.numpy())
        plt.imsave(f"../result_img/{str(filename)}_truth_img.jpg",truth_image.numpy())
        plt.imsave(f"../result_img/{str(filename)}_hls_img.jpg",enhanced_image_hls_clipped)

        filename = filename + 1

        # plt.figure(figsize=(18, 6))
        # plt.subplot(1, 3, 1); plt.imshow(original_image); plt.title('Original Image'); plt.axis('off')
        # plt.subplot(1, 3, 2); plt.imshow(enhanced_image_keras_clipped); plt.title('Enhanced (Keras Model)'); plt.axis('off')
        # plt.subplot(1, 3, 3); plt.imshow(enhanced_image_hls_clipped); plt.title('Enhanced (hls4ml Model)'); plt.axis('off')
        # plt.tight_layout(); plt.show()

##-----------------------------------------------------------------------------------------------------