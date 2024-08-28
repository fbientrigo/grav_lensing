import tensorflow as tf

@tf.function
def dice_loss_bin(y_true, y_pred):
    mean_true = tf.reduce_mean(y_true)
    mean_pred = tf.reduce_mean(y_pred)
 
    y_true_bin = tf.cast(y_true > mean_true, tf.float32)
    y_pred_bin = tf.cast(y_pred > mean_pred, tf.float32)
 
    intersection = tf.reduce_sum(y_true_bin * y_pred_bin)
    dice_coefficient = (2. * intersection + 1e-7) / (tf.reduce_sum(y_true_bin) + tf.reduce_sum(y_pred_bin) + 1e-7)
 
    return 1 - dice_coefficient


mape_callable = tf.keras.losses.MeanAbsolutePercentageError(
    reduction='sum_over_batch_size',
    name='mean_absolute_percentage_error'
)


def combined_loss(weight_kl=0.1, weight_dice = 32768, weight_mape = 1):
    #  32768 es igual al MAE maximo asumiendo ranogs (-1,1) con 128x128 pixeles
    def loss(y_true, y_pred):
        mae_loss = tf.keras.losses.mae(y_true, y_pred)
        # kl_loss = tf.keras.losses.kld(y_true, y_pred) # requeire modificacion
        dice_loss = dice_loss_bin(y_true, y_pred)
        mape_loss = mape_callable(y_true, y_pred)
        return mae_loss + weight_kl * kl_loss + weight_dice * dice_loss + weight_mape * mape_loss
    return loss

class IdentityLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs

def create_model(learning_rate=1e-4, 
                 h_kernel_size=3, 
                 hidden_filters=64,
                 out_kernel_size=3,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 decay_steps=10000,
                 decay_rate=0.96):
    """
    Modelo que agranda las dimensiones espaciales hasta 256x256 y luego las reduce a 128x128.
    """

    input_layer = tf.keras.layers.Input(shape=(128, 128, 3)) #128x128

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="sigmoid", padding='same')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x) # 256, 256
    x = tf.keras.layers.Conv2D(hidden_filters, (h_kernel_size, h_kernel_size), activation="sigmoid", padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(2 * hidden_filters, (h_kernel_size, h_kernel_size), activation="sigmoid", padding='same')(x)


    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x) # 128x128
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="tanh", padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # output 128x128
    output_image = tf.keras.layers.Conv2D(1, (out_kernel_size, out_kernel_size), activation="tanh", padding='same', name='output_image')(x)


    # Salidas adicionales para las pérdidas, usando la capa de identidad personalizada
    output_mae = IdentityLayer(name='output_mae')(output_image)
    output_mse = IdentityLayer(name='output_mse')(output_image)
    output_logcosh = IdentityLayer(name='output_logcosh')(output_image)
    output_dice = IdentityLayer(name='output_dice')(output_image)

    model = tf.keras.models.Model(inputs=input_layer, outputs=[output_image, output_mae, output_mse, output_logcosh, output_dice])


    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=learning_rate,
      decay_steps=decay_steps,
      decay_rate=decay_rate,
      staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, 
                                         beta_1=beta_1, 
                                         beta_2=beta_2, 
                                         epsilon=epsilon)

    # Compile con múltiples funciones de pérdida
    model.compile(optimizer=optimizer, 
                  loss={'output_image': 'mae',  # La salida original sigue siendo la principal
                        'output_mae': 'mae', 
                        'output_mse': 'mse', 
                        'output_logcosh': 'logcosh', 
                        'output_dice': dicen},_loss_bi
                  loss_weights={'output_image': 1.0,  # Prioridad a la salida principal
                                'output_mae': 0.0,  # Las demás se usan para cálculo de pérdidas solamente
                                'output_mse': 0.0,
                                'output_logcosh': 0.0,
                                'output_dice': 0.0})


    return model
