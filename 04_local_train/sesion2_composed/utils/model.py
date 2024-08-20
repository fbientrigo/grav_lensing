import tensorflow as tf

dice_loss_callable = tf.keras.losses.Dice(
    reduction='sum_over_batch_size', name='dice'
)

mape_callable = tf.keras.losses.MeanAbsolutePercentageError(
    reduction='sum_over_batch_size',
    name='mean_absolute_percentage_error'
)



def combined_loss(weight_kl=0.1, weight_dice = 32768, weight_mape = 1):
    #  32768 es igual al MAE maximo asumiendo ranogs (-1,1) con 128x128 pixeles
    def loss(y_true, y_pred):
        mae_loss = tf.keras.losses.mae(y_true, y_pred)
        kl_loss = tf.keras.losses.kld(y_true, y_pred)
        dice_loss = dice_loss_callable(y_true, y_pred)
        mape_loss = mape_callable(y_true, y_pred)
        return mae_loss + weight_kl * kl_loss + weight_dice * dice_loss + weight_mape * mape_loss
    return loss

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
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="sigmoid", padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2D(hidden_filters, (h_kernel_size, h_kernel_size), activation="sigmoid", padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(2 * hidden_filters, (h_kernel_size, h_kernel_size), activation="sigmoid", padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="tanh", padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(1, (out_kernel_size, out_kernel_size), activation="tanh", padding='same')
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=learning_rate,
      decay_steps=decay_steps,
      decay_rate=decay_rate,
      staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, 
                                         beta_1=beta_1, 
                                         beta_2=beta_2, 
                                         epsilon=epsilon)

    model.compile(optimizer=optimizer, loss=combined_loss(1))

    return model
