import tensorflow as tf

def combined_loss(weight_kl=0.1):
    def loss(y_true, y_pred):
        mae_loss = tf.keras.losses.mae(y_true, y_pred)
        kl_loss = tf.keras.losses.kld(y_true, y_pred)
        return mae_loss + weight_kl * kl_loss
    return loss

def create_model(learning_rate=1e-4, 
                 in_activation="sigmoid",
                 h_activation="sigmoid", 
                 out_activation="sigmoid",
                 h_kernel_size=3, 
                 hidden_filters=64,
                 out_kernel_size=3,
                 weight_kl=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 decay_steps=10000,
                 decay_rate=0.96):
    """
    Modelo que agranda las dimensiones espaciales hasta 256x256 y luego las reduce a 128x128.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation=in_activation, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2D(hidden_filters, (h_kernel_size, h_kernel_size), activation=h_activation, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(2 * hidden_filters, (h_kernel_size, h_kernel_size), activation=h_activation, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation=h_activation, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(1, (out_kernel_size, out_kernel_size), activation=out_activation, padding='same')
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=learning_rate,
      decay_steps=decay_steps,
      decay_rate=decay_rate,
      staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, 
                                         beta_1=beta_1, 
                                         beta_2=beta_2, 
                                         epsilon=epsilon, 
                                         amsgrad=amsgrad)

    model.compile(optimizer=optimizer, loss=combined_loss(weight_kl))

    return model
