import matplotlib.pyplot as plt

def test_model_image(model, test_dataset, outs=1):
    """
    Se ingresa el modelo y el dataset para hacer testing

    outs corresponde al numero de salidas que tiene el modelo,
    en caso de que el modelo tenga mas de una salida, se tomara la primera output
    """
    for X, Y in test_dataset.take(1):
        if outs == 1:
            prediction = model.predict(X)
        else:
            prediction = model.predict(X)[0]
        


    # Obtener la primera imagen de la predicción
    start = 3
    end = start + 3
    predicted_images = prediction[start:end]
    true_images = Y[start:end]
    print(predicted_images.shape)

    # Configurar la figura
    fig, axes = plt.subplots(2, predicted_images.shape[0], figsize=(20, 10))
    axes = axes.flatten()

    # Graficar cada imagen predicha
    for i, (img_pred, img_true) in enumerate(zip(predicted_images, true_images)):
        axes[i].imshow(img_pred)  # Mostrar imagen predicha
        axes[i].axis('off')
        if i == 0:
            axes[i].set_title("Prediction")
        
        axes[i + predicted_images.shape[0]].imshow(img_true)  # Mostrar imagen verdadera
        axes[i + predicted_images.shape[0]].axis('off')
        if i == 0:
            axes[i + predicted_images.shape[0]].set_title("True")

    plt.show()