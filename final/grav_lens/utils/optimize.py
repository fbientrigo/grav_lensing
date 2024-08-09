import skopt
from skopt import gp_minimize
from skopt.space import Real
from skopt.callbacks import CheckpointSaver
from skopt.utils import use_named_args


import numpy as np
import gc
import time



def define_dimensions():
    dim_lr0 = Real(low=1e-5, high=1e-2, prior='log-uniform', name='lr0')
    dim_lrf = Real(low=1e-4, high=1e-1, prior='log-uniform', name='lrf')
    dim_momentum = Real(low=0.7, high=0.98, name='momentum')
    dim_weight_decay = Real(low=0.0, high=0.001, name='weight_decay')
    dim_warmup_epochs = Real(low=0.0, high=3.0, name='warmup_epochs')
    dim_warmup_momentum = Real(low=0.0, high=0.95, name='warmup_momentum')
    dim_box = Real(low=0.4, high=6.0, name='box')
    dim_cls = Real(low=1.1, high=4.0, name='cls')
    dim_dfl = Real(low=0.4, high=6.0, name='dfl')
    dim_hsv_h = Real(low=0.0, high=0.2, name='hsv_h')
    dim_hsv_s = Real(low=0.0, high=0.9, name='hsv_s')
    dim_hsv_v = Real(low=0.0, high=0.9, name='hsv_v')
    dim_translate = Real(low=0.0, high=0.9, name='translate')
    dim_scale = Real(low=0.0, high=0.95, name='scale')
    dim_fliplr = Real(low=0.0, high=1.0, name='fliplr')
    dim_mosaic = Real(low=0.2, high=0.8, name='mosaic')
    dim_dropout = Real(low=0.0, high=0.8, name='dropout')
    
    dimensions = [
        dim_lr0, dim_lrf, dim_momentum, dim_weight_decay, dim_warmup_epochs, dim_warmup_momentum,
        dim_box, dim_cls, dim_dfl, dim_hsv_h, dim_hsv_s, dim_hsv_v, dim_translate, dim_scale, dim_fliplr, dim_mosaic, dim_dropout
    ]
    return dimensions

def get_random_initialization_points(dimensions, n_points=1):
    """
    # Ejemplo de uso
    dimensions = define_dimensions()
    random_points = get_random_initialization_points(dimensions, n_points=5)
    for i, point in enumerate(random_points):
        print(f"Point {i+1}: {point}")
    """
    points = []
    for _ in range(n_points):
        point = [dim.rvs(random_state=np.random.RandomState()) for dim in dimensions]
        points.append(point)
    return points

def objective_func(dataset_path, dimensions, device, epochs, batch, base_model_path):
    global best_fitness, counter
    best_fitness = 0
    counter = 0

    if not torch.cuda.is_available():
        device='cpu'
    
    @use_named_args(dimensions=dimensions)
    def F_objective(lr0, lrf, momentum, weight_decay, warmup_epochs, warmup_momentum, box, cls, dfl, hsv_h, hsv_s, hsv_v, translate, scale, fliplr, mosaic, dropout):
        global best_fitness, counter

        # Inicializa el modelo
        model = initialize_model(base_model_path)
        # Entrenar el modelo
        # if device == 'cpu': # NO GPU
        #     model.train(data=dataset_path,
        #                 epochs=epochs, batch=batch, cache=False, plots=False, save=False, val=False, 
        #                 optimizer='AdamW', lr0=lr0, lrf=lrf, momentum=momentum, weight_decay=weight_decay, 
        #                 warmup_epochs=warmup_epochs, warmup_momentum=warmup_momentum, 
        #                 box=box, cls=cls, dfl=dfl, hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v, 
        #                 translate=translate, scale=scale, fliplr=fliplr, mosaic=mosaic, 
        #                 dropout=dropout)
        #     results = model.val(dataset_path)
        # else:
        model.train(data=dataset_path,
                    epochs=epochs, batch=batch, cache=False, plots=False, save=False, val=False, 
                    optimizer='AdamW', lr0=lr0, lrf=lrf, momentum=momentum, weight_decay=weight_decay, 
                    warmup_epochs=warmup_epochs, warmup_momentum=warmup_momentum, 
                    box=box, cls=cls, dfl=dfl, hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v, 
                    translate=translate, scale=scale, fliplr=fliplr, mosaic=mosaic, 
                    dropout=dropout, device=device)
        results = model.val(dataset_path, device=device)

        fitness = results.fitness

        # Print the classification accuracy.
        print()
        print("Fitness: {0:.2%}".format(results.fitness))
        print()

        # Guardamos la mejor precision
        if fitness > best_fitness:
            # Salvar modelo
            model.train(data=dataset_path,
                        epochs=1, batch=batch, cache=True,
                        plots=False, save=True, val=False, name=f'best_model_{counter}_')
            best_fitness = fitness

        # Borrar el modelo y los hiperparametro de memoria
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return -fitness
    
    return F_objective

def optimize_hyperparameters(dataset_path, dimensions, default_parameters, device, epochs, batch, base_model_path):
    checkpoint_saver = CheckpointSaver("checkpoint.pkl", compress=9)
    F_objective = objective_func(dataset_path, dimensions, device, epochs, batch, base_model_path)
    start_time = time.time()
    res = gp_minimize(func=F_objective,
                      dimensions=dimensions,
                      acq_func='EI',
                      n_calls=30,
                      x0=default_parameters,
                      callback=[checkpoint_saver])
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Execution time: {execution_time_minutes:.2f} minutes")
    return res
