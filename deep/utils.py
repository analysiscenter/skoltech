import PIL
import numpy as np
import matplotlib.pyplot as plt

from batchflow import P, R, B, V, W

def run_train(data, model_class, config, description, batch_size, n_iters):
    
    model_config = {
            **config,
        
           'output': {'predicted': ['proba']},
           'loss': 'ce',
           'device': 'gpu',
                }
    
    train_pipeline = (data.p
                    .init_model('dynamic', model_class, 'classification', model_config)
                    .init_variable('loss', [])
                    .to_array(channels='first')
                    .train_model('classification', B('images'), B('labels'), 
                                 fetches='loss', save_to=V('loss', mode='a'))
                    .run_later(batch_size, n_iters=n_iters, drop_last=True, shuffle=42, bar=True)
                    )
    
    train_pipeline.run(batch_size, n_iters=n_iters, bar=True,
                      bar_desc=W(V('loss')[-1].format('Loss is {:7.7}')))
    
    print('{} is done'.format(description))
    return train_pipeline

def run_test(data, train_pipeline, batch_size):
    
    test_pipeline = (data.p
                    .to_array(channels='first')
                    .import_model('classification', train_pipeline)
                    .init_variable('metrics')
                    .predict_model('classification', B('images'), fetches='predicted_proba', 
                                   save_to=B('predictions'))
                    .gather_metrics('class', targets=B.labels, predictions=B.predictions,
                                    fmt='proba', axis=-1, save_to=V('metrics'))
                    .run_later(batch_size, shuffle=True, n_iters=1, drop_last=True, bar=True)
                )
    test_pipeline.run()
    return test_pipeline

def plot_loss(pipeline, description):
    loss = pipeline.v('loss')
    plt.figure(figsize=(10, 5))
    plt.plot(loss)
    plt.grid()
    plt.title(description + ' loss')
    
def plot_images(images, labels=None, proba=None, ncols=5, models_names=None,
                classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck'], **kwargs):
    """ Plot images and optionally true labels as well as predicted class proba.
        - In case labels and proba are not passed, just shows images.
        - In case labels are passed and proba is not, shows images with labels.
        - Otherwise shows everything.
    In case the predictions of several models provided, i.e proba is an iterable containing np.arrays,
    shows predictions for every model.
    Parameters
    ----------
    images : np.array
        batch of images
     labels : array-like, optional
        images labels
     proba: np.array with the shape (n_images, n_classes) or list of such arrays, optional
        predicted probabilities for each class for each model
     ncols: int
        number of images to plot in a row
     classes: list of strings
        class names. In case not specified the list [`1`, `2`, .., `proba.shape[1]`] would be assigned.
     models_names: string or list of strings
        models names. In case not specified and the single model predictions provided will not display any name.
        Otherwise the list [`Model 1`, `Model 2`, ..] is being assigned.
     kwargs : dict
        additional keyword arguments for plt.subplots().
    """
    if isinstance(models_names, str):
        models_names = (models_names, )
    if not isinstance(proba, (list, tuple)):
        proba = (proba, )
        if models_names is None:
            models_names = ['']
    else:
        if models_names is None:
            models_names = ['Model ' + str(i+1) for i in range(len(proba))]

     # if the classes names are not specified they can be implicitely infered from the `proba` shape,
    if classes is None:
        if proba[0] is not None:
            classes = [str(i) for i in range(proba[0].shape[1])]
        elif labels is None:
            pass
        elif proba[0] is None:
            raise ValueError('Specify classes')

    n_items = len(images)
    nrows = (n_items // ncols) + 1
    fig, ax = plt.subplots(nrows, ncols, **kwargs)
    ax = ax.flatten()
    for i in range(n_items):
        ax[i].imshow(images[i])
        if labels is not None: # plot images with labels
            true_class_name = classes[labels[i]]
            title = 'Label: {}'.format(true_class_name)
            if proba[0] is not None: # plot images with labels and predictions
                for j, model_proba in enumerate(proba): # the case of preidctions of several models
                    class_pred = np.argmax(model_proba, axis=1)[i]
                    class_proba = model_proba[i][class_pred]
                    pred_class_name = classes[class_pred]
                    title += '\n {0} pred: {1}, p = {2:.2f}'.format(models_names[j], pred_class_name, class_proba)
            ax[i].title.set_text(title)
        ax[i].grid(b=None)

    for i in range(n_items, nrows * ncols):
        fig.delaxes(ax[i])
        
def segmentation_plot(*args):
    if isinstance(args[0], np.ndarray):
        size = args[0].shape
    else:
        size = args[0].size
        
    img = PIL.Image.new('RGB', (len(args) * size[0] , size[1]))
    
    for i, image in enumerate(args):
        if isinstance(image, np.ndarray):
            try:
                image = PIL.Image.fromarray(image, 'RGB')
            except:
                image = PIL.Image.fromarray(image, 'L')
        img.paste(image, (i * size[0], 0))
    return img

        
        