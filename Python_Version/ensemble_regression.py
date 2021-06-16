import pickle
from tensorflow import keras
import numpy as np
import tensorflow_model_optimization as tfmot

# Load_ensemble_models definition loads ANNs
# from files and stores them in ANN_stack
def load_ensemble_models(n_models):
    model_stack=[None] * n_models
    for i in range(n_models):
        # Define filename for ANN ensemble (if not in current directory, add path)
        filename=str(i*100 + 1000) + 'nodes.h5'
        # load model from file
        model_stack[i]=keras.models.load_model(filename)
        # add ANN to ANN ensemble
        print('>loaded %s' % filename)
    return model_stack


# Ensemble_regression uses the predictions by all
# ANNs in the ensemble and merges them by taking
# the average or the median
def ensemble_regression(model_ensemble,input_vector,ensemble_mode="mean"):
    stack_ensemble=None
    for model in model_ensemble:
        if stack_ensemble is None:
            stack_ensemble = model.predict(input_vector)
        else:
            stack_ensemble = np.dstack((stack_ensemble,
                                        model.predict(input_vector)))
    if ensemble_mode=="median":
        ensemble_output=np.median(stack_ensemble,2)
    else:
        ensemble_output=np.mean(stack_ensemble,2)
    return ensemble_output


#import scaler
scaler=pickle.load(open("scaler.pickle","rb"))
# import ANNs
number_of_models=5 #number of ANNs you like to load
ensemble_model_list = load_ensemble_models(number_of_models)
print('Loaded %d models' % len(ensemble_model_list))

#example input vector
input_features=np.array([1.383,1.317,4.052,3.704,32.850,37.520,26.3,-13.3])
input_features=np.reshape(input_features,(1,8))

ensemble_modus="mean"
ensemble_out=ensemble_regression(ensemble_model_list,
                                 scaler.transform(input_features),ensemble_modus)
#example output. True values are: 9.7 3.18 2.6
print('Ensemble Output: A: %.3f\t B: %.3f \tC: %.3f'
      % (ensemble_out[0,0],ensemble_out[0,1],ensemble_out[0,2]))
