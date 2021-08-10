import pickle
from tensorflow import keras
import numpy as np
import tensorflow_model_optimization as tfmot

# Load_ensemble_models definition loads ANNs
# from files and stores them in ANN_stack
def load_ensemble_models(noModels):
    modelStack=[None] * noModels
    for i in range(noModels):
        # Define fileName for ANN ensemble (if not in current directory, add path)
        fileName=str(i*100 + 1000) + 'nodes.h5'
        # load model from file
        model_stack[i]=keras.models.load_model(fileName)
        # add ANN to ANN ensemble
        print('>loaded %s' % fileName)
    return model_stack


# Ensemble_regression uses the predictions by all
# ANNs in the ensemble and merges them by taking
# the average or the median
def ensemble_regression(modelEnsemble,inputVector,ensembleMode="mean"):
    stackEnsemble=None
    for model in modelEnsemble:
        if stackEnsemble is None:
            stackEnsemble = model.predict(inputVector)
        else:
            stackEnsemble = np.dstack((stackEnsemble,
                                        model.predict(inputVector)))
    if ensembleMode=="median":
        ensembleOutput=np.median(stackEnsemble,2)
    else:
        ensembleOutput=np.mean(stackEnsemble,2)
    return ensembleOutput


#import scaler
scaler=pickle.load(open("scaler.pickle","rb"))
# import ANNs
numberOfModels=5 #number of ANNs you like to load
ensembleModelList = load_ensemble_models(numberOfModels)
print('Loaded %d models' % len(ensembleModelList))

#example input vector
inputFeatures=np.array([1.383,1.317,4.052,3.704,32.850,37.520,26.3,-13.3])
inputFeatures=np.reshape(inputFeatures,(1,8))

ensembleModus="mean"
ensembleOutput=ensemble_regression(ensembleModelList,
                                 scaler.transform(inputFeatures),ensembleModus)
#example output. True values are: 9.7 3.18 2.6
print('Ensemble Output: A: %.3f\t B: %.3f \tC: %.3f'
      % (ensembleOutput[0,0],ensembleOutput[0,1],ensembleOutput[0,2]))
