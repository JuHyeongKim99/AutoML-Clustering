####################################################################################
# Author: Kim Ju Hyeong, Lee Jun Hyeok, Park Ye Jin, Seo Ji Won                    #
# Dept of SW. School of Computing. Gachon Univ, 2022. Sep                          #        
# Automous clustering based on Scikit-learn libraries                              #
####################################################################################

#requirements
from statistics import quantiles
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
from pyclustering.cluster.clarans import clarans
from pyclustering.utils import timedcall
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

models = {
    "kmeans":KMeans,
    "gmm": GaussianMixture,
    "clarans":clarans,
    "dbscan": DBSCAN,
    "spectral": SpectralClustering
}

k_default = range(2, 13) #default 'K' range from 2 to 12
#default lists for scaler, encoders 
scaler_list_default = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), Normalizer()]
encoder_list_default = [LabelEncoder(), OneHotEncoder(), OrdinalEncoder()]

random_state = 0 #fixed random_state

results_default = {
        'model_name':None,
        'model_param':None,
        'X':None,
        'y':None,
        'silhouette_score_euclid':None,
        'silhouette_score_manhattan':None
    }

key_feature = None

#function automatically scale nummeric value, and encode categorical value
def scale_encode_df(df, scaler, encoder):
    df_ctg = df.select_dtypes(include=['object']).copy() #categorical columns
    df_num = df.select_dtypes(include=['int64', 'float64']).copy() #numeric columns

    df_new_ctg = pd.DataFrame(encoder.fit_transform(df_ctg), columns=df_ctg.columns) #encode categorical columns
    df_new_num = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns) #scale nummeric columns
    
    df_new = pd.concat([df_new_ctg, df_new_num], axis=1) #new df of modified data

    return df_new


def purity_score(y_true, y_pred):
    cf_matrix = confusion_matrix(y_true, y_pred)  # compute confusion matrix

    return np.sum(np.amax(cf_matrix, axis=0)) / np.sum(cf_matrix)

#Cluster with K Means
def model_kmeans(X, model_name ,model_params, key_feature, key_feature_values, quantiles, str_scaler, str_encoder):
    pca = PCA(n_components=2) #to prevent curse of dimensions 
    X = pca.fit_transform(X)
    n_clusters = model_params['n_clusters']
    results = results_default.copy()
    results['model_name'] = model_name
    

    for n in n_clusters:
        kmeans = KMeans(n_clusters=n, random_state=random_state)
        kmeans.fit(X)
        y = kmeans.predict(X)

        results['y'] = y
        pd_X = pd.DataFrame(X)
        results['X'] = pd_X

        results['model_param'] = kmeans.get_params(deep=True)
        results['silhouette_score_euclid'] = silhouette_score(pd_X, y, metric='euclidean')
        results['silhouette_score_manhattan'] = silhouette_score(pd_X, y, metric='manhattan')

        visualize(results, key_feature, key_feature_values, quantiles, str_scaler, str_encoder)
    

#Cluster with Gaussian EM
def model_gmm(X, model_name, model_params, key_feature, key_feature_values, quantiles, str_scaler, str_encoder):
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    results = results_default.copy()
    results['model_name'] = model_name
    

    n_components = model_params['n_components']
    covariance_type = model_params['covariance_type']

    for n in n_components:
        for cov in covariance_type:
            gmm = GaussianMixture(n_components=n, covariance_type=cov)
            gmm.fit(X)
            y = gmm.predict(X)

            results['X'] = pd.DataFrame(X)
            results['y'] = y
            results['model_param'] = gmm.get_params(deep=True)
            results['silhouette_score_euclid'] = silhouette_score(X, y, metric='euclidean')
            results['silhouette_score_manhattan'] = silhouette_score(X, y, metric='manhattan')

            visualize(results, key_feature, key_feature_values, quantiles, str_scaler, str_encoder)
    

#Cluster with Clarans
def model_clarans(X, model_name, model_params, key_feature, key_feature_values, quantiles, str_scaler, str_encoder):
    pca = PCA(n_components=2)
    X = pd.DataFrame(pca.fit_transform(X))
    results = results_default.copy()
    results['model_name'] = model_name
    results['X'] = X


    X_data = X.values.tolist()

    number_clusters = model_params['nuumber_clusters']
    numlocals = model_params['numlocal']
    maxneighbors = model_params['maxneighbor']

    for n in number_clusters:
        for numlocal in numlocals:
            for maxneighbor in maxneighbors:
                clarans_instance = clarans(data=X.to_dict(), number_clusters=n, numlocal=numlocal, maxneighbor=maxneighbor)
                clarans_instance.process()
                clusters = clarans_instance.get_clusters()
                medoids = clarans_instance.get_medoids()
                y = np.zeros(len(key_feature_values)) #y : cluser labels with same length of X

                for cluster in range(np.shape(clusters)[0]):
                    for index in clusters[cluster]:
                        y[index] = cluster
                
                results['y'] = y
                results['model_param'] = clarans_instance.get_cluster_encoding()
                results['silhouette_score_euclid'] = silhouette_score(X, y, metric='euclidean')
                results['silhouette_score_manhattan'] = silhouette_score(X, y, metric='manhattan')
            
                visualize(results, key_feature, key_feature_values, quantiles, str_scaler, str_encoder)

    

#Cluster with DBSCAN
def model_dbscan(X, model_name, model_params, key_feature, key_feature_values, quantiles, str_scaler, str_encoder):
    pca = PCA(n_components=2)
    df_new = pd.DataFrame(pca.fit_transform(X))

    
    results = results_default.copy()
    results['model_name'] = model_name
    results['X'] = X

    eps = model_params['eps']
    min_samples = model_params['min_samples']

    for ep in eps:
        for min_sample in min_samples:
            dbscan = DBSCAN(min_samples=min_sample, eps=ep)
            y = dbscan.fit_predict(df_new)
            
            results['y'] = y
            results['model_param'] = {'eps':ep, 'min_sample':min_sample}
            results['silhouette_score_euclid'] = silhouette_score(X, y, metric='euclidean')
            results['silhouette_score_manhattan'] = silhouette_score(X, y, metric='manhattan')
            visualize(results, key_feature, key_feature_values, quantiles, str_scaler, str_encoder)
    
    

#cluster with SpectralClustering
def model_spectral(X, model_name, model_params, key_feature, key_feature_values, quantiles, str_scaler, str_encoder):
    results = results_default.copy()
    results['model_name'] = model_name
    results['X'] = X

    n_clusters = model_params['n_clusters']
    n_neighbors = model_params['n_neighbors']

    for n in n_clusters:
        for neighbor in n_neighbors:
            spectral = SpectralClustering(n_clusters=n, n_components=n,n_neighbors=neighbor, random_state=random_state)

            y = spectral.fit_predict(X)
            results['y'] = y
            results['model_param'] = spectral.get_params(deep=True)
            results['silhouette_score_euclid'] = silhouette_score(X, y, metric='euclidean')
            results['silhouette_score_manhattan'] = silhouette_score(X, y, metric='manhattan')

            visualize(results, key_feature, key_feature_values, quantiles, str_scaler, str_encoder)

#function to visualize results of each models
def visualize(results, key_feature, key_feature_values, quantiles, str_scaler, str_encoder):

    model_name = results['model_name']
    X = results['X']
    y = results['y']
    model_param = results['model_param']
    silhouette_score_euclid = results['silhouette_score_euclid']
    silhouette_score_manhatthan = results['silhouette_score_manhattan']

    labels_quantile = list(map(str, np.arange(0, quantiles)))
    labeled_key_value = pd.cut(key_feature_values, quantiles, labels=labels_quantile, include_lowest=True)

    new_X = pd.concat([X, labeled_key_value], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    axes[0].set_title("{}, {}, {}".format(model_name, str_scaler, str_encoder))
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("y")
    axes[0].scatter(X.iloc[:, 0], X.iloc[:,1], c=y)

    axes[1].set_title("Key feature value distribution")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel('y')

    sns.scatterplot(ax=axes[1], data=new_X, x=new_X.iloc[:, 0], y=new_X.iloc[:, 1], hue=key_feature)
    plt.show()

    new_X[key_feature] = pd.to_numeric(new_X[key_feature]) #encode type of key feature to int

    ps = purity_score(new_X[key_feature], y)

    print("Model : ", model_name)
    print("Parameter : ", model_param)
    print("purity_score: ", ps)
    print("Euclidean Silhouette score :", silhouette_score_euclid)
    print("Manhattan Silhouette score : ", silhouette_score_manhatthan)



def AutoML(df,**kwargs):
    key_feature_values = []
    scaler_list = kwargs.get('scaler_list', scaler_list_default)

    encoder_list=kwargs.get('encoder_list', encoder_list_default)
    model_list=kwargs.get('models', models.keys())

    k_range =kwargs.get('k_range', k_default)

    quantiles=kwargs.get('quantiles', 5)

    #dictionary of default model parameters
    #key: name of each model
    model_params_default = {
        "kmeans":{
            'n_clusters':k_range,
        },
        "gmm": {
            'n_components':k_range,
            'covariance_type':['full', 'tied', 'diag', 'spherical'],
            'max_iter':[50]
        },
        "clarans":{
            'nuumber_clusters': k_range,
            'numlocal':[1, 2, 3],
            'maxneighbor':[10]
        
        },
        "dbscan": {
            'eps':[0.5, 0.6, 0.7],
            'min_samples':[10, 30, 50],
        },
        "spectral": {
            'n_clusters':k_range,
            'n_neighbors':[10]
        }
    }

    model_params=kwargs.get('model_param',model_params_default) #model paramters dictionary, default: model_params_default

    #key_feature is feature that will not be used for clustering. e.g) medianHouseValue
    key_feature=kwargs.get('key_feature', None)
    if key_feature != None:
        key_feature_values = df[key_feature]
        features=kwargs.get('features', df.columns.drop(key_feature))
    else:
        features=df.columns

    df = df[features] #select features by user



    for scaler in scaler_list:
        for encoder in encoder_list:
            se_df = scale_encode_df(df, scaler, encoder) #modify dataset: encode categorical data, scale nummeric data
            X = se_df

            model_names = models.keys()
            model_functions = {
                'kmeans':model_kmeans,
                'gmm':model_gmm,
                'clarans':model_clarans,
                'dbscan':model_dbscan,
                'spectral':model_spectral
            }

            for model_name in model_list:
                model_functions[model_name](X, model_name, model_params[model_name], key_feature, key_feature_values, quantiles, str(scaler), str(encoder))