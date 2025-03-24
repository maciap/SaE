import numpy as np 
import pandas as pd 
from surprise import Dataset
from PIL import Image
import pickle
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import LabelEncoder
import scipy 
import sys 


def read_data(data_string): 
    ''' function to read real-world datasets
    @params: 
    data_string: input dataset name 
    
    returns: 
    D: real-world matrix dataset 
    '''

    elif data_string == "movielens": 
        data = Dataset.load_builtin('ml-100k')
        trainset = data.build_full_trainset()
        testset = trainset.build_testset()
        D = np.zeros((trainset.n_users, trainset.n_items))
        for (uid, iid, rating) in trainset.all_ratings():
            D[uid - 1, iid - 1] = rating

    elif data_string == "cameraman": 
        image = Image.open("real_datasets/cameraman.png")
        # Convert the image to a NumPy array
        D = np.array(image)

    elif data_string == "isolet": 
        D = np.load( "real_datasets/isolet_array.npy" ) 

    elif data_string == "olivetti": 
        faces = fetch_olivetti_faces(shuffle=True, random_state=42)
        D = faces.data  # Each row is a flattened 64x64 face image

    elif data_string == "orlRnSp": 
        mat_data = scipy.io.loadmat('real_datasets/orlRnSp.mat')
        D = mat_data["X1"].astype(float)

    elif data_string == "al_genes": 
        train_df = pd.read_csv("real_datasets/data_set_ALL_AML_train.csv")
        train_columns = [col for col in train_df if "call" not in col]
        train_adjusted = train_df[train_columns]
        numeric_columns = train_adjusted.select_dtypes(include='number')
        D = numeric_columns.to_numpy() 
    
    elif data_string == "mandrill": 
        image = Image.open("real_datasets/mandrill.tiff")
        gray_image = image.convert("L")
        D = np.array(gray_image)

    elif data_string == "ozone": 
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/onehr.data"
        df_ozone = pd.read_csv(url, header=None)
        X_ozone = df_ozone.iloc[:, 1:-1].values  
        X_ozone[X_ozone == "?"] = 0.000001         
        D =  X_ozone.astype("float")

    elif data_string == "BRCA1": 
        df = pd.read_csv("real_datasets/BRCA1_miRNASeqGene-20160128.csv")
        D= df.to_numpy()[:,1:].astype(float)
    
    elif data_string == "google_reviews": 
        df = pd.read_csv('real_datasets/google_review_ratings.csv')
        df = df.iloc[:, 1:]
        float_columns = df.select_dtypes(include=['float'])
        column_means = float_columns.mean()
        float_columns = float_columns.fillna(column_means)
        D = float_columns.to_numpy().astype(float)

    elif data_string == "NPAS": 
        data = pd.read_csv("real_datasets/NPAS-data.csv", sep = "," , )
        data_columns = data.select_dtypes(include=['int64', 'float64'])
        D = np.array(data_columns)
        
    elif data_string == "movie_trust": 
        data = np.loadtxt("real_datasets/ratings.txt", dtype=int)  # Adjust delimiter if needed

        # Extract columns
        user_ids = data[:, 0]
        item_ids = data[:, 1]
        ratings = data[:, 2]

        num_users = user_ids.max() + 1  
        num_items = item_ids.max() + 1  

        user_item_matrix = np.zeros((num_users, num_items)) 

        for user, item, rating in zip(user_ids, item_ids, ratings):
            user_item_matrix[user, item] = rating 

        row_sums = np.abs(user_item_matrix).sum(axis=1) 
        col_sums = np.abs(user_item_matrix).sum(axis=0)  

        top_rows = np.argsort(row_sums)[-200:] 
        top_cols = np.argsort(col_sums)[-200:]

        D = user_item_matrix[np.ix_(top_rows, top_cols)]


    elif data_string== "hearth": 
        data = pd.read_csv("real_datasets/heart_disease_uci.csv")
        label_encoder = LabelEncoder()
        # Iterate over each column with "object" data type
        for column in data.select_dtypes(include=['object']):
            data[column] = label_encoder.fit_transform(data[column])
            
        D = data.to_numpy()[:,1:]
        nan_mask = np.isnan(D)
        num_nan_values = np.sum(nan_mask)
        column_means = np.nanmean(D, axis=0)
        D[nan_mask] = np.take(column_means, np.where(nan_mask)[1])


    elif data_string == "imagenet": 
        filepath = "real_datasets/VGG16_imagenet_matrix_weights"
        with open(filepath, "rb") as fileim: 
            matrix_weights = pickle.load(fileim)
        D = matrix_weights

    else: 
        print("invalid data string.") 
        sys.exit(1)       

    return D 


def read_synethetic_data(dist, nrows, nrowssubmatrix, noise_level_string, matrix_counter): 
    ''' function to read real-world datasets
    @params: 
    dist: distribution 
    nrows: number of rows and columns in the matirx 
    nrowssubmatrix: number of rows and columns in the ground-truth submatrix 
    noise_level_string: level of noise 
    matrix_counter: id of the dataset 

    returns: 
    D: synthetic matrix dataset 
    '''

    data_path = "data/synthetic_datasets/"  + dist + "_" + str(nrowssubmatrix) + "_" + str(nrows) + "_"  + str(noise_level_string)  + "_" + str(matrix_counter)
    base_path = data_path + "/"

    # File path 
    file_path =  base_path + "array_data_" + str(nrowssubmatrix) + "_" + str(nrows) + "_"  + str(noise_level_string)  + "_" + str(matrix_counter) + ".npy"

    # Generated matrix to read 
    D = np.load(file_path)

    return D
