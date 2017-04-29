from scipy.sparse import bsr_matrix,csc_matrix,lil_matrix,csr_matrix
import numpy as np

def generate_sparse_adj(mask):
    size = len(mask)
    sparse = lil_matrix((size*100, size*100), dtype=np.int8)
    for i,x in enumerate(mask):
        current = np.load('../../../../data/new_data/matrices/adjacency/a_{}.npy'.format(x))        
        current[:,98] = [1 for i in range(100)]
        transp = np.transpose(current)
        current += transp
        np.fill_diagonal(current, 1)
        current_sparse = lil_matrix(current)
        sparse[i*100:i*100+100,i*100:i*100+100] = current_sparse
    return sparse

def generate_features(mask):
    size = len(mask)
    features = np.ones(size*100)
    for i,x in enumerate(mask):
        current = np.load('../../../../data/new_data/matrices/features/f_{}.npy'.format(x))
        current = [i +1 for i in current]
        features[i*100:i*100+100] = current
    features.astype('float32')
    features = np.reshape(features,(len(features),1))
    features = csr_matrix(features)
    return features

def load_labels(mask):
    labels = np.load('../../../../data/tiny/tinyY.npy')
    masked_labels = []
    for i in mask:
        for j in range(100):
            temp = [1 if k == labels[i] else 0 for k in range(39)]
            masked_labels.append(temp)
    masked_labels = np.asarray(masked_labels,dtype='float32')
    return masked_labels


def gen_mask(size=26342,n=26342):
    mask = np.random.randint(1,n,size)
    np.random.shuffle(mask)

    train_size = int(size*0.7)
    val_size = int(size*0.2)
    test_size = int(size*0.1)

    train_mask = mask[:train_size]
    val_mask = mask[train_size:train_size + val_size]
    test_mask = mask[train_size+val_size:]

    return train_mask,val_mask,test_mask


def load_custom_data():
    train_mask,val_mask,test_mask = gen_mask(size=26342,n=26342)
    train_adj = generate_sparse_adj(train_mask)
    val_adj = generate_sparse_adj(val_mask)
    test_adj = generate_sparse_adj(test_mask)

    train_feat = generate_features(train_mask)
    val_feat = generate_features(val_mask)
    test_feat = generate_features(test_mask)

    train_labels = load_labels(train_mask)
    val_labels = load_labels(val_mask)
    test_labels = load_labels(test_mask)

    return train_adj,train_feat,train_labels,val_adj,val_feat,val_labels,test_adj,test_feat,test_labels
