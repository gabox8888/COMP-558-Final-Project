from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from custom_utils import *
from gcn.models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
train_adj,train_feat,train_labels,val_adj,val_feat,val_labels,test_adj,test_feat,test_labels = load_custom_data()

# Some preprocessing
features_train = preprocess_features(train_feat)
features_val = preprocess_features(val_feat)
features_test = preprocess_features(test_feat)

if FLAGS.model == 'gcn':
    support_train = [preprocess_adj(train_adj)]
    support_val = [preprocess_adj(val_adj)]
    support_test = [preprocess_adj(test_adj)]

    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support_train = [preprocess_adj(train_adj)]
    support_val = [preprocess_adj(val_adj)]
    support_test = [preprocess_adj(test_adj)]

    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support_train = [preprocess_adj(train_adj)]
    support_val = [preprocess_adj(val_adj)]
    support_test = [preprocess_adj(test_adj)]

    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features_train[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None,train_labels.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features_train[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

train_mask = np.asarray([1 if i%99 ==0 and i==0 else 0 for i in range(train_adj.shape[0])],dtype=np.bool)
val_mask =  np.asarray([1 if i%99 ==0 and i==0 else 0 for i in range(val_adj.shape[0])],dtype=np.bool)
test_mask =  np.asarray([1 if i%99 ==0 and i==0 else 0 for i in range(test_adj.shape[0])],dtype=np.bool)

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features_train, support_train, train_labels, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features_val, support_val, val_labels, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features_test, support_test, test_labels, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
