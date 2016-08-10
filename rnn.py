import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


#tested with tensorflow 0.9/0.10 CPU/GPU
if tf.__version__.startswith("0.9"):
    print "Warning: rnn.bidirectional_dynamic_rnn not defined for tf 0.9. bidirectional_dynamic_rnn tests will skip."

flags = tf.app.flags
flags.DEFINE_string("rnn_type", "fw", "fw or bi")
FLAGS = flags.FLAGS

tf.set_random_seed(1)
embed = tf.Variable(tf.constant([[8,8],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]],dtype=tf.float32),trainable=False)
seq1  = tf.constant([[1,2,3,0,0,0]]) # batch_size = 1
seq2  = tf.constant([[1,2,3]])       # batch_size = 1
seq3 = tf.constant([[1,2,3,4,5,6],
                     [1,2,3,0,0,0]]) # batch_size = 2
seq4 = tf.constant([[1,2,3],
                     [1,2,3]])       # batch_size = 2
seq5 = tf.constant([[1,2,3,4,5,6],
                     [1,2,3,4,5,6]]) # batch_size = 2

unpacked_seqs_embeds = [None,]
seqs_embeds          = [None,]
for seq in [seq1,seq2,seq3,seq4,seq5]:
    seqs_embed = tf.nn.embedding_lookup(embed,seq)
    seqs_embeds.append(seqs_embed)
    unpacked_seqs_embeds.append(tf.unpack(tf.transpose(seqs_embed,perm=[1, 0, 2])))


with tf.variable_scope("naive",initializer=tf.truncated_normal_initializer(seed=1)) as scope:
    if FLAGS.rnn_type == "fw":
        n_output1,_   = rnn.rnn(rnn_cell.BasicRNNCell(2),unpacked_seqs_embeds[1],dtype=tf.float32,sequence_length=None )
    elif FLAGS.rnn_type == "bi":
        n_output1,_,_   = rnn.bidirectional_rnn(rnn_cell.BasicRNNCell(1),rnn_cell.BasicRNNCell(1),unpacked_seqs_embeds[1],dtype=tf.float32,sequence_length=None )
with tf.variable_scope("naive2",initializer=tf.truncated_normal_initializer(seed=1)) as scope:
    if FLAGS.rnn_type == "fw":
        n_output2,_   = rnn.rnn(rnn_cell.BasicRNNCell(2),unpacked_seqs_embeds[2],dtype=tf.float32,sequence_length=None )
    elif FLAGS.rnn_type == "bi":
        n_output2,_,_   = rnn.bidirectional_rnn(rnn_cell.BasicRNNCell(1),rnn_cell.BasicRNNCell(1),unpacked_seqs_embeds[2],dtype=tf.float32,sequence_length=None )
with tf.variable_scope("batch1",initializer=tf.truncated_normal_initializer(seed=1)) as scope:
    if FLAGS.rnn_type == "fw":
        n_output3,_   = rnn.rnn(rnn_cell.BasicRNNCell(2),unpacked_seqs_embeds[3],dtype=tf.float32,sequence_length=None )
    elif FLAGS.rnn_type == "bi":
        n_output3,_,_   = rnn.bidirectional_rnn(rnn_cell.BasicRNNCell(1),rnn_cell.BasicRNNCell(1),unpacked_seqs_embeds[3],dtype=tf.float32,sequence_length=None )
with tf.variable_scope("batch2",initializer=tf.truncated_normal_initializer(seed=1)) as scope:
    if FLAGS.rnn_type == "fw":
        n_output4,_   = rnn.rnn(rnn_cell.BasicRNNCell(2),unpacked_seqs_embeds[4],dtype=tf.float32,sequence_length=None )
    elif FLAGS.rnn_type == "bi":
        n_output4,_,_   = rnn.bidirectional_rnn(rnn_cell.BasicRNNCell(1),rnn_cell.BasicRNNCell(1),unpacked_seqs_embeds[4],dtype=tf.float32,sequence_length=None )
with tf.variable_scope("batch3",initializer=tf.truncated_normal_initializer(seed=1)) as scope:
    if FLAGS.rnn_type == "fw":
        n_output5,_   = rnn.rnn(rnn_cell.BasicRNNCell(2),unpacked_seqs_embeds[5],dtype=tf.float32,sequence_length=None )
    elif FLAGS.rnn_type == "bi":
        n_output5,_,_   = rnn.bidirectional_rnn(rnn_cell.BasicRNNCell(1),rnn_cell.BasicRNNCell(1),unpacked_seqs_embeds[5],dtype=tf.float32,sequence_length=None )
with tf.variable_scope("batch4",initializer=tf.truncated_normal_initializer(seed=1)) as scope:
    if FLAGS.rnn_type == "fw":
        n_output6,_   = rnn.rnn(rnn_cell.BasicRNNCell(2),unpacked_seqs_embeds[3],dtype=tf.float32,sequence_length=tf.constant([6,3]) )
    elif FLAGS.rnn_type == "bi":
        n_output6,_,_   = rnn.bidirectional_rnn(rnn_cell.BasicRNNCell(1),rnn_cell.BasicRNNCell(1),unpacked_seqs_embeds[3],dtype=tf.float32,sequence_length=tf.constant([6,3]) )
with tf.variable_scope("batch5",initializer=tf.truncated_normal_initializer(seed=1)) as scope:
    if FLAGS.rnn_type == "fw":
        n_output7,_   = rnn.rnn(rnn_cell.BasicRNNCell(2),unpacked_seqs_embeds[5],dtype=tf.float32,sequence_length=tf.constant([6,3]) )
    elif FLAGS.rnn_type == "bi":
        n_output7,_,_   = rnn.bidirectional_rnn(rnn_cell.BasicRNNCell(1),rnn_cell.BasicRNNCell(1),unpacked_seqs_embeds[5],dtype=tf.float32,sequence_length=tf.constant([6,3]) )
with tf.variable_scope("dynamic_batch1",initializer=tf.truncated_normal_initializer(seed=1)) as scope:
    if FLAGS.rnn_type == "fw":
        n_output8,_   = rnn.dynamic_rnn(rnn_cell.BasicRNNCell(2),seqs_embeds[3],dtype=tf.float32,time_major=False )
    elif FLAGS.rnn_type == "bi" and tf.__version__.startswith("0.10"):
        n_output8,_   = rnn.bidirectional_dynamic_rnn(rnn_cell.BasicRNNCell(1),rnn_cell.BasicRNNCell(1),seqs_embeds[3],dtype=tf.float32,time_major=False, sequence_length=tf.constant([6,6],dtype=tf.int64) )
with tf.variable_scope("dynamic_batch2",initializer=tf.truncated_normal_initializer(seed=1)) as scope:
    if FLAGS.rnn_type == "fw":
        n_output9,_   = rnn.dynamic_rnn(rnn_cell.BasicRNNCell(2),seqs_embeds[5],dtype=tf.float32,time_major=False )
    elif FLAGS.rnn_type == "bi" and tf.__version__.startswith("0.10"):
        n_output9,_   = rnn.bidirectional_dynamic_rnn(rnn_cell.BasicRNNCell(1),rnn_cell.BasicRNNCell(1),seqs_embeds[5],dtype=tf.float32,time_major=False, sequence_length=tf.constant([6,6],dtype=tf.int64) )
with tf.variable_scope("dynamic_batch3",initializer=tf.truncated_normal_initializer(seed=1)) as scope:
    if FLAGS.rnn_type == "fw":
        n_output10,_   = rnn.dynamic_rnn(rnn_cell.BasicRNNCell(2),seqs_embeds[3],dtype=tf.float32,sequence_length=tf.constant([6,3]),time_major=False )
    elif FLAGS.rnn_type == "bi" and tf.__version__.startswith("0.10"):
        n_output10,_   = rnn.bidirectional_dynamic_rnn(rnn_cell.BasicRNNCell(1),rnn_cell.BasicRNNCell(1),seqs_embeds[3],dtype=tf.float32,sequence_length=tf.constant([6,3],dtype=tf.int64),time_major=False )
with tf.variable_scope("dynamic_batch4",initializer=tf.truncated_normal_initializer(seed=1)) as scope:
    if FLAGS.rnn_type == "fw":
        n_output11,_   = rnn.dynamic_rnn(rnn_cell.BasicRNNCell(2),seqs_embeds[5],dtype=tf.float32,sequence_length=tf.constant([6,3]),time_major=False )
    elif FLAGS.rnn_type == "bi" and tf.__version__.startswith("0.10"):
        n_output11,_   = rnn.bidirectional_dynamic_rnn(rnn_cell.BasicRNNCell(1),rnn_cell.BasicRNNCell(1),seqs_embeds[5],dtype=tf.float32,sequence_length=tf.constant([6,3],dtype=tf.int64),time_major=False )

avgLosses = [] # average pooling loss
lstLosses = [] # last hidden state loss
if FLAGS.rnn_type == "fw" or (FLAGS.rnn_type == "bi" and tf.__version__.startswith("0.10")):
    outputs = [n_output1,n_output2,n_output3,n_output4,n_output5,n_output6,n_output7,n_output8,n_output9,n_output10,n_output11]
else:
    outputs = [n_output1,n_output2,n_output3,n_output4,n_output5,n_output6,n_output7]

for i,n_output in enumerate(outputs):
    if isinstance(n_output,list):
        neuron = tf.transpose(tf.pack(n_output), perm=[1, 0, 2]) # rnn.rnn, rnn.bidirectional_rnn
    elif isinstance(n_output,tuple):
        neuron = tf.concat(2,n_output)  # rnn.bidirectional_dynamic_rnn
    else:
        neuron = n_output # rnn.dynamic_rnn
    seq_len = neuron.get_shape()[1].value - 1
    avgLosses.append(tf.reduce_mean( tf.reduce_mean(neuron,[1,0]) - tf.constant([3,3],dtype=tf.float32)))
    lstLosses.append(tf.reduce_mean( tf.reduce_mean(neuron[:,seq_len,:],0) - tf.constant([3,3],dtype=tf.float32)))
opt = tf.train.GradientDescentOptimizer(0.1)
params = tf.trainable_variables()
avgOptims = []
lstOptims = []
for loss in avgLosses:
    avgOptims.append(opt.apply_gradients(zip(tf.gradients(loss, params), params)))
for loss in lstLosses:
    lstOptims.append(opt.apply_gradients(zip(tf.gradients(loss, params), params)))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

compareLoss = [] # losses to compare after 9 iterations
for optim,loss in zip(avgOptims,avgLosses):
    printstuff = []
    for i in range(10):
        output = sess.run([optim,loss])
        if i==0 or i == 9:
            printstuff.append(output[1])
    print "itr0:%.6f itr9:%.6f"%(printstuff[0],printstuff[1])
    print "="*30
    compareLoss.append(printstuff[1])
assert compareLoss[0] != compareLoss[1], "123000 and 123 produce same avgLoss"
assert compareLoss[1] == compareLoss[3], "123 and 123,123 produce different avgLoss"
assert compareLoss[5] == compareLoss[6], "123456,123(000) and 123456,123(456)  produce different avgLoss"
if FLAGS.rnn_type == "fw" or (FLAGS.rnn_type == "bi" and tf.__version__.startswith("0.10")):
    assert compareLoss[2] == compareLoss[7], "123456,123000 and 123456,123000  produce different avgLoss under rnn.rnn and rnn.dynamic_rnn"
    assert compareLoss[9] == compareLoss[10], "123456,123(000) and 123456,123(456) produce different avgLoss under rnn.dynamic_rnn"


compareLoss = [] # losses to compare after 9 iterations
for optim,loss in zip(lstOptims,lstLosses):
    printstuff = []
    for i in range(10):
        output = sess.run([optim,loss])
        if i==0 or i == 9:
            printstuff.append(output[1])
    print "itr0:%.6f itr9:%.6f"%(printstuff[0],printstuff[1])
    print "="*30
    compareLoss.append(printstuff[1])
assert compareLoss[0] != compareLoss[1], "123000 and 123 produce same lstLoss"
assert compareLoss[1] == compareLoss[3], "123 and 123,123 produce different lstLoss"
assert compareLoss[5] == compareLoss[6], "123456,123(000) and 123456,123(456)  produce different lstLoss"
if FLAGS.rnn_type == "fw" or (FLAGS.rnn_type == "bi" and tf.__version__.startswith("0.10")):
    assert compareLoss[2] == compareLoss[7], "123456,123000 and 123456,123000  produce different lstLoss under rnn.rnn and rnn.dynamic_rnn"
    assert compareLoss[9] == compareLoss[10], "123456,123(000) and 123456,123(456) produce different lstLoss under rnn.dynamic_rnn"
print "Passes all assertions"
print "Therefore, once sequence_length is specified, there is no need to mask or select lastRelevant based on input data length. The computation graph is updated by sequence_length's automatically."

sess.close()
