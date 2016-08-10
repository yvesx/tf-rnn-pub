import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.client import timeline
import numpy as np
from collections import defaultdict
import json

#tested with tensorflow 0.9/0.10 CPU/GPU
if tf.__version__.startswith("0.9"):
    print "Warning: rnn.bidirectional_dynamic_rnn not defined for tf 0.9. bidirectional_dynamic_rnn tests will skip."

flags = tf.app.flags
flags.DEFINE_string("rnn_type", "fw", "fw or bi")
flags.DEFINE_integer("iterations", 10, "number of iterations to benchmark")
FLAGS = flags.FLAGS

tf.set_random_seed(1)
np.random.seed(1)
vocab_size = 1000
batch_size = 128
embed_dim  = 512
rnn_dim    = 1024
max_len    = 100
label  = tf.constant(np.random.rand(rnn_dim),dtype=tf.float32)
embed  = tf.Variable(tf.constant(np.random.rand(vocab_size,embed_dim),dtype=tf.float32),trainable=False)
seq_len= tf.placeholder(tf.int32,[batch_size])
seq    = tf.placeholder(tf.int32,[batch_size,max_len])
seqs_embed = tf.nn.embedding_lookup(embed,seq)
unpacked_seqs_embed = tf.unpack(tf.transpose(seqs_embed,perm=[1, 0, 2]))



with tf.variable_scope("naive",initializer=tf.truncated_normal_initializer(seed=1)) as scope:
    if FLAGS.rnn_type == "fw":
        n_output1,_   = rnn.rnn(rnn_cell.BasicRNNCell(rnn_dim),unpacked_seqs_embed,dtype=tf.float32,sequence_length=seq_len )
    elif FLAGS.rnn_type == "bi":
        n_output1,_,_   = rnn.bidirectional_rnn(rnn_cell.BasicRNNCell(rnn_dim/2),rnn_cell.BasicRNNCell(rnn_dim/2),unpacked_seqs_embeds,dtype=tf.float32,sequence_length=seq_len )


with tf.variable_scope("dynamic",initializer=tf.truncated_normal_initializer(seed=1)) as scope:
    if FLAGS.rnn_type == "fw":
        n_output2,_   = rnn.dynamic_rnn(rnn_cell.BasicRNNCell(rnn_dim),seqs_embed,dtype=tf.float32,time_major=False,sequence_length=seq_len )
    elif FLAGS.rnn_type == "bi" and tf.__version__.startswith("0.10"):
        n_output2,_   = rnn.bidirectional_dynamic_rnn(rnn_cell.BasicRNNCell(rnn_dim/2),rnn_cell.BasicRNNCell(rnn_dim/2),seqs_embed,dtype=tf.float32,time_major=False,sequence_length=seq_len )

avgLosses = [] # average pooling loss
if FLAGS.rnn_type == "fw" or (FLAGS.rnn_type == "bi" and tf.__version__.startswith("0.10")):
    outputs = [n_output1,n_output2]
else:
    outputs = [n_output1]

for i,n_output in enumerate(outputs):
    if isinstance(n_output,list):
        neuron = tf.transpose(tf.pack(n_output), perm=[1, 0, 2]) # rnn.rnn, rnn.bidirectional_rnn
    elif isinstance(n_output,tuple):
        neuron = tf.concat(2,n_output)  # rnn.bidirectional_dynamic_rnn
    else:
        neuron = n_output # rnn.dynamic_rnn
    avgLosses.append(tf.reduce_mean( tf.reduce_mean(neuron,[1,0]) - label))
opt = tf.train.GradientDescentOptimizer(0.1)
params = tf.trainable_variables()
avgOptims = []
for loss in avgLosses:
    avgOptims.append(opt.apply_gradients(zip(tf.gradients(loss, params), params)))

sess = tf.Session()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
sess.run(tf.initialize_all_variables(),options=run_options, run_metadata=run_metadata)

for i in range(FLAGS.iterations):
    np_seq_len = np.random.randint(low=max_len/3, high=max_len, size=batch_size)
    np_seq     = np.random.randint(vocab_size,size=(batch_size,max_len))
    for optim,loss in zip(avgOptims,avgLosses):
        printstuff = []
        feed_dict = {
                     seq_len : np_seq_len,
                     seq     : np_seq
                     }
        output = sess.run([optim,loss],feed_dict=feed_dict)
        printstuff.append(output[1])

tl = timeline.Timeline(run_metadata.step_stats)
json_str = tl.generate_chrome_trace_format()
json_obj = json.loads(json_str)
sess.close()

times = defaultdict(int)
for event in json_obj['traceEvents']:
    if event.get("name") and event.get("dur") and event.get("args") and event.get("args").get("name"):
        prefix = event.get("args").get("name").split("/")[0]
        if prefix in ["naive","dynamic"]:
            key = "%s.%s"%(prefix,event.get("name"))
            val = event.get("dur")
            times[key] += val
            times[prefix] += val

for k,v in sorted(times.items()):
    print "%s\t%i"%(k,v)

print "Timing is average (not aggregated) across iterations."
