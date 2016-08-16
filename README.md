# tf-rnn-pub
a study of RNNs in tensorflow (v0.9/0.10)

* Backpropogation on variable length (bi)RNNs.
* mean pooling on variable length (bi)RNNs.

Based on experiments, it seems that tensorflow handles backprop correctly when the sequences in a batch have different length
as long as the `sequence_length` variable is set properly. When one uses mean pooling from the RNN outputs, tensorflow also correctly calculates the mean according to the `sequence_length` for each sequence in a batch (so there is no need to manually calculate the sum of output then divide by actual sequence lengths). This is especially helpful when one uses the bidirectional RNNs: if the user needs to manually keep track of the sequence paddings in a batch, correctly concatenating the forward output to the backward output requires further bookkeeping of the padding steps (end-padding becomes head-padding in another direction).

In addition, `dynamic_rnn` and `rnn` seem to be the same model except for the input format.


Experiment One
==============

`python rnn.py --rnn_type [fw|bi]`
If the `sequence_length` parameter is properly set in `rnn` or `dynamic_rnn` or their `bidirectional_` versions,
the forward and backward propagation (from average-pooling state or last hidden state) will be calculated correctly
and requires no manual masking on the output.

Experiment Two
==============

`python benchmark.py --rnn_type [fw|bi] --iterations 200`

if running GPU mode, make sure `LD_LIBRARY_PATH` include both `/usr/local/cuda/lib64/` and `/usr/local/cuda/extras/CUPTI/lib64/`
so that GPU tracking will work..

`benchmark.py` runs the RNN cells through deterministically random input sequences (with random lengths for each sequence).

My timing on GPU (fw rnn):
```
dynamic	314704
dynamic.Add	192
dynamic.Assign	156817
dynamic.Const	23
dynamic.Mul	415
dynamic.TruncatedNormal	157089
dynamic.Variable	168
naive	314706
naive.Add	230
naive.Assign	156769
naive.Const	26
naive.Mul	447
naive.TruncatedNormal	157122
naive.Variable	112
```
and on CPU (fw rnn):
```
dynamic	48077
dynamic.Add	4288
dynamic.Assign	3099
dynamic.Const	10
dynamic.Mul	4777
dynamic.TruncatedNormal	35253
dynamic.Variable	650
naive	48768
naive.Add	5693
naive.Assign	3117
naive.Const	12
naive.Mul	5120
naive.TruncatedNormal	33952
naive.Variable	874
```

It seems that `dynamic_rnn` is slightly faster than the `rnn` counterparts.
