""" A neural chatbot using sequence to sequence model with
attentional decoder. 

This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu

This file contains the hyperparameters for the model.

See README.md for instruction on how to run the starter code.
"""

# parameters for processing the dataset
DATA_PATH = ''
CONVO_FILE = 'train.en'
LINE_FILE = 'train.vi'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = 'util_files'
CPT_PATH = 'model'
DEC_VOCAB = 18778
ENC_VOCAB = 41303
THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 25000

BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63)]


CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "), 
				("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),
				("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),
				("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LR = 0.5
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512





""" A neural chatbot using sequence to sequence model with
attentional decoder. 

This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu

This file contains the code to do the pre-processing for the
Cornell Movie-Dialogs Corpus.

See readme.md for instruction on how to run the starter code.
"""
import os
import random
import re

import numpy as np


def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words

def build_vocab(filename, normalize_digits=True):
    in_path = os.path.join(PROCESSED_PATH, filename)
    out_path = os.path.join(PROCESSED_PATH, 'vocab.{}'.format(filename[-3:]))

    vocab = {}
    with open(in_path, 'r') as f:
        for line in f.readlines():
            for token in basic_tokenizer(line):
                if not token in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'w') as f:
        f.write('<pad>' + '\n')
        f.write('<unk>' + '\n')
        f.write('<s>' + '\n')
        f.write('<\s>' + '\n') 
        index = 4
        for word in sorted_vocab:
            if vocab[word] < THRESHOLD:
                break
            f.write(word + '\n')
            index += 1
        with open('py', 'a') as cf:
            if filename[-3:] == 'enc':
                cf.write('ENC_VOCAB = ' + str(index) + '\n')
            else:
                cf.write('DEC_VOCAB = ' + str(index) + '\n')

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}

def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)]

def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    _, vocab = load_vocab(os.path.join(PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(PROCESSED_PATH, in_path), 'r')
    out_file = open(os.path.join(PROCESSED_PATH, out_path), 'w')
    
    lines = in_file.read().splitlines()
    for line in lines:
        if mode == 'dec': # we only care about '<s>' and </s> in encoder
            ids = [vocab['<s>']]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        # ids.extend([vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)])
        if mode == 'dec':
            ids.append(vocab['<\s>'])
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def process_data():
    print('Preparing data to be model-ready ...')
    build_vocab('train.enc')
    build_vocab('train.dec')
    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')

def load_data(enc_filename, dec_filename, max_training_size=None):
    encode_file = open(os.path.join(PROCESSED_PATH, enc_filename), 'r')
    decode_file = open(os.path.join(PROCESSED_PATH, dec_filename), 'r')
    encode, decode = encode_file.readline(), decode_file.readline()
    data_buckets = [[] for _ in BUCKETS]
    i = 0
    while encode and decode:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i)
        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break
        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    return data_buckets

def _pad_input(input_, size):
    return input_ + [PAD_ID] * (size - len(input_))

def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                    for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs


def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size, decoder_size = BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks


import time

import numpy as np
import tensorflow as tf


class ChatBotModel:
    def __init__(self, forward_only, batch_size):
        """forward_only: if set, we do not construct the backward pass in the model.
        """
        self.fw_only = forward_only
        self.batch_size = batch_size

    def _create_placeholders(self):
        # Feeds for inputs. It's a list of placeholders
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                               for i in range(BUCKETS[-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                               for i in range(BUCKETS[-1][1] + 1)]
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in range(BUCKETS[-1][1] + 1)]

        # Our targets are decoder inputs shifted by one (to ignore <GO> symbol)
        self.targets = self.decoder_inputs[1:]

    def _inference(self):

        # If we use sampled softmax, we need an output projection.
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if NUM_SAMPLES > 0 and NUM_SAMPLES < DEC_VOCAB:
            w = tf.get_variable('proj_w', [HIDDEN_SIZE, DEC_VOCAB])
            b = tf.get_variable('proj_b', [DEC_VOCAB])
            self.output_projection = (w, b)

        def sampled_loss(logits, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(weights=tf.transpose(w), 
                                              biases=b, 
                                              inputs=logits, 
                                              labels=labels, 
                                              num_sampled=NUM_SAMPLES, 
                                              num_classes=DEC_VOCAB)
        self.softmax_loss_function = sampled_loss

        single_cell = tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
        self.cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(NUM_LAYERS)])

    def _create_loss(self):
        print('Creating loss please wait')
        start = time.time()
        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
            setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, self.cell,
                    num_encoder_symbols=ENC_VOCAB,
                    num_decoder_symbols=DEC_VOCAB,
                    embedding_size=HIDDEN_SIZE,
                    output_projection=self.output_projection,
                    feed_previous=do_decode)

        if self.fw_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoder_inputs, 
                                        self.decoder_inputs, 
                                        self.targets,
                                        self.decoder_masks, 
                                        BUCKETS, 
                                        lambda x, y: _seq2seq_f(x, y, True),
                                        softmax_loss_function=self.softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if self.output_projection:
                for bucket in range(len(BUCKETS)):
                    self.outputs[bucket] = [tf.matmul(output, 
                                            self.output_projection[0]) + self.output_projection[1]
                                            for output in self.outputs[bucket]]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoder_inputs, 
                                        self.decoder_inputs, 
                                        self.targets,
                                        self.decoder_masks,
                                        BUCKETS,
                                        lambda x, y: _seq2seq_f(x, y, False),
                                        softmax_loss_function=self.softmax_loss_function)

    def _creat_optimizer(self):
        print('Creating optimizer please wait')
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            if not self.fw_only:
                self.optimizer = tf.train.GradientDescentOptimizer(LR)
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []
                start = time.time()
                for bucket in range(len(BUCKETS)):
                    
                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket], 
                                                                 trainables),
                                                                 MAX_GRAD_NORM)
                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables), 
                                                            global_step=self.global_step))
                    print('Creating opt for bucket {} took {} seconds'.format(bucket, time.time() - start))
                    start = time.time()


    def _create_summary(self):
        pass

    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._creat_optimizer()
        self._create_summary()
		
		
		
		
		
		
		
		

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random
import sys
import time

import numpy as np
import tensorflow as tf

def _get_random_bucket(train_buckets_scale):
    """ Get a random bucket from which to choose a training sample """
    try:
        rand = random.random()
        return min([i for i in range(len(train_buckets_scale))
                    if train_buckets_scale[i] > rand])
    except:
        pass

def _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    """ Assert that the encoder inputs, decoder inputs, and decoder masks are
    of the expected lengths """
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                        " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_masks), decoder_size))

def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
    """ Run one step in training.
    @forward_only: boolean value to decide whether a backward path should be created
    forward_only is set to True when you just want to evaluate on the test set,
    or when you want to the bot to be in chat mode. """
    encoder_size, decoder_size = BUCKETS[bucket_id]
    _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

    # input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for step in range(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]

    last_target = model.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

    # output feed: depends on whether we do a backward step or not.
    if not forward_only:
        output_feed = [model.train_ops[bucket_id],  # update op that does SGD.
                       model.gradient_norms[bucket_id],  # gradient norm.
                       model.losses[bucket_id]]  # loss for this batch.
    else:
        output_feed = [model.losses[bucket_id]]  # loss for this batch.
        for step in range(decoder_size):  # output logits.
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
        return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

def _get_buckets():
    """ Load the dataset into buckets based on their lengths.
    train_buckets_scale is the inverval that'll help us 
    choose a random bucket later on.
    """
    test_buckets = load_data('test_ids.enc', 'test_ids.dec')
    data_buckets = load_data('train_ids.enc', 'train_ids.dec')
    train_bucket_sizes = [len(data_buckets[b]) for b in range(len(BUCKETS))]
    train_total_size = sum(train_bucket_sizes)
    # list of increasing numbers from 0 to 1 that we'll use to select a bucket.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    return test_buckets, data_buckets, train_buckets_scale

def _get_skip_step(iteration):
    """ How many steps should the model train before it saves all the weights. """
    if iteration < 100:
        return 30
    return 100

def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(CPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot")
        saver.restore(sess, './model/chatbot-33800')
    else:
        print("Initializing fresh parameters for the Chatbot")

def _eval_test_set(sess, model, test_buckets):
    """ Evaluate on the test set. """
    for bucket_id in range(len(BUCKETS)):
        if len(test_buckets[bucket_id]) == 0:
            print("  Test: empty bucket %d" % (bucket_id))
            continue
        start = time.time()
        encoder_inputs, decoder_inputs, decoder_masks = get_batch(test_buckets[bucket_id], 
                                                                        bucket_id,
                                                                        batch_size=BATCH_SIZE)
        _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, 
                                   decoder_masks, bucket_id, True)
        print('Test bucket {}: loss {}, time {}'.format(bucket_id, step_loss, time.time() - start))

def train():
    """ Train the bot """
    test_buckets, data_buckets, train_buckets_scale = _get_buckets()
    # in train mode, we need to create the backward path, so forwrad_only is False
    model = ChatBotModel(False, BATCH_SIZE)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Running session')
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)

        iteration = model.global_step.eval()
        total_loss = 0
        while True:
            skip_step = _get_skip_step(iteration)
            bucket_id = _get_random_bucket(train_buckets_scale)
            encoder_inputs, decoder_inputs, decoder_masks = get_batch(data_buckets[bucket_id], 
                                                                           bucket_id,
                                                                           batch_size=BATCH_SIZE)
            start = time.time()
            _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
            total_loss += step_loss
            iteration += 1

            if iteration % skip_step == 0:
                print('Iter {}: loss {}, time {}'.format(iteration, total_loss/skip_step, time.time() - start))
                start = time.time()
                total_loss = 0
                saver.save(sess, os.path.join(CPT_PATH, 'chatbot'), global_step=model.global_step)
                if iteration % (10 * skip_step) == 0:
                    # Run evals on development set and print their loss
                    _eval_test_set(sess, model, test_buckets)
                    start = time.time()
                sys.stdout.flush()

def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()

def _find_right_bucket(length):
    """ Find the proper bucket for an encoder input based on its length """
    return min([b for b in range(len(BUCKETS))
                if BUCKETS[b][0] >= length])

def _construct_response(output_logits, inv_dec_vocab):
    """ Construct a response to the user's encoder input.
    @output_logits: the outputs from sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB
    
    This is a greedy decoder - outputs are just argmaxes of output_logits.
    """
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if EOS_ID in outputs:
        outputs = outputs[:outputs.index(EOS_ID)]
    # Print out sentence corresponding to outputs.
    return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])

def chat():
    """ in test mode, we don't to create the backward path
    """
    _, enc_vocab = load_vocab(os.path.join(PROCESSED_PATH, 'vocab.enc'))
    inv_dec_vocab, _ = load_vocab(os.path.join(PROCESSED_PATH, 'vocab.dec'))

    model = ChatBotModel(True, batch_size=1)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        output_file = open(os.path.join(PROCESSED_PATH, OUTPUT_FILE), 'a+')
        # Decode from standard input.
        max_length = BUCKETS[-1][0]
        print('Please type your sentence to translate')
        while True:
            line = _get_user_input()
            if len(line) > 0 and line[-1] == '\n':
                line = line[:-1]
            if line == '':
                break
            output_file.write('HUMAN ++++ ' + line + '\n')
            # Get token-ids for the input sentence.
            token_ids = sentence2id(enc_vocab, str(line))
            if (len(token_ids) > max_length):
                print('Please type the sentences with length less than', max_length)
                line = _get_user_input()
                continue
            # Which bucket does it belong to?
            bucket_id = _find_right_bucket(len(token_ids))
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, decoder_masks = get_batch([(token_ids, [])], 
                                                                            bucket_id,
                                                                            batch_size=1)
            # Get output logits for the sentence.
            _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
            response = _construct_response(output_logits, inv_dec_vocab)
            print(response)
            output_file.write('BOT ++++ ' + response + '\n')
        output_file.write('=============================================\n')
        output_file.close()

def load_test_data(filename, max_training_size=None):
    encode_file = os.path.join(PROCESSED_PATH, filename)
    data_buckets = [[] for _ in BUCKETS]
    data = []
    with open(encode_file, 'r',encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line)
    return data

def _load_test_data():
    eng_test_ids = load_test_data('tst2013.en')
    vit_test_ids = load_test_data('tst2013.vi')
    return eng_test_ids,vit_test_ids

def _get_ids_for_test(output_logits, inv_dec_vocab):
    """ Construct a response to the user's encoder input.
    @output_logits: the outputs from sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB
    
    This is a greedy decoder - outputs are just argmaxes of output_logits.
    """
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if EOS_ID in outputs:
        outputs = outputs[:outputs.index(EOS_ID)]
    # Print out sentence corresponding to outputs.
    return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])

def test():
      import nltk
      from nltk.translate.bleu_score import sentence_bleu
      import numpy as np

    
      print("Loading Model...")
      _, enc_vocab = load_vocab(os.path.join(PROCESSED_PATH, 'vocab.enc'))
      inv_dec_vocab, _ = load_vocab(os.path.join(PROCESSED_PATH, 'vocab.dec'))
      eng_test,vit_test = _load_test_data()
      model = ChatBotModel(True, batch_size=1)
      model.build_graph()
      saver = tf.train.Saver()
      
      with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          _check_restore_parameters(sess, saver)
          bleu = []
          print("Now Calculating Bleu Score")
          for i in range(len(eng_test)):
             # print(str(eng_test[i]))
              token_ids = sentence2id(enc_vocab, str(eng_test[i]))
              target_ids = vit_test[i]
              # Which bucket does it belong to?
              try:
                  bucket_id = _find_right_bucket(len(token_ids))
              except Exception:
                  continue
                  
              # Get a 1-element batch to feed the sentence to the model.
              encoder_inputs, decoder_inputs, decoder_masks = get_batch([(token_ids, [])], 
                                                                              bucket_id,
                                                                              batch_size=1)
              # Get output logits for the sentence.
              _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                             decoder_masks, bucket_id, True)
              response = _get_ids_for_test(output_logits, inv_dec_vocab)
              bleu.append(sentence_bleu([target_ids.split()],response.split(),smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1))
          
          print(np.mean(bleu))



import sys
def main(argv):

    if argv[0] == "test":
        #value = argv[0]
        test()

    if argv[0] == "train":
        #value = argv[0]
        train()
		
    if argv[0] == "translate":
        #value = argv[0]
        chat()

if __name__ == "__main__":
    main(sys.argv[1:])
