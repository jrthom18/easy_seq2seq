# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model

try:
    from ConfigParser import SafeConfigParser
except:
    from configparser import SafeConfigParser # In Python 3, ConfigParser has been renamed to configparser for PEP 8 compliance.
    
gConfig = {}

def get_config(config_file='seq2seq.ini'):
    parser = SafeConfigParser()
    parser.read(config_file)
    # get the ints, floats and strings
    _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
    _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
    return dict(_conf_ints + _conf_floats + _conf_strings)

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):

  """Create model and initialize or load parameters"""
  model = seq2seq_model.Seq2SeqModel( gConfig['enc_vocab_size'], gConfig['dec_vocab_size'], _buckets, gConfig['layer_size'], gConfig['num_layers'], gConfig['max_gradient_norm'], gConfig['batch_size'], gConfig['learning_rate'], gConfig['learning_rate_decay_factor'], forward_only=forward_only)

  if 'pretrained_model' in gConfig:
      model.saver.restore(session,gConfig['pretrained_model'])
      return model

  ckpt = tf.train.get_checkpoint_state(gConfig['working_directory'])
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train():
  # prepare dataset
  print("Preparing data in %s" % gConfig['working_directory'])
  enc_train, dec_train, enc_dev, dec_dev, _, _ = data_utils.prepare_custom_data(gConfig['working_directory'],gConfig['train_enc'],gConfig['train_dec'],gConfig['test_enc'],gConfig['test_dec'],gConfig['enc_vocab_size'],gConfig['dec_vocab_size'])

  # setup config to use BFC allocator
  config = tf.ConfigProto()  
  config.gpu_options.allocator_type = 'BFC'

  with tf.Session(config=config) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (gConfig['num_layers'], gConfig['layer_size']))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % gConfig['max_train_data_size'])
    dev_set = read_data(enc_dev, dec_dev)
    train_set = read_data(enc_train, dec_train, gConfig['max_train_data_size'])
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / gConfig['steps_per_checkpoint']
      loss += step_loss / gConfig['steps_per_checkpoint']
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % gConfig['steps_per_checkpoint'] == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(gConfig['working_directory'], "seq2seq.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()


def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    enc_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.enc" % gConfig['enc_vocab_size'])
    dec_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.dec" % gConfig['dec_vocab_size'])

    enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
    _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()

def runTestScript():
  print('Automated testing script for short Q&A...')
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    enc_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.enc" % gConfig['enc_vocab_size'])
    dec_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.dec" % gConfig['dec_vocab_size'])

    enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
    _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)

    originalQuestions = ['Can I copy from Finder the current Path','for what should I use fn key by default on mac book pro','How can Find My iPhone be working with no data plan','Is it possible to queue songs on the iPhone','Force simple UNIX path names for optimal us of command line apps','How can you change the mouse cursor in OSX','Can you tell if updates came from OS X caching server','Why do certain apps fill the background of Mission Control in Mavericks','I\'m looking for a quick entry TODO application that syncs across an iPhone and my laptop','Play Audio from a Mac over an iPhone Call','Mapping Home and End of Apple keyboard in VMWare fusion','How do I use GCC on El Capitan','How to import DVD\'s to iTunes','HP P4515 printer in 10.7.4 - what driver','Is iphone purchased from Canadian online Apple Store factory unlocked','Can I share a 3G connection from an iPhone/iPad by creating a Wi-fi hotspot','Is storing an iMac sideways safe','How can I change the default text color in Mail','How to clean a sticky Magic Trackpad click','Can anyone recommend an app for creating flowcharts and diagrams',' How Do I Run Xcode','Dragging a window to a space doesn\'t work the first time','How can I control iTunes from another Mac','iMac won\'t recognise .MTS videos on my hard drive, how can I play them']
    correctAnswers = ['I recommend DTerm, which gives you a hovering command prompt','It swaps the function of the function keys between the OS X function and the actual F1-F12 key','Yes, the device can communicate with Find my iPhone over Wi-Fi','You can create an on the go playlist directly on the iPhone that acts like a queue','Unfortunately, you just have to remember to do it each time you make a file or directory','Sorry, Apple doesn\'t allow people to change the mouse cursor anymore','The caching server will print to log when a client requests an update','This appears to have been fixed in either Yosemite or the latest version of Office','Check out OmniFocus (available for Mac, iPhone and iPad)','You should be able to do something like this with a product like IK Multimedia\'s iRig (or a generic equivalent..','KeyMap4MacBook can actually do this','Instead of invoking using gcc you need to call gcc-4.9','You can use Handbrake to rip it to a file, then just copy to iTunes','It appears HP supports this printer with the P4014 series drivers','Yes, they are unlocked,  The price is going to be substantially higher since its not the subsidized price','Yes, it\'s called "Personal Hotspot" by Apple and works on iPhone 4 and 4S','Yes, it won\'t hurt the iMac','Unfortunately I\'m unaware of any way to do this without a workaround','Try some Goo-B-Gone or other citrus based cleaning fluid.','If you can stretch your budget, get OmniGraffle for Mac','Xcode is located in /Developer/Applications/Xcode.app','Confirmed: the mountain lion 10.8.2 update resolves this problem.','I\'ve been using TuneConnect to do pretty much what you\'re after','Try VideoLan']
    questionVariations = [['copy from Finder the current path','Can I copy the current path from Finder','can i copy from finder the current path'],['for what should I use fn key by default on macbook pro','what should I use the fn key for on mac book pro','using the fn key on mac book pro'],['how can find my iphone be working with no data plan','Does Find My iPhone work with no data plan','will Find My iPhone still work with no data plan'],['Is it possible to queue songs on iphone','How to queue songs on iPhone','Can I queue songs on an iPhone'],['how to force simple UNIX path names for optimal use of command line apps','Force simple UNIX paths for optimal use of command line apps','force simple UNIX path names for use of command line apps'],['how can you change the mouse cursor in osx','can I change the mouse cursor in OSX','can you change the mouse cursor in OSX'],['how can you tell if updates came from OS X caching server','can you tell if updates come from the OSX caching server','Can you tell if updates are from the OS X caching server'],['why do some apps fill the background of Mission Control in Mavericks','why do certain apps fill the back ground of mission control in mavericks','Why do some apps fill the background of Mission Control in OSX Mavericks'],['what is a quick entry TODO application that syncs across iPhone and laptop','I\'m looking for a quick entry TODO application that syncs across iPhone and my laptop','What is a quick entry TODO application that sycs across my iPhone and my laptop'],['How do you play audio from a Mac over an iPhone call','How to play Audio from a Mac over an iPhone call','Can you play audio from a Mac over an iPhone call'],['Map Home and End of Apple keyboard in VMWare fusion','Mapping Home and End on Apple key board in VMWare Fusion','How to map Home and End of Apply keyboard in VMWare fusion'],['How can I use GCC on El Capitan','How do you use gcc on el capitan','How to use GCC on El Capitan'],['How do you import DVD\'s to iTunes','How to import DVD to itunes','How can I import DVDs to iTunes'],['What driver for HP P4515 printer in 10.7.4','What driver do I use for the HP P4515 printer on 10.7.4','Driver for HP P4515 printer in 10.7.4'],['Is an iPhone purchased from the Canadian online Apple Store unlocked','Are iPhones purchased from Canadian online Apple Store unlocked','Is the iphone from Canadian online Apple Store factory unlocked'],['Can I share a 3g connection from iPhone/iPad by creating a Wi-fi hotspot','Can I share a 3G connection from an iPhone by creating a wifi hotspot','Can you share 3G data from an iPhone/iPad by creating a Wi-fi hotspot'],['is storing an imac sideways safe','Can I store an iMac sideways safely','Is it safe to store an iMac sideways'],['Can I change the default text color in Mail','Can you change the default text color in mail','Is it possible to change the text color in Mail'],['How do you clean a sticky Magic Trackpad click','How can I clean a sticky Magic Trackpad','How to clean a Magic Trackpad '],['recommend an app for creating flowcharts and diagrams','What is a good app for creating flowcharts and diagrams','Can you recommend a good app for creating flow charts and diagrams'],['how do i run xcode','How do you run Xcode','How can I run Xcode'],['Why won\'t dragging a window to a space work the first time','dragging a window to a space does not work the first time','Dragging window doesn\'t work the first time'],['Can I control iTunes from another Mac','How do you control itunes from another mac','How to control iTunes from a different Mac'],['how can I play .MTS videos on my hard drive','iMac doesn\'t recognize MTS videos, how can I play them?','imac won\'t recognize the .MTS videos on my hard drive so how can I play them']]

    for q in range(0, len(originalQuestions)):
      sentence = originalQuestions[q]
      correctAnswer = correctAnswers[q]
      print("Original Q:\t{0}".format(sentence))
      print("Correct A:\t{0}".format(correctAnswer))
      print_output(model, sess, enc_vocab, rev_dec_vocab, sentence)
      for v in range(0, 3):
        sentence = questionVariations[q][v]
        print("Modified Q:\t{0}".format(sentence))
        print("Correct A:\t{0}".format(correctAnswer))
        print_output(model, sess, enc_vocab, rev_dec_vocab, sentence)

def print_output(model, sess, enc_vocab, rev_dec_vocab, sentence):
  # Get token-ids for the input sentence.
  token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)
  # Which bucket does it belong to?
  bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])
  # Get a 1-element batch to feed the sentence to the model.
  encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
  # Get output logits for the sentence.
  _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
  # This is a greedy decoder - outputs are just argmaxes of output_logits.
  outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
  # If there is an EOS symbol in outputs, cut them at that point.
  if data_utils.EOS_ID in outputs:
    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
  # Print out French sentence corresponding to outputs.
  print("Bot:\t" + " ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs]))


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def init_session(sess, conf='seq2seq.ini'):
    global gConfig
    gConfig = get_config(conf)
 
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    enc_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.enc" % gConfig['enc_vocab_size'])
    dec_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.dec" % gConfig['dec_vocab_size'])

    enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
    _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)

    return sess, model, enc_vocab, rev_dec_vocab

def decode_line(sess, model, enc_vocab, rev_dec_vocab, sentence):
    # Get token-ids for the input sentence.
    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)

    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])

    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)

    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]

    return " ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])

if __name__ == '__main__':
    if len(sys.argv) - 1:
        gConfig = get_config(sys.argv[1])
    else:
        # get configuration from seq2seq.ini
        gConfig = get_config()

    print('\n>> Mode : %s\n' %(gConfig['mode']))

    if gConfig['mode'] == 'train':
        # start training
        train()
    elif gConfig['mode'] == 'test':
        # interactive decode
        runTestScript()
        #decode()
    else:
        # wrong way to execute "serve"
        #   Use : >> python ui/app.py
        #           uses seq2seq_serve.ini as conf file
        print('Serve Usage : >> python ui/app.py')
        print('# uses seq2seq_serve.ini as conf file')
