#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json
import random

import tensorflow as tf
import model as bm
import util

def get_training_examples(config, test_fold=-1):
  num_fold = config['cross_validation_fold']
  train_examples = []
  cnt = 0
  for i, line in enumerate(open(config["train_path"])):
    example = json.loads(line)
    example['main_train'] = True
    cnt += 1
    if num_fold <=1 or i % num_fold != test_fold:
      train_examples.append(example)
  print("Find %d documents from %s use %d" % (cnt, config['train_path'], len(train_examples)))

  if config["second_train_path"]:
    cnt = 0
    for line in open(config["second_train_path"]):
      cnt += 1
      example = json.loads(line)
      example['main_train'] = False
      train_examples.append(example)

    print("Using %d additional documents from %s." % (cnt, config['second_train_path']))
  return train_examples


if __name__ == "__main__":

  config = util.initialize_from_env()
  num_fold = config['cross_validation_fold']
  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]
  max_step = config["max_step"]


  if num_fold > 1:
    root_log_dir = config["log_dir"]
    for test_fold in xrange(num_fold):
      print("\n\nStarting %d of %d fold" % (test_fold+1,num_fold))
      tf.reset_default_graph()

      config["log_dir"] = util.mkdirs(os.path.join(root_log_dir, '%d_of_%d' % (test_fold, num_fold)))
      model = bm.BridgingModel(config)
      saver = tf.train.Saver(max_to_keep=1)
      log_dir = config["log_dir"]
      writer = tf.summary.FileWriter(log_dir, flush_secs=20)

      session_config = tf.ConfigProto()
      session_config.gpu_options.allow_growth = True
      session_config.allow_soft_placement = True


      with tf.Session(config=session_config) as session:
        session.run(tf.global_variables_initializer())
        accumulated_loss = 0.0

        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
          print("Restoring from: {}".format(ckpt.model_checkpoint_path))
          saver.restore(session, ckpt.model_checkpoint_path)

        initial_time = time.time()
        tf_global_step = 0

        train_examples = [model.tensorize_example(example, is_training=True) for example in
                          get_training_examples(config, test_fold)]

        while max_step<=0 or tf_global_step<max_step:
          random.shuffle(train_examples)
          for example in train_examples:
            feed_dict = dict(zip(model.input_tensors, model.truncate_example(*example)))

            tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op],feed_dict=feed_dict)
            accumulated_loss += tf_loss

            if tf_global_step % report_frequency == 0:
              total_time = time.time() - initial_time
              steps_per_second = tf_global_step / total_time

              average_loss = accumulated_loss / report_frequency
              print("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
              writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
              accumulated_loss = 0.0

            if tf_global_step % eval_frequency == 0:
              saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
              util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)),
                                   os.path.join(log_dir, "model.max.ckpt"))



  else:
    model = bm.BridgingModel(config)
    saver = tf.train.Saver(max_to_keep=1)

    train_examples = get_training_examples(config)

    log_dir = config["log_dir"]
    writer = tf.summary.FileWriter(log_dir, flush_secs=20)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True

    max_f1 = -1
    best_step = 0
    with tf.Session(config=session_config) as session:
      session.run(tf.global_variables_initializer())
      accumulated_loss = 0.0

      ckpt = tf.train.get_checkpoint_state(log_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print("Restoring from: {}".format(ckpt.model_checkpoint_path))
        saver.restore(session, ckpt.model_checkpoint_path)

      initial_time = time.time()
      tf_global_step = 0

      train_examples = [model.tensorize_example(example, is_training=True) for example in
                        get_training_examples(config)]

      while max_step <= 0 or tf_global_step < max_step:
        random.shuffle(train_examples)
        for example in train_examples:
          feed_dict = dict(zip(model.input_tensors, model.truncate_example(*example)))

          tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op], feed_dict=feed_dict)
          accumulated_loss += tf_loss

          if tf_global_step % report_frequency == 0:
            total_time = time.time() - initial_time
            steps_per_second = tf_global_step / total_time

            average_loss = accumulated_loss / report_frequency
            print("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
            writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
            accumulated_loss = 0.0

          if tf_global_step % eval_frequency == 0:
            saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
            eval_summary, eval_f1 = model.evaluate(session)

            if eval_f1 > max_f1:
              max_f1 = eval_f1
              best_step = tf_global_step
              util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

            writer.add_summary(eval_summary, tf_global_step)
            writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

            print("[{}] evaL_f1={:.2f}, max_f1={:.2f} at step {}".format(tf_global_step, eval_f1, max_f1, best_step))

        