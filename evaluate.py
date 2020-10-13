#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import model as bm
import util

if __name__ == "__main__":
  config = util.initialize_from_env()
  num_fold = config['cross_validation_fold']

  #cross validataion
  if num_fold > 1:
    tp, fn, fp = 0, 0, 0
    tpa, fna, fpa = 0, 0, 0
    log_dir = config["log_dir"]
    for test_fold in xrange(num_fold):
      tf.reset_default_graph()
      config['log_dir'] = os.path.join(log_dir, '%d_of_%d' % (test_fold, num_fold))
      model = bm.BridgingModel(config)
      with tf.Session() as session:
        model.restore(session)
        ctp, cfn, cfp, ctpa, cfna, cfpa = model.evaluate(session, test_fold, num_fold, is_final_test=True)
        tp += ctp
        fn += cfn
        fp += cfp
        tpa += ctpa
        fna += cfna
        fpa += cfpa

    bridging_recall = 0.0 if tp == 0 else float(tp) / (tp + fn)
    bridging_precision = 0.0 if tp == 0 else float(tp) / (tp + fp)
    bridging_f1 = 0.0 if bridging_precision == 0.0 else 2.0 * bridging_recall * bridging_precision / (
        bridging_recall + bridging_precision)

    bridging_anaphora_recall = 0.0 if tpa == 0 else float(tpa) / (tpa + fna)
    bridging_anaphora_precision = 0.0 if tpa == 0 else float(tpa) / (tpa + fpa)
    bridging_anaphora_f1 = 0.0 if bridging_anaphora_precision == 0.0 else 2.0 * bridging_anaphora_recall * bridging_anaphora_precision / (
        bridging_anaphora_recall + bridging_anaphora_precision)

    print("Final Bridging anaphora detection F1: {:.2f}%".format(bridging_anaphora_f1 * 100))
    print("Final Bridging anaphora detection recall: {:.2f}%".format(bridging_anaphora_recall * 100))
    print("Final Bridging anaphora detection precision: {:.2f}%".format(bridging_anaphora_precision * 100))

    print("Final Bridging F1: {:.2f}%".format(bridging_f1 * 100))
    print("Final Bridging recall: {:.2f}%".format(bridging_recall * 100))
    print("Final Bridging precision: {:.2f}%".format(bridging_precision * 100))
  else:
    #evaluate on test set
    config['eval_path'] = config['test_path']

    model = bm.BridgingModel(config)
    with tf.Session() as session:
      model.restore(session)
      model.evaluate(session)
