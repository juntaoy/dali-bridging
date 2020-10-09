from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import math
import json
import numpy as np
import tensorflow as tf
import h5py

import util



class BridgingModel(object):
  def __init__(self, config):
    self.config = config
    self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
    self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
    self.char_embedding_size = config["char_embedding_size"]
    self.char_dict = util.load_char_dict(config["char_vocab_path"])
    self.max_span_width = config["max_span_width"]
    self.genres = {g: i for i, g in enumerate(config["genres"])}
    if config["lm_path"]:
      self.lm_file = h5py.File(self.config["lm_path"], "r")
    else:
      self.lm_file = None
    self.lm_layers = self.config["lm_layers"]
    self.lm_size = self.config["lm_size"]
    self.eval_data = None  # Load eval data lazily.
    self.undersampling_probability = self.config["undersampling_probability"]
    self.second_undersampling_probability = self.config["second_undersampling_probability"]
    self.cross_validation_fold = self.config["cross_validation_fold"]
    self.skip_comparative_bridging = 'skip_comparative_bridging' in self.config and self.config['skip_comparative_bridging']

    input_props = []
    input_props.append((tf.float32, [None, None, self.context_embeddings.size]))  # Context embeddings.
    input_props.append((tf.float32, [None, None, self.head_embeddings.size]))  # Head embeddings.
    input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers]))  # LM embeddings.
    input_props.append((tf.int32, [None, None, None]))  # Character indices.
    input_props.append((tf.int32, [None]))  # Text lengths.
    input_props.append((tf.int32, [None]))  # Speaker IDs.
    input_props.append((tf.int32, []))  # Genre.
    input_props.append((tf.bool, []))  # Is training.
    input_props.append((tf.int32, [None]))  # Gold starts.
    input_props.append((tf.int32, [None]))  # Gold ends.
    input_props.append((tf.int32, [None]))  # Cluster ids.
    input_props.append((tf.int32, [None]))  # Bridging antecedent cluster ids
    input_props.append((tf.int32, [None]))  # IS status 0-DN 1-DO 2-Bridging
    input_props.append((tf.bool, [None]))  # undersampling mask

    self.input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]

    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.reset_global_step = tf.assign(self.global_step, 0)

    self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
    learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                               self.config["decay_frequency"], self.config["decay_rate"],
                                               staircase=True)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
    optimizers = {
      "adam": tf.train.AdamOptimizer,
      "sgd": tf.train.GradientDescentOptimizer
    }
    optimizer = optimizers[self.config["optimizer"]](learning_rate)
    self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)


  def restore(self, session):
    # Don't try to restore unused variables from the TF-Hub ELMo module.
    vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
    saver = tf.train.Saver(vars_to_restore)
    checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
    print("Restoring from {}".format(checkpoint_path))
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint_path)

  def load_lm_embeddings(self, doc_key):
    if self.lm_file is None:
      return np.zeros([0, 0, self.lm_size, self.lm_layers])
    file_key = doc_key.replace("/", ":")
    group = self.lm_file[file_key]
    num_sentences = len(list(group.keys()))
    sentences = [group[str(i)][...] for i in range(num_sentences)]
    lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
    for i, s in enumerate(sentences):
      lm_emb[i, :s.shape[0], :, :] = s
    return lm_emb

  def tensorize_mentions(self, mentions):
    starts, ends = [], []
    for m in mentions:
      starts.append(m[0])
      ends.append(m[1])

    return np.array(starts), np.array(ends)

  def tensorize_example(self, example, is_training):
    if is_training:
      undersampling_probability = self.undersampling_probability if example[
        "main_train"] else self.second_undersampling_probability
    clusters = example["clusters"]
    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
    cluster_ids = np.zeros(len(gold_mentions))
    ismap = {}
    for cluster_id, cluster in enumerate(clusters):
      for mid, mention in enumerate(cluster):
        cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1
        ismap[(mention[0], mention[1])] = 0 if mid == 0 else 1 #0-DN 1-DO 2-Bridging

    sentences = example["sentences"]
    num_words = sum(len(s) for s in sentences)


    max_sentence_length = max(len(s) for s in sentences)
    max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
    text_len = np.array([len(s) for s in sentences])
    tokens = [[""] * max_sentence_length for _ in sentences]
    context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
    head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
    char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
    for i, sentence in enumerate(sentences):
      for j, word in enumerate(sentence):
        tokens[i][j] = word
        context_word_emb[i, j] = self.context_embeddings[word]
        head_word_emb[i, j] = self.head_embeddings[word]
        char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]

    if "speakers" in example:
      speakers = util.flatten(example["speakers"])
      assert num_words == len(speakers)
      speaker_dict = {s: i for i, s in enumerate(set(speakers))}
      speaker_ids = np.array([speaker_dict[s] for s in speakers])
    else:
      speaker_ids = np.zeros(num_words,dtype=np.int32)

    doc_key = example["doc_key"]
    genre = self.genres[doc_key[:2]]
    gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
    gold_mentions_to_clusters = {(s, e): cid for s, e, cid in zip(gold_starts, gold_ends, cluster_ids)}

    if self.skip_comparative_bridging and 'bridging_pairs_no_comparative' in example:
      #for training ISNOTE and using BASHI as second train corpus, see Hou et al 2020 ACL for more detail
      bridging_map = {(p[0][0], p[0][1]): (p[1][0], p[1][1]) for p in example["bridging_pairs_no_comparative"]}
    else:
      bridging_map = {(p[0][0], p[0][1]): (p[1][0], p[1][1]) for p in example["bridging_pairs"]}
    bridging_ante_cids = np.zeros(len(gold_mentions), np.int32)
    is_status = np.zeros(len(gold_mentions), np.int32)
    us_mask = np.ones(len(gold_mentions), dtype=np.bool)
    mask_cap = np.ones(len(gold_mentions))
    for i, (s, e) in enumerate(zip(gold_starts, gold_ends)):
      if (s, e) in bridging_map:
        bridging_ante_cids[i] = gold_mentions_to_clusters[bridging_map[(s, e)]]
        is_status[i] = 2  # 2-bridging
        mask_cap[i] = 1.0
      else:
        is_status[i] = ismap[(s, e)]  # 0-DN 1-DO
        if is_training:
          mask_cap[i] = undersampling_probability[is_status[i]] * self.config["ntimes_negative_examples"]


    lm_emb = self.load_lm_embeddings(doc_key)
    if is_training:
      return (context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts,
    gold_ends, cluster_ids, bridging_ante_cids, is_status,mask_cap)

    return (context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts,
    gold_ends, cluster_ids, bridging_ante_cids, is_status, us_mask)

  def truncate_example(self, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre,
                       is_training, gold_starts, gold_ends, cluster_ids, bridging_ante_cids, is_status,mask_cap):
    num_words = sum(text_len)
    us_mask = np.random.random(gold_starts.shape[0]) < mask_cap

    if num_words > self.config["max_training_words"]:
      num_sentences = context_word_emb.shape[0]
      max_training_sentences = num_sentences

      while num_words > self.config["max_training_words"]:
        max_training_sentences -= 1
        sentence_offset = random.randint(0, num_sentences - max_training_sentences)
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()

      context_word_emb = context_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
      head_word_emb = head_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
      lm_emb = lm_emb[sentence_offset:sentence_offset + max_training_sentences, :, :, :]
      char_index = char_index[sentence_offset:sentence_offset + max_training_sentences, :, :]
      text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

      speaker_ids = speaker_ids[word_offset: word_offset + num_words]
      gold_spans = np.logical_and(gold_starts >= word_offset, gold_ends < word_offset + num_words)
      gold_starts = gold_starts[gold_spans] - word_offset
      gold_ends = gold_ends[gold_spans] - word_offset
      cluster_ids = cluster_ids[gold_spans]
      bridging_ante_cids = bridging_ante_cids[gold_spans]
      is_status = is_status[gold_spans]
      us_mask = us_mask[gold_spans]

    return (context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, bridging_ante_cids, is_status, us_mask)

  def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels,
                           bridging_ante_cids):
    same_start = tf.equal(tf.expand_dims(labeled_starts, 1),
                          tf.expand_dims(candidate_starts, 0))  # [num_labeled, num_candidates]
    same_end = tf.equal(tf.expand_dims(labeled_ends, 1),
                        tf.expand_dims(candidate_ends, 0))  # [num_labeled, num_candidates]
    same_span = tf.logical_and(same_start, same_end)  # [num_labeled, num_candidates]
    candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
    candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
    candidate_bridging_ante_cids = tf.matmul(tf.expand_dims(bridging_ante_cids, 0),
                                             tf.to_int32(same_span))  # [1, num_candidates]
    candidate_bridging_ante_cids = tf.squeeze(candidate_bridging_ante_cids, 0)  # [num_candidates]
    return candidate_labels, candidate_bridging_ante_cids

  def get_dropout(self, dropout_rate, is_training):
    return 1 - (tf.to_float(is_training) * dropout_rate)

  
  def distance_pruning(self, top_span_emb, top_span_mention_scores, c):
    k = util.shape(top_span_emb, 0)
    top_antecedent_offsets = tf.tile(tf.expand_dims(tf.range(c) + 1, 0), [k, 1])  # [k, c]
    raw_top_antecedents = tf.expand_dims(tf.range(k), 1) - top_antecedent_offsets  # [k, c]
    top_antecedents_mask = raw_top_antecedents >= 0  # [k, c]
    top_antecedents = tf.maximum(raw_top_antecedents, 0)  # [k, c]

    top_fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.gather(top_span_mention_scores,
                                                                                        top_antecedents)  # [k, c]
    top_fast_antecedent_scores += tf.log(tf.to_float(top_antecedents_mask))  # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

  def get_predictions_and_loss(self, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids,
                               genre, is_training, gold_starts, gold_ends, cluster_ids, bridging_ante_cids, is_status,
                               us_mask):
    self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
    self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
    self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

    num_sentences = tf.shape(context_word_emb)[0]
    max_sentence_length = tf.shape(context_word_emb)[1]

    context_emb_list = [context_word_emb]
    head_emb_list = [head_word_emb]

    if self.config["char_embedding_size"] > 0:
      char_emb = tf.gather(
        tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]),
        char_index)  # [num_sentences, max_sentence_length, max_word_length, emb]
      flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2),
                                                 util.shape(char_emb,
                                                            3)])  # [num_sentences * max_sentence_length, max_word_length, emb]
      flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"],
                                               self.config["filter_size"])  # [num_sentences * max_sentence_length, emb]
      aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length,
                                                                       util.shape(flattened_aggregated_char_emb,
                                                                                  1)])  # [num_sentences, max_sentence_length, emb]
      context_emb_list.append(aggregated_char_emb)
      head_emb_list.append(aggregated_char_emb)

    if self.lm_file:
      lm_emb_size = util.shape(lm_emb, 2)
      lm_num_layers = util.shape(lm_emb, 3)
      with tf.variable_scope("lm_aggregation"):
        self.lm_weights = tf.nn.softmax(
          tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
        self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))
      flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
      flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights,
                                                                               1))  # [num_sentences * max_sentence_length * emb, 1]
      aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
      aggregated_lm_emb *= self.lm_scaling
      context_emb_list.append(aggregated_lm_emb)

    context_emb = tf.concat(context_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
    head_emb = tf.concat(head_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
    context_emb = tf.nn.dropout(context_emb, self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]
    head_emb = tf.nn.dropout(head_emb, self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]

    text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)  # [num_sentence, max_sentence_length]

    context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask)  # [num_words, emb]

    genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]]),
                          genre)  # [emb]

    flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask)  # [num_words]


    top_span_starts = gold_starts
    top_span_ends = gold_ends
    top_span_cluster_ids = cluster_ids

    top_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, top_span_starts, top_span_ends)
    top_span_mention_scores = tf.zeros_like(gold_starts, dtype=tf.float32)  # [k]

    top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts)
    top_span_bridging_ante_cids = bridging_ante_cids
    top_us_mask = us_mask
    top_is_status = is_status

    k = util.shape(top_span_starts, 0)

    c = tf.minimum(self.config["max_top_antecedents"], k)

    top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.distance_pruning(
      top_span_emb, top_span_mention_scores, c)

    top_antecedent_emb = tf.gather(top_span_emb, top_antecedents)  # [k, c, emb]

    pair_emb = self.get_pair_embeddings(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets,
                                        top_span_speaker_ids, genre_emb)  # [k, c,emb]

    top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents)  # [k, c]
    top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask)))  # [k, c]

    shared_depth = 0
    if self.config["shared_depth"] > 0:
      flattened_pair_emb = tf.reshape(pair_emb, [k * c, util.shape(pair_emb, 2)])
      shared_depth = min(self.config["shared_depth"], self.config["ffnn_depth"])
      for i in range(shared_depth):
        hidden_weights = tf.get_variable("shared_hidden_weights_{}".format(i),
                                         [util.shape(flattened_pair_emb, 1), self.config["ffnn_size"]])
        hidden_bias = tf.get_variable("shared_hidden_bias_{}".format(i), [self.config["ffnn_size"]])
        flattened_pair_emb = tf.nn.relu(tf.nn.xw_plus_b(flattened_pair_emb, hidden_weights, hidden_bias))
        flattened_pair_emb = tf.nn.dropout(flattened_pair_emb, self.dropout)
      pair_emb = tf.reshape(flattened_pair_emb, [k, c, self.config["ffnn_size"]])

    ante_score_list = []
    pairwise_label_list = []
    dummy_scores = tf.zeros([k, 1])  # [k, 1]
    ante_score_list.append(dummy_scores)

    with tf.variable_scope("slow_bridging_scores"):
      slow_bridging_scores = util.ffnn(pair_emb, self.config["ffnn_depth"] - shared_depth, self.config["ffnn_size"], 1,
                                       self.dropout)  # [k, c, 1]
      slow_bridging_scores = tf.squeeze(slow_bridging_scores, 2)  # [k, c]
      top_bridging_scores = slow_bridging_scores + top_fast_antecedent_scores
      ante_score_list.append(top_bridging_scores)

    bridging_cluster_indicator = tf.equal(top_antecedent_cluster_ids,
                                          tf.expand_dims(top_span_bridging_ante_cids, 1))  # [k, c]
    non_dummy_bridging_indicator = tf.expand_dims(top_span_bridging_ante_cids > 0, 1)  # [k, 1]

    bridging_pairwise_labels = tf.logical_and(bridging_cluster_indicator, non_dummy_bridging_indicator)  # [k, c]
    pairwise_label_list.append(bridging_pairwise_labels)

    if self.config["train_with_coref"]:
      with tf.variable_scope("slow_coreference_scores"):
        slow_coref_scores = util.ffnn(pair_emb, self.config["ffnn_depth"] - shared_depth, self.config["ffnn_size"], 1,
                                      self.dropout)  # [k, c, 1]
        slow_coref_scores = tf.squeeze(slow_coref_scores, 2)  # [k, c]
        top_coref_scores = slow_coref_scores + top_fast_antecedent_scores
        ante_score_list.append(top_coref_scores)

      coref_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1))  # [k,c]

      non_dummy_coref_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1)  # [k,1]

      coref_pairwise_labels = tf.logical_and(coref_cluster_indicator, non_dummy_coref_indicator)  # [k,c]
      pairwise_label_list.append(coref_pairwise_labels)

    top_antecedent_scores = tf.concat(ante_score_list, 1)  # [k, c + 1] or [k, 2*c+1]
    pairwise_labels = tf.concat(pairwise_label_list, 1)  # [k,c] or [k,2*c]

    top_antecedent_scores = tf.boolean_mask(top_antecedent_scores, top_us_mask)
    pairwise_labels = tf.boolean_mask(pairwise_labels, top_us_mask)


    dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True))  # [k, 1]
    pairwise_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, c + 1] or [k,2*c+1]

    loss = self.softmax_loss(top_antecedent_scores, pairwise_labels)
    loss = tf.reduce_sum(loss)

    if self.config["use_gold_bridging_anaphora"]:
      bridging_mask = tf.equal(top_is_status, 2)  # bridging
      top_span_starts = tf.boolean_mask(top_span_starts, bridging_mask)
      top_span_ends = tf.boolean_mask(top_span_ends, bridging_mask)
      top_antecedents = tf.boolean_mask(top_antecedents, bridging_mask)
      top_antecedent_scores_output = tf.boolean_mask(top_bridging_scores, bridging_mask)
    elif self.config["remove_coref_anaphora"]:
      bridging_mask = tf.not_equal(top_is_status, 1)  # DO
      top_span_starts = tf.boolean_mask(top_span_starts, bridging_mask)
      top_span_ends = tf.boolean_mask(top_span_ends, bridging_mask)
      top_antecedents = tf.boolean_mask(top_antecedents, bridging_mask)
      top_antecedent_scores_output = tf.boolean_mask(tf.concat([dummy_scores, top_bridging_scores], 1), bridging_mask)
    else:
      top_antecedent_scores_output = top_antecedent_scores

    return [top_span_starts, top_span_ends, top_span_cluster_ids, top_antecedents, top_antecedent_scores_output], loss

  def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
    span_emb_list = []

    span_start_emb = tf.gather(context_outputs, span_starts)  # [k, emb]
    span_emb_list.append(span_start_emb)

    span_end_emb = tf.gather(context_outputs, span_ends)  # [k, emb]
    span_emb_list.append(span_end_emb)

    span_width = tf.minimum(1 + span_ends - span_starts, self.config["max_span_width"])  # [k]

    if self.config["use_features"]:
      span_width_index = span_width - 1  # [k]
      span_width_emb = tf.gather(
        tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]),
        span_width_index)  # [k, emb]
      span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
      span_emb_list.append(span_width_emb)

    if self.config["model_heads"]:
      span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts,
                                                                                                 1)  # [k, max_span_width]
      span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices)  # [k, max_span_width]
      span_text_emb = tf.gather(head_emb, span_indices)  # [k, max_span_width, emb]
      with tf.variable_scope("head_scores"):
        self.head_scores = util.projection(context_outputs, 1)  # [num_words, 1]
      span_head_scores = tf.gather(self.head_scores, span_indices)  # [k, max_span_width, 1]
      span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32),
                                 2)  # [k, max_span_width, 1]
      span_head_scores += tf.log(span_mask)  # [k, max_span_width, 1]
      span_attention = tf.nn.softmax(span_head_scores, 1)  # [k, max_span_width, 1]
      span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1)  # [k, emb]
      span_emb_list.append(span_head_emb)

    span_emb = tf.concat(span_emb_list, 1)  # [k, emb]
    return span_emb  # [k, emb]

  
  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [k, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]
    return log_norm - marginalized_gold_scores  # [k]

  def bucket_distance(self, distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances)) / math.log(2))) + 3
    use_identity = tf.to_int32(distances <= 4)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.clip_by_value(combined_idx, 0, 9)

  def get_pair_embeddings(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets,
                          top_span_speaker_ids, genre_emb):
    k = util.shape(top_span_emb, 0)
    c = util.shape(top_antecedents, 1)

    feature_emb_list = []

    if self.config["use_metadata"]:
      top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents)  # [k, c]
      same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids)  # [k, c]
      speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]),
                                   tf.to_int32(same_speaker))  # [k, c, emb]
      feature_emb_list.append(speaker_pair_emb)

      tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1])  # [k, c, emb]
      feature_emb_list.append(tiled_genre_emb)

    if self.config["use_features"]:
      antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets)  # [k, c]
      antecedent_distance_emb = tf.gather(tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]),
                                          antecedent_distance_buckets)  # [k, c]
      feature_emb_list.append(antecedent_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2)  # [k, c, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.dropout)  # [k, c, emb]

    target_emb = tf.expand_dims(top_span_emb, 1)  # [k, 1, emb]
    similarity_emb = top_antecedent_emb * target_emb  # [k, c, emb]
    target_emb = tf.tile(target_emb, [1, c, 1])  # [k, c, emb]

    pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)  # [k, c, emb]

    return pair_emb

  
  def flatten_emb_by_sentence(self, emb, text_len_mask):
    num_sentences = tf.shape(emb)[0]
    max_sentence_length = tf.shape(emb)[1]

    emb_rank = len(emb.get_shape())
    if emb_rank == 2:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
    return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

  def lstm_contextualize(self, text_emb, text_len, text_len_mask):
    num_sentences = tf.shape(text_emb)[0]

    current_inputs = text_emb  # [num_sentences, max_sentence_length, emb]

    for layer in range(self.config["contextualization_layers"]):
      with tf.variable_scope("layer_{}".format(layer)):
        with tf.variable_scope("fw_cell"):
          cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
        with tf.variable_scope("bw_cell"):
          cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
        state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]),
                                                 tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
        state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]),
                                                 tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

        (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw,
          cell_bw=cell_bw,
          inputs=current_inputs,
          sequence_length=text_len,
          initial_state_fw=state_fw,
          initial_state_bw=state_bw)

        text_outputs = tf.concat([fw_outputs, bw_outputs], 2)  # [num_sentences, max_sentence_length, emb]
        text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)
        if layer > 0:
          highway_gates = tf.sigmoid(
            util.projection(text_outputs, util.shape(text_outputs, 2)))  # [num_sentences, max_sentence_length, emb]
          text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
        current_inputs = text_outputs

    return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

  def get_predicted_bridging_pairs(self, predictions):
    top_span_starts, top_span_ends, top_span_cluster_ids, top_antecedents, top_antecedent_scores = predictions
    pred_bridging_pairs = set()
    pred_bridging_anaphora = set()
    c = np.shape(top_antecedents)[1]  # c
    pred_ante_indices = np.argmax(top_antecedent_scores, axis=1) - (
      0 if self.config["use_gold_bridging_anaphora"] else 1)

    for i, index in enumerate(pred_ante_indices):
      if self.config["use_gold_bridging_anaphora"] or (index < c and index >= 0):
        pred_bridging_pairs.add((top_span_starts[i], top_span_ends[i], top_span_cluster_ids[top_antecedents[i, index]]))
        pred_bridging_anaphora.add((top_span_starts[i], top_span_ends[i]))

    return pred_bridging_pairs, pred_bridging_anaphora

  def get_gold_bridging_pairs(self, gold_starts, gold_ends, cluster_ids, bridgings,pred_bridging_pairs):
    bridging_map = {}
    mention2cid = {(s, e): cid for s, e, cid in zip(gold_starts, gold_ends, cluster_ids)}
    gold_bridging_pairs = set()
    gold_bridging_anaphora = set()
    for p in bridgings:
      ana = (p[0][0], p[0][1])
      ant_cid = mention2cid[(p[1][0], p[1][1])]
      if not ana in bridging_map:
        bridging_map[ana] = set()
      bridging_map[ana].add(ant_cid)

    for ana in bridging_map:
      gold_bridging_anaphora.add(ana)
      s, e = ana
      ant_cid = list(bridging_map[ana])
      if len(ant_cid) > 1:
        cid = ant_cid[0]
        for c in ant_cid:
          if (s, e, c) in pred_bridging_pairs:
            # we follow hou et al to count bridgings with multi-antecedent as correct
            # as long as any gold bridging ant is recovered (only for BASHI) (only 15 multi-ant)
            # The impact on results are very small e.g. 0 ~ 0.2
            cid = c
        gold_bridging_pairs.add((s, e, cid))
      else:
        gold_bridging_pairs.add((s, e, ant_cid[0]))
    return gold_bridging_pairs, gold_bridging_anaphora

  def load_eval_data(self, eval_fold=-1, num_fold=-1):
    if self.eval_data is None:
      def load_line(line):
        example = json.loads(line)
        return self.tensorize_example(example, is_training=False), example
      if num_fold > 1: #cross validation
        with open(self.config["train_path"]) as f:
          self.eval_data = [load_line(l) for i, l in enumerate(f.readlines()) if i % num_fold == eval_fold]
      else:
        with open(self.config["eval_path"]) as f:
          self.eval_data = [load_line(l) for l in f.readlines()]
      print("Loaded {} eval examples.".format(len(self.eval_data)))

  def evaluate(self, session, eval_fold=-1, num_fold=-1, is_final_test=False):
    self.load_eval_data(eval_fold, num_fold)

    if "eval_on_test_part_only" in self.config and self.config["eval_on_test_part_only"]:
      eval_on_test_part_only = True
      print("Evaluate on the test part only!!!!")
    else:
      eval_on_test_part_only = False

    if num_fold > 1 and is_final_test:
      print("Evaluating %d/%d fold." % (eval_fold+1, num_fold))
    tp, fn, fp = 0, 0, 0
    tpa, fna, fpa = 0, 0, 0

    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
      _, _, _, _, _, _, _, _, gold_starts, gold_ends, cluster_ids, bridging_ante_cids, _, _,  = tensorized_example
      feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
      predictions = session.run(self.predictions, feed_dict=feed_dict)

      pred_bridging_pairs, pred_bridging_anaphora = self.get_predicted_bridging_pairs(predictions)
      if self.config["has_multi_bridging_ant"]:
        #we follow hou et al to count birdings with multi-antecedent as correct
        # as long as any gold bridging ant is recovered (only for BASHI)
        gold_bridging_pairs, gold_bridging_anaphora = self.get_gold_bridging_pairs(gold_starts, gold_ends, cluster_ids,
                                                                                   example["bridging_pairs"],
                                                                                   pred_bridging_pairs)
      else:
        gold_bridging_pairs = set(
          [(s, e, cid) for s, e, cid in zip(gold_starts, gold_ends, bridging_ante_cids) if cid > 0])
        gold_bridging_anaphora = set(
          [(s, e) for s, e, cid in zip(gold_starts, gold_ends, bridging_ante_cids) if cid > 0])

      add2eval = True
      if eval_on_test_part_only and not example["doc_key"].endswith('_test'):
        add2eval = False
      if add2eval:
        tp += len(gold_bridging_pairs & pred_bridging_pairs)
        fn += len(gold_bridging_pairs - pred_bridging_pairs)
        fp += len(pred_bridging_pairs - gold_bridging_pairs)

        tpa += len(gold_bridging_anaphora & pred_bridging_anaphora)
        fna += len(gold_bridging_anaphora - pred_bridging_anaphora)
        fpa += len(pred_bridging_anaphora - gold_bridging_anaphora)

      if example_num % 10 == 0:
        print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

    bridging_recall = 0.0 if tp == 0 else float(tp) / (tp + fn)
    bridging_precision = 0.0 if tp == 0 else float(tp) / (tp + fp)
    bridging_f1 = 0.0 if bridging_precision == 0.0 else 2.0 * bridging_recall * bridging_precision / (
          bridging_recall + bridging_precision)

    bridging_anaphora_recall = 0.0 if tpa == 0 else float(tpa) / (tpa + fna)
    bridging_anaphora_precision = 0.0 if tpa == 0 else float(tpa) / (tpa + fpa)
    bridging_anaphora_f1 = 0.0 if bridging_anaphora_precision == 0.0 else 2.0 * bridging_anaphora_recall * bridging_anaphora_precision / (
        bridging_anaphora_recall + bridging_anaphora_precision)


    print("Bridging anaphora detection F1: {:.2f}%".format(bridging_anaphora_f1 * 100))
    print("Bridging anaphora detection recall: {:.2f}%".format(bridging_anaphora_recall * 100))
    print("Bridging anaphora detection precision: {:.2f}%".format(bridging_anaphora_precision * 100))

    print("Bridging F1: {:.2f}%".format(bridging_f1 * 100))
    print("Bridging recall: {:.2f}%".format(bridging_recall * 100))
    print("Bridging precision: {:.2f}%".format(bridging_precision * 100))

    summary_dict = {}
    summary_dict["Bridging anaphora detection F1"] = bridging_anaphora_f1
    summary_dict["Bridging anaphora detection recall"] = bridging_anaphora_recall
    summary_dict["Bridging anaphora detection precision"] = bridging_anaphora_precision

    summary_dict["Bridging F1"] = bridging_f1
    summary_dict["Bridging recall"] = bridging_recall
    summary_dict["Bridging precision"] = bridging_precision
    f1 = bridging_f1

    if is_final_test:
      return tp, fn, fp, tpa, fna, fpa
    return util.make_summary(summary_dict), f1 * 100
