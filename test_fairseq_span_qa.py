from fstokenizers import FairSeqSPTokenizer, char_anchors_to_tok_pos
from tqdm import tqdm
import numpy as np
import json
import torch
from torch import nn
from glob import glob
import random
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_files', metavar='DIR', default='')
parser.add_argument('--tokenizer_dir', metavar='DIR', default='')
parser.add_argument('--model_file', metavar='DIR', default='')
parser.add_argument('--max_seq_length', default=512, type=int)
parser.add_argument('--max_query_length', default=368, type=int)
parser.add_argument('--doc_stride', default=128, type=int)



args = parser.parse_args()
test_files = args.test_files
max_seq_length = args.max_seq_length
max_query_length = args.max_query_length
doc_stride = args.doc_stride
model_file = args.model_file

tk = FairSeqSPTokenizer(args.tokenizer_dir)







##############################################################################
##############################################################################
####
####   Custom data utils
####
##############################################################################
##############################################################################

    
import marshal

def read(dat):
    uid, inp, start, end, unanswerable = marshal.loads(dat)
    inp = np.frombuffer(inp, dtype=np.uint32).astype(np.int32)
    return uid, inp, start, end, unanswerable

def fread(f):
    uid, inp, start, end, unanswerable = marshal.load(f)
    inp = np.frombuffer(inp, dtype=np.uint32).astype(np.int32)
    return uid, inp, start, end, unanswerable
            
         
def pad(list_of_tokens, 
        max_seq_length,
        dtype=np.long,
        torch_tensor=None,
        pad_idx=1):
    k = np.empty((len(list_of_tokens),max_seq_length), dtype=dtype)
    k.fill(pad_idx)
    i = 0
    for tokens in list_of_tokens:
        k[i,:len(tokens)] = tokens
        i += 1
    return k if torch_tensor is None else torch_tensor(k)


def chunks(l, n):
    if type(l) == type((e for e in range(1))):
        it = iter(l)
        while True:
            out = []
            try:
                for _ in range(n):
                    out.append(next(it))
            except StopIteration:
                if out:
                    yield out
                break

            yield out
    else:
    
        for i in range(0, len(l), n):
            yield l[i:i + n]

def from_records(records, max_seq_length):
    fn_style = isinstance(records,str)
    if fn_style:
      def from_file(fn):
        with open(fn, 'rb') as f:
            while True:
                try:
                    record = fread(f)
                    yield record
                except EOFError:
                    break
      records = from_file(records)

    records = list(records)
      
    prepared_records = []
    for record_samples in chunks(records,48):
        uid, inp, start, end, unanswerable = zip(*record_samples) if fn_style else zip(*(read(record) for record in record_samples))
        start = start
        end = end
        unanswerable = unanswerable
        inp = pad(inp, max_seq_length,dtype=np.long, torch_tensor=torch.LongTensor)

        for e in zip(inp, start, end, unanswerable):
            yield e




            
            
            


##############################################################################
##############################################################################
####
####   Borrowed from original SQuAD evaluate file
####
##############################################################################
##############################################################################

            


import collections
import json
import numpy as np
import os
import re
import string
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

def make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid_to_has_ans[qa['id']] = bool(qa['answers'] if 'answers' in qa else qa['answer_text'])
  return qid_to_has_ans

exclude = set(string.punctuation)
remove_articles_regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    return remove_articles_regex.sub(' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return re.findall(r'[\u4e00-\u9fff]|[^ ]+',normalize_answer(s))
  #return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def get_raw_scores(dataset, preds):
  exact_scores = {}
  f1_scores = {}
  wrongs = []
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid = qa['id']
        gold_answers = [a['text'] for a in qa['answers']
                        if normalize_answer(a['text'])] if 'answers' in qa else [qa['answer_text']]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = ['']
        if qid not in preds:
          print('Missing prediction for %s' % qid)
          continue
        a_pred = preds[qid]
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
        if f1_scores[qid] < 0.5:
          wrongs.append(qid)
          
  return exact_scores, f1_scores, wrongs

def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
  new_scores = {}
  for qid, s in scores.items():
    pred_na = na_probs[qid] > na_prob_thresh
    if pred_na:
      new_scores[qid] = float(not qid_to_has_ans[qid])
    else:
      new_scores[qid] = s
  return new_scores

def make_eval_dict(exact_scores, f1_scores, qid_list=None):
  if not qid_list:
    total = len(exact_scores)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores.values()) / total),
        ('f1', 100.0 * sum(f1_scores.values()) / total),
        ('total', total),
    ])
  else:
    qid_list = [e for e in qid_list if e in exact_scores]
    total = len(qid_list)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
        ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
        ('total', total),
    ])

def merge_eval(main_eval, new_eval, prefix):
  for k in new_eval:
    main_eval['%s_%s' % (prefix, k)] = new_eval[k]

def plot_pr_curve(precisions, recalls, out_image, title):
  plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
  plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.xlim([0.0, 1.05])
  plt.ylim([0.0, 1.05])
  plt.title(title)
  plt.savefig(out_image)
  plt.clf()

def make_precision_recall_eval(scores, na_probs, num_true_pos, qid_to_has_ans,
                               out_image=None, title=None):
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  true_pos = 0.0
  cur_p = 1.0
  cur_r = 0.0
  precisions = [1.0]
  recalls = [0.0]
  avg_prec = 0.0
  for i, qid in enumerate(qid_list):
    if qid_to_has_ans[qid]:
      true_pos += scores[qid]
    cur_p = true_pos / float(i+1)
    cur_r = true_pos / float(num_true_pos)
    if i == len(qid_list) - 1 or na_probs[qid] != na_probs[qid_list[i+1]]:
      # i.e., if we can put a threshold after this point
      avg_prec += cur_p * (cur_r - recalls[-1])
      precisions.append(cur_p)
      recalls.append(cur_r)
  if out_image:
    plot_pr_curve(precisions, recalls, out_image, title)
  return {'ap': 100.0 * avg_prec}

def run_precision_recall_analysis(main_eval, exact_raw, f1_raw, na_probs, 
                                  qid_to_has_ans, out_image_dir):
  if out_image_dir and not os.path.exists(out_image_dir):
    os.makedirs(out_image_dir)
  num_true_pos = sum(1 for v in qid_to_has_ans.values() if v)
  if num_true_pos == 0:
    return
  pr_exact = make_precision_recall_eval(
      exact_raw, na_probs, num_true_pos, qid_to_has_ans,
      out_image=os.path.join(out_image_dir, 'pr_exact.png'),
      title='Precision-Recall curve for Exact Match score')
  pr_f1 = make_precision_recall_eval(
      f1_raw, na_probs, num_true_pos, qid_to_has_ans,
      out_image=os.path.join(out_image_dir, 'pr_f1.png'),
      title='Precision-Recall curve for F1 score')
  oracle_scores = {k: float(v) for k, v in qid_to_has_ans.items()}
  pr_oracle = make_precision_recall_eval(
      oracle_scores, na_probs, num_true_pos, qid_to_has_ans,
      out_image=os.path.join(out_image_dir, 'pr_oracle.png'),
      title='Oracle Precision-Recall curve (binary task of HasAns vs. NoAns)')
  merge_eval(main_eval, pr_exact, 'pr_exact')
  merge_eval(main_eval, pr_f1, 'pr_f1')
  merge_eval(main_eval, pr_oracle, 'pr_oracle')

def histogram_na_prob(na_probs, qid_list, image_dir, name):
  if not qid_list:
    return
  x = [na_probs[k] for k in qid_list]
  weights = np.ones_like(x) / float(len(x))
  plt.hist(x, weights=weights, bins=20, range=(0.0, 1.0))
  plt.xlabel('Model probability of no-answer')
  plt.ylabel('Proportion of dataset')
  plt.title('Histogram of no-answer probability: %s' % name)
  plt.savefig(os.path.join(image_dir, 'na_prob_hist_%s.png' % name))
  plt.clf()

def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
  num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
  cur_score = num_no_ans
  best_score = cur_score
  best_thresh = 0.0
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  for i, qid in enumerate(qid_list):
    if qid not in scores: continue
    if qid_to_has_ans[qid]:
      diff = scores[qid]
    else:
      if preds[qid]:
        diff = -1
      else:
        diff = 0
    cur_score += diff
    if cur_score > best_score:
      best_score = cur_score
      best_thresh = na_probs[qid]
  return 100.0 * best_score / len(scores), best_thresh

def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
  best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
  best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
  main_eval['best_exact'] = best_exact
  main_eval['best_exact_thresh'] = exact_thresh
  main_eval['best_f1'] = best_f1
  main_eval['best_f1_thresh'] = f1_thresh

def evaluate(dataset, preds, na_probs=None, na_prob_thresh=1.0, out_file=None, out_image_dir=None):
  if na_probs is None:
    na_probs = {k: 0.0 for k in preds}
  qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
  has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
  no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
  exact_raw, f1_raw, wrongs = get_raw_scores(dataset, preds)
  exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans,
                                        na_prob_thresh)
  f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans,
                                     na_prob_thresh)
  out_eval = make_eval_dict(exact_thresh, f1_thresh)
  if has_ans_qids:
    has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
    merge_eval(out_eval, has_ans_eval, 'HasAns')
  if no_ans_qids:
    no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
    merge_eval(out_eval, no_ans_eval, 'NoAns')
  if na_probs:
    find_all_best_thresh(out_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans)
  if na_probs and out_image_dir:
    run_precision_recall_analysis(out_eval, exact_raw, f1_raw, na_probs, 
                                  qid_to_has_ans, out_image_dir)
    histogram_na_prob(na_probs, has_ans_qids, out_image_dir, 'hasAns')
    histogram_na_prob(na_probs, no_ans_qids, out_image_dir, 'noAns')
    
  if out_file:
    with open(out_file, 'w') as f:
      json.dump(out_eval, f)
  else:
    print(json.dumps(out_eval, indent=2))
  return out_eval, exact_raw, f1_raw, wrongs

            
            
##############################################################################
##############################################################################
####
####   SQuAD Utils
####
##############################################################################
##############################################################################

            
            

import collections
_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index",
    "start_log_prob", "end_log_prob", "this_paragraph_text",
    "cur_null_score"])
_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob","cur_null_score"])

import math
def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs
            
            
            
            
            

def handle_prediction_by_qid(self, 
                             prediction_by_qid, 
                             start_n_top = 5,
                             end_n_top = 5,
                             n_best_size = 5,
                             threshold = -1.5,
                             max_answer_length = 48,
                             debug = False,
                             wrong_only = False):
  global prelim_predictions
  use_ans_class = True
  all_predictions = {}
  all_predictions_output = {}
  scores_diff_json = {}
  score = 0
  for qid, predictions in tqdm(prediction_by_qid.items()):
    q = orig_data[qid]
    ri = 0
    prelim_predictions = []
    for result, r in predictions:
      paragraph_text = r.original_text
      original_s, original_e = r.original_text_span # exclusive
      this_paragraph_text = paragraph_text[original_s:original_e]
      cur_null_score = -1e6
      sub_prelim_predictions = []
      if use_ans_class:
        start_top_log_probs, end_top_log_probs, cls_logits = result
        cur_null_score = cls_logits.tolist()
      else:
        start_top_log_probs, end_top_log_probs = result
      if True:
        start_top_log_probs = start_top_log_probs.cpu().detach().numpy()
        end_top_log_probs = end_top_log_probs.cpu().detach().numpy()
        start_top_index = start_top_log_probs.argsort()[-start_n_top:][::-1].tolist()
        end_top_index = end_top_log_probs.argsort()[-end_n_top:][::-1].tolist()
        start_top_log_probs = start_top_log_probs.tolist()
        end_top_log_probs = end_top_log_probs.tolist()
        for start_index in start_top_index:
            for end_index in end_top_index:
              if start_index == 0 or end_index == 0:
                continue
              if end_index < start_index:
                continue
              if start_index >= len(r.segments) or end_index >= len(r.segments):
                continue
              seg_s = r.segments[start_index]
              seg_e = r.segments[end_index]
              if seg_s != seg_e:
                continue
              if r.is_max_context[start_index] == 0 :
                continue
              length = end_index - start_index + 1
              if length > max_answer_length:
                continue
              start_log_prob = start_top_log_probs[start_index]
              end_log_prob = end_top_log_probs[end_index]
              sub_prelim_predictions.append(
                  _PrelimPrediction(
                      feature_index=ri,
                      start_index=start_index,
                      end_index=end_index,
                      start_log_prob=start_log_prob,
                      end_log_prob=end_log_prob,
                      this_paragraph_text=this_paragraph_text,
                      cur_null_score=cur_null_score
                  ))
      prelim_predictions.extend(sub_prelim_predictions)
      ri += 1
    prelim_predictions = sorted(
        prelim_predictions,
        key=(lambda x: (x.start_log_prob + x.end_log_prob)),
        reverse=True)
    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
          break
      r = predictions[pred.feature_index][1]
      cur_null_score = pred.cur_null_score
      this_paragraph_text = pred.this_paragraph_text
      s,e = pred.start_index, pred.end_index  # e is inclusive
      char_s  = r.tok_to_char_offset[s]
      char_e  = r.tok_to_char_offset[e]  # inclusive
      char_e += len(r.all_text_tokens[r.char_to_tok_offset[char_e]])
      final_text = r.text[char_s:char_e].strip() # this_paragraph_text[char_s:char_e]
      if False:
        print(final_text, '>>', r.all_text_tokens[s:e+1])
      if final_text in seen_predictions:
          continue
      seen_predictions[final_text] = True
      nbest.append(
        _NbestPrediction(
            text=final_text,
            start_log_prob=pred.start_log_prob,
            end_log_prob=pred.end_log_prob,
            cur_null_score=cur_null_score))
    if len(nbest) == 0:
        nbest.append(
          _NbestPrediction(text="", start_log_prob=-1e6,
          end_log_prob=-1e6,
          cur_null_score=-1e6))
    total_scores = []
    best_non_null_entry = None
    best_null_score = None
    best_score_no_ans = None
    for entry in nbest:
      total_scores.append(entry.start_log_prob + entry.end_log_prob)
      if not best_non_null_entry:
        best_non_null_entry = entry
        best_null_score = entry.cur_null_score
        best_score_no_ans = entry.cur_null_score
    probs = _compute_softmax(total_scores)
    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_log_prob"] = entry.start_log_prob
      output["end_log_prob"] = entry.end_log_prob
      nbest_json.append(output)
    s = compute_f1(normalize_answer(q['answer_text']), normalize_answer(best_non_null_entry.text) if best_null_score < threshold else '')
    all_predictions_output[qid] = [q['answer_text'], best_non_null_entry.text, best_null_score, s]
    if debug:
      ans = normalize_answer(best_non_null_entry.text) if best_null_score < threshold else '*No answer*'
      truth = normalize_answer(q['answer_text']) or '*No answer*'
      if (not wrong_only or ans != truth):
        print('Q:', q['question'])
        print('A:', ans, '(',best_null_score,')',  '[',best_score_no_ans,']', )
        print('Truth:', truth)
        print('')
      score += s
    assert len(nbest_json) >= 1
    assert best_non_null_entry is not None
    all_predictions[qid] = best_non_null_entry.text
    scores_diff_json[qid] = best_null_score
  print('score: ', score, '/', len(all_predictions), '=', score / len(all_predictions))
  return nbest_json, all_predictions, scores_diff_json, all_predictions_output

            
            
            
            
            

from fairseq.models.roberta.model_span_qa import RobertaQAModel

roberta_single = RobertaQAModel.from_pretrained(
    args.tokenizer_dir, 
    checkpoint_file=model_file, 
    strict=True).model



num_cores = torch.cuda.device_count() # 8
batch_size = 128                      # 16 per device

fp16 = True

            
          
if num_cores > 1:
    roberta = nn.DataParallel(roberta_single)
  
            
            
print("Using ", num_cores, "GPUs!")
            
            
use_gpu = torch.cuda.is_available()


device = torch.device("cuda:0" if use_gpu else "cpu")

if not use_gpu:
  fp16 = False

            
roberta.to(device)
            
if fp16:
  roberta.half()
  
roberta.eval()
  
  
            
        
  

for fn in glob(test_files):
    print(fn)
    with open(fn) as f:
        j = json.load(f)

    orig_data = {} 
        
    records = []
    all_rs = []
    ps = [p for e in j['data'] for p in e['paragraphs']]
    for p in tqdm(ps):
        for q in p['qas']:
            orig_data[q['id']] = q
        unique_index += 1
        context = p['context']
        qas = p['qas']
        rss = tokenizer.merge_cq(context, 
                                 qas,
                                 max_seq_length = max_seq_length,
                                 max_query_length=max_query_length,
                                 doc_stride = doc_stride,
                                 unique_index=unique_index,
                                 is_training=is_training
                               )

        for q, rs in zip(qas, rss):
            ty = lang+'-'+q['type']
            if ty not in bucket:
                bucket[ty] = []
            if ty+'-neg' not in bucket:
                bucket[ty+'-neg'] = []
            for r in rs:
                
                inp = tk.convert_tokens_to_ids(r.all_doc_tokens)
                start_position,end_position = char_anchors_to_tok_pos(r)
                p_mask = r.p_mask
                uid = r.unique_index[0]*1000 + r.unique_index[1]

                no_ans = start_position == 0

                
                    
                assert start_position >= 0 and end_position >= 0 and start_position < len(inp) and end_position < len(inp)
                assert len(inp) <= max_seq_length

                S, E = start_position, end_position
                
                record = marshal.dumps(
                    (
                    uid,
                    np.array(inp,dtype=np.uint32).tobytes(),
                    start_position,
                    end_position,
                    int(no_ans)
                    )
                )
                
                
                records.append(record)


    batches = list(zip(from_records(records,batch_size, half=fp16, shuffle=False), chunks(all_rs,batch_size)))
    prediction_by_qid = {}

    with torch.no_grad():
        for e, rs in tqdm(batches):
            inp, start, end, _ = e
            (start_logits, end_logits, cls_logits), _ = roberta(inp.to(device=device))
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            for result, r in zip(zip(*(start_logits, end_logits, cls_logits)), rs):
                qid = r.qid
                if qid not in prediction_by_qid:
                    prediction_by_qid[qid] = []
                prediction_by_qid[qid].append((result, r))

    nbest_json, all_predictions, scores_diff_json, all_predictions_output = \
        handle_prediction_by_qid(
            roberta_single, 
            prediction_by_qid, 
            qthreshold=-1, 
            debug=False, 
            wrong_only=True)
  
