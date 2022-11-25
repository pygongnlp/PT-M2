from tqdm import tqdm
import numpy as np
import sys

sys.path.append("m2score")
from m2score.m2scorer import load_annotation
from m2score.util import smart_open
from m2score.levenshtein import batch_multi_pre_rec_f1, batch_multi_pre_rec_f1_sent
from errant_score import batch_multi_pre_rec_f1_errant, batch_multi_pre_rec_f1_sent_errant, errant_load_annotation
from bart_score import BARTScorer
from bert_score import BERTScorer


class PTM2:
    def __init__(self, args, corpus=None):
        self.args = args
        self.beta = args.beta
        self.device = args.device
        self.model_type = args.model_type

        self.corpus = corpus
        self.scorer = self.get_plm_scorer(corpus)

    def compute_sentm2(self, m2_file, hyp_file, sources, references):
        _, gold_edits = load_annotation(m2_file)
        fin = smart_open(hyp_file, 'r')
        system_sentences = [line.strip() for line in fin.readlines()]
        fin.close()

        score_lst = []
        for hyp, src, refs, golds in tqdm(zip(system_sentences, sources, references, gold_edits)):
            f1 = batch_multi_pre_rec_f1_sent(candidates=[hyp], sources=[src], gold_edits=[golds],
                                             references=[refs], scorer=self.scorer, scorer_type=self.args.scorer, beta=self.beta)[-1]
            score_lst.append(f1)

        return sum(np.array(score_lst)) / len(system_sentences)

    def compute_m2(self, m2_file, hyp_file, sources, references):
        _, gold_edits = load_annotation(m2_file)
        fin = smart_open(hyp_file, 'r')
        system_sentences = [line.strip() for line in fin.readlines()]
        fin.close()

        score = batch_multi_pre_rec_f1(candidates=system_sentences, sources=sources, gold_edits=gold_edits,
                                       references=references, scorer=self.scorer, scorer_type=self.args.scorer, beta=self.beta)[-1]
        return score

    def compute_senterrant(self, m2_file, hyp_file, sources, references):
        sys_file = f"{hyp_file}.m2"
        _, gold_edits, sys_edits = errant_load_annotation(sys_file, m2_file)

        fin = smart_open(hyp_file, 'r')
        system_sentences = [line.strip() for line in fin.readlines()]
        fin.close()

        score_lst = []
        for hyp, src, refs, sys, golds in tqdm(
                zip(system_sentences, sources, references, sys_edits, gold_edits)):
            f1 = batch_multi_pre_rec_f1_sent_errant(candidates=[hyp], sources=[src], system_edits=[sys], gold_edits=[golds],
                                                    references=[refs], scorer=self.scorer, scorer_type=self.args.scorer, beta=self.beta)[-1]
            score_lst.append(f1)

        return sum(np.array(score_lst)) / len(system_sentences)

    def compute_errant(self, m2_file, hyp_file, sources, references):
        sys_file = f"{hyp_file}.m2"
        _, gold_edits, sys_edits = errant_load_annotation(sys_file, m2_file)

        fin = smart_open(hyp_file, 'r')
        system_sentences = [line.strip() for line in fin.readlines()]
        fin.close()

        score = \
            batch_multi_pre_rec_f1_errant(candidates=system_sentences, sources=sources, system_edits=sys_edits, gold_edits=gold_edits,
                                          references=references, scorer=self.scorer, scorer_type=self.args.scorer, beta=self.beta)[-1]
        return score

    def get_plm_scorer(self, corpus=None):
        scorer = None
        if self.args.scorer == "bertscore":
            if corpus:
                scorer = BERTScorer(device=self.device, model_type=self.model_type,
                                    lang="en", rescale_with_baseline=True,
                                    idf=True, idf_sents=corpus)
            else:
                scorer = BERTScorer(device=self.device, model_type=self.model_type,
                                    lang="en", rescale_with_baseline=True)
        elif self.args.scorer == "bartscore":
            scorer = BARTScorer(device=self.device, checkpoint=f"facebook/{self.model_type}")
        return scorer