from copy import deepcopy
import math
from tqdm import tqdm

def get_ref(edits, src):
    cnt = 0
    src = src.split()
    e_s = src
    for edit in edits:
        s_idx, e_idx, rep_tok = edit
        s_idx = cnt + s_idx
        e_idx = cnt + e_idx
        e_s = e_s[:s_idx] + rep_tok.split() + e_s[e_idx:] if rep_tok else e_s[:s_idx] + e_s[e_idx:]
        cnt += len(rep_tok.split()) - (e_idx - s_idx)
    return " ".join(e_s)


def compute_weight_edits(editSeq, gold, source, cand, ref, w_t, scorer=None, sent_level=False):
    weight_edits = {}
    editSeq = sorted(editSeq, key=lambda x: (x[0], x[1]))
    assert cand == get_ref(editSeq, source), f"src: {source}\nref: {cand}\nref_s: {get_ref(editSeq, source)}\nedits: {editSeq}"
    gold = sorted(gold, key=lambda x: (x[0], x[1]))
    assert ref == get_ref(gold, source), f"src: {source}\nref: {ref}\nref_s: {get_ref(gold, source)}\nedits: {gold}"
    edits = list(set(editSeq) | set(gold))
    edits = sorted(edits, key=lambda x: (x[0], x[1]))

    for i, edit in enumerate(edits):
        edit_s = [edit]
        edit_s = sorted(edit_s, key=lambda x: (x[0], x[1]))
        ref_s = get_ref(edit_s, source)

        if w_t == "self":
            weight_edits[edit] = 1
        elif w_t == "bartscore":
            s1, s2 = scorer.score([ref, ref], [source, ref_s], batch_size=2)
            weight_edits[edit] = abs(s1 - s2)
        elif w_t == "bertscore":
            s1 = scorer.score([source], [ref])[-1]
            s1 = s1[0].item()
            s2 = scorer.score([ref_s], [ref])[-1]
            s2 = s2[0].item()
            weight_edits[edit] = abs(s1 - s2)
    if sent_level:
        w_sum = sum(v for v in weight_edits.values())
        if w_sum == 0:
            weight_edits = {k: 1 / len(weight_edits) for k in weight_edits.keys()}
    return weight_edits


def errant_batch_pre_rec_f1(editSeq, gold, source, candidate, ref, scorer, args, beta=0.5):
    correct = matchSeq(editSeq, gold, ignore_whitespace_casing=False)
    #print(f"correct {correct}  sys_edit {editSeq}  gold_edit {gold}")
    weight_editSeq = compute_weight_edits(editSeq, source, candidate, args.scorer, args.direction, scorer)
    weight_gold = compute_weight_edits(gold, source, ref, args.scorer, args.direction, scorer)
    if not editSeq:
        p = 1.0
    else:
        p = sum(weight_editSeq[c] for c in correct)
    if not gold:
        r = 1.0
    else:
        r = sum(weight_gold[c] for c in correct)
    if not beta * beta * p + r:
        f1 = 0.0
    else:
        f1 = (1.0 + beta * beta) * p * r / (beta * beta * p + r)

    return (p, r, f1)


def matchSeq(editSeq, gold_edits, ignore_whitespace_casing=False, verbose=False):
    m = []
    goldSeq = deepcopy(gold_edits)
    last_index = 0
    CInsCDel = False
    CInsWDel = False
    CDelWIns = False
    for e in editSeq:
        # print(e)
        # print("====")
        for i in range(last_index, len(goldSeq)):
            g = goldSeq[i]
            # print(g)
            if matchEdit(e, g, ignore_whitespace_casing):
                # print(f"* {e}")
                m.append(e)
                last_index = i + 1
    return m



def matchEdit(e, g, ignore_whitespace_casing=False):
    # start offset
    if e[0] != g[0]:
        return False
    # end offset
    if e[1] != g[1]:
        return False
    # original string
    if e[2] != g[2]:
        return False
    return True

# if __name__ == "__main__":
#     print(matchSeq([(3, 3, ','), (19, 19, ','), (21, 22, '')], [(3, 3, ','), (7, 8, 'testing'), (19, 19, ',')]
# ))

def errant_load_annotation(hyp_m2, ref_m2):
    hyp_m2 = open(hyp_m2, encoding="utf8").read().strip().split("\n\n")
    ref_m2 = open(ref_m2, encoding="utf8").read().strip().split("\n\n")
    assert len(hyp_m2) == len(ref_m2)

    sources, gold_edits, sys_edits = [], [], []
    for sent_id, sent in enumerate(zip(hyp_m2, ref_m2)):
        # Simplify the edits into lists of lists
        hyp_edits = simplify_edits(sent[0])
        ref_edits = simplify_edits(sent[1])
        # Process the edits for detection/correction based on args
        hyp_dict = process_edits(hyp_edits)
        ref_dict = process_edits(ref_edits)
        hyp_dict = [k for v in hyp_dict.values() for k in v.keys() if k != (-1, -1, '-NONE-')]
        ref_dict = {key: [k for k in value.keys() if k != (-1, -1, '-NONE-')] for key, value in ref_dict.items()}
        # original sentence for logging
        original_sentence = sent[0][2:].split("\nA")[0]
        sources.append(original_sentence)
        gold_edits.append(ref_dict)
        sys_edits.append(hyp_dict)

    return sources, gold_edits, sys_edits



def simplify_edits(sent):
    out_edits = []
    # Get the edit lines from an m2 block.
    edits = sent.split("\n")[1:]
    # Loop through the edits
    for edit in edits:
        # Preprocessing
        edit = edit[2:].split("|||") # Ignore "A " then split.
        span = edit[0].split()
        start = int(span[0])
        end = int(span[1])
        cat = edit[1]
        cor = edit[2]
        coder = int(edit[-1])
        out_edit = [start, end, cat, cor, coder]
        out_edits.append(out_edit)
    return out_edits


def process_edits(edits, dt=False, ds=False, single=False, filt=None, multi=False, cse=False):
    if filt is None:
        filt = []
    coder_dict = {}
    # Add an explicit noop edit if there are no edits.
    if not edits: edits = [[-1, -1, "noop", "-NONE-", 0]]
    # Loop through the edits
    for edit in edits:
        # Name the edit elements for clarity
        start = edit[0]
        end = edit[1]
        cat = edit[2]
        cor = edit[3]
        coder = edit[4]
        # Add the coder to the coder_dict if necessary
        if coder not in coder_dict: coder_dict[coder] = {}

        # Optionally apply filters based on args
        # 1. UNK type edits are only useful for detection, not correction.
        if not dt and not ds and cat == "UNK": continue
        # 2. Only evaluate single token edits; i.e. 0:1, 1:0 or 1:1
        if single and (end-start >= 2 or len(cor.split()) >= 2): continue
        # 3. Only evaluate multi token edits; i.e. 2+:n or n:2+
        if multi and end-start < 2 and len(cor.split()) < 2: continue
        # 4. If there is a filter, ignore the specified error types
        if filt and cat in filt: continue

        # Token Based Detection
        if dt:
            # Preserve noop edits.
            if start == -1:
                if (start, start) in coder_dict[coder].keys():
                    coder_dict[coder][(start, start)].append(cat)
                else:
                    coder_dict[coder][(start, start)] = [cat]
            # Insertions defined as affecting the token on the right
            elif start == end and start >= 0:
                if (start, start+1) in coder_dict[coder].keys():
                    coder_dict[coder][(start, start+1)].append(cat)
                else:
                    coder_dict[coder][(start, start+1)] = [cat]
            # Edit spans are split for each token in the range.
            else:
                for tok_id in range(start, end):
                    if (tok_id, tok_id+1) in coder_dict[coder].keys():
                        coder_dict[coder][(tok_id, tok_id+1)].append(cat)
                    else:
                        coder_dict[coder][(tok_id, tok_id+1)] = [cat]

        # Span Based Detection
        elif ds:
            if (start, end) in coder_dict[coder].keys():
                coder_dict[coder][(start, end)].append(cat)
            else:
                coder_dict[coder][(start, end)] = [cat]

        # Span Based Correction
        else:
            # With error type classification
            if cse:
                if (start, end, cat, cor) in coder_dict[coder].keys():
                    coder_dict[coder][(start, end, cat, cor)].append(cat)
                else:
                    coder_dict[coder][(start, end, cat, cor)] = [cat]
            # Without error type classification
            else:
                if (start, end, cor) in coder_dict[coder].keys():
                    coder_dict[coder][(start, end, cor)].append(cat)
                else:
                    coder_dict[coder][(start, end, cor)] = [cat]
    return coder_dict


def evaluate_edits(hyp_dict, ref_dict, best, sent_id, original_sentence, beta=0.5, verbose=False):
    # Verbose output: display the original sentence
    if verbose:
        print('{:-^40}'.format(""))
        print("Original sentence " + str(sent_id) + ": " + original_sentence)
    # Store the best sentence level scores and hyp+ref combination IDs
    # best_f is initialised as -1 cause 0 is a valid result.
    best_tp, best_fp, best_fn, best_f, best_hyp, best_ref = 0, 0, 0, -1, 0, 0
    best_cat = {}
    # Compare each hyp and ref combination
    for hyp_id in hyp_dict.keys():
        for ref_id in ref_dict.keys():
            # Get the local counts for the current combination.
            tp, fp, fn, cat_dict = compareEdits(hyp_dict[hyp_id], ref_dict[ref_id])
            # Compute the local sentence scores (for verbose output only)
            loc_p, loc_r, loc_f = computeFScore(tp, fp, fn, beta)
            # Compute the global sentence scores
            p, r, f = computeFScore(
                tp+best["tp"], fp+best["fp"], fn+best["fn"], beta)
            # Save the scores if they are better in terms of:
            # 1. Higher F-score
            # 2. Same F-score, higher TP
            # 3. Same F-score and TP, lower FP
            # 4. Same F-score, TP and FP, lower FN
            if (f > best_f) or \
                (f == best_f and tp > best_tp) or \
                (f == best_f and tp == best_tp and fp < best_fp) or \
                (f == best_f and tp == best_tp and fp == best_fp and fn < best_fn):
                best_tp, best_fp, best_fn = tp, fp, fn
                best_f, best_hyp, best_ref = f, hyp_id, ref_id
                best_cat = cat_dict
            # Verbose output
            if verbose:
                # Prepare verbose output edits.
                hyp_verb = list(sorted(hyp_dict[hyp_id].keys()))
                ref_verb = list(sorted(ref_dict[ref_id].keys()))
                # add categories
                # hyp_dict[hyp_id] looks like (0, 1, "str")
                # hyp_dict[hyp_id][h] is a list, always length one, of the corresponding category
                hyp_verb = [h + (hyp_dict[hyp_id][h][0],) for h in hyp_verb]
                ref_verb = [r + (ref_dict[ref_id][r][0],) for r in ref_verb]
                # Ignore noop edits
                if not hyp_verb or hyp_verb[0][0] == -1: hyp_verb = []
                if not ref_verb or ref_verb[0][0] == -1: ref_verb = []
                # Print verbose info
                print('{:-^40}'.format(""))
                print("SENTENCE "+str(sent_id)+" - HYP "+str(hyp_id)+" - REF "+str(ref_id))
                print("HYPOTHESIS EDITS :", hyp_verb)
                print("REFERENCE EDITS  :", ref_verb)
                print("Local TP/FP/FN   :", str(tp), str(fp), str(fn))
                print("Local P/R/F"+str(beta)+"  :", str(loc_p), str(loc_r), str(loc_f))
                print("Global TP/FP/FN  :", str(tp+best["tp"]), str(fp+best["fp"]), str(fn+best["fn"]))
                print("Global P/R/F"+str(beta)+"  :", str(p), str(r), str(f))
    # Verbose output: display the best hyp+ref combination
    if verbose:
        print('{:-^40}'.format(""))
        print("^^ HYP "+str(best_hyp)+", REF "+str(best_ref)+" chosen for sentence "+str(sent_id))
        print("Local results:")
        header = ["Category", "TP", "FP", "FN"]
        body = [[k, *v] for k, v in best_cat.items()]
        print_table([header] + body)
    # Save the best TP, FP and FNs as a dict, and return this and the best_cat dict
    best_dict = {"tp":best_tp, "fp":best_fp, "fn":best_fn}
    return best_dict, best_cat


def compareEdits(hyp_edits, ref_edits):
    tp = 0    # True Positives
    fp = 0    # False Positives
    fn = 0    # False Negatives
    cat_dict = {} # {cat: [tp, fp, fn], ...}

    for h_edit, h_cats in hyp_edits.items():
        # noop hyp edits cannot be TP or FP
        if h_cats[0] == "noop": continue
        # TRUE POSITIVES
        if h_edit in ref_edits.keys():
            # On occasion, multiple tokens at same span.
            for h_cat in ref_edits[h_edit]: # Use ref dict for TP
                tp += 1
                # Each dict value [TP, FP, FN]
                if h_cat in cat_dict.keys():
                    cat_dict[h_cat][0] += 1
                else:
                    cat_dict[h_cat] = [1, 0, 0]
        # FALSE POSITIVES
        else:
            # On occasion, multiple tokens at same span.
            for h_cat in h_cats:
                fp += 1
                # Each dict value [TP, FP, FN]
                if h_cat in cat_dict.keys():
                    cat_dict[h_cat][1] += 1
                else:
                    cat_dict[h_cat] = [0, 1, 0]
    for r_edit, r_cats in ref_edits.items():
        # noop ref edits cannot be FN
        if r_cats[0] == "noop": continue
        # FALSE NEGATIVES
        if r_edit not in hyp_edits.keys():
            # On occasion, multiple tokens at same span.
            for r_cat in r_cats:
                fn += 1
                # Each dict value [TP, FP, FN]
                if r_cat in cat_dict.keys():
                    cat_dict[r_cat][2] += 1
                else:
                    cat_dict[r_cat] = [0, 0, 1]
    return tp, fp, fn, cat_dict


def comp_p(a, b):
    if b:
        p = a / b
    else:
        p = 1.0
    return p

def comp_r(c, g):
    if g:
        r = c / g
    else:
        r = 1.0
    return r

def comp_f1(p, r, beta):
    if beta*beta*p+r:
        f = (1.0+beta*beta) * p * r / (beta*beta*p+r)
    else:
        f = 0.0
    return f


def print_table(table):
    longest_cols = [
        (max([len(str(row[i])) for row in table]) + 3)
        for i in range(len(table[0]))
    ]
    row_format = "".join(["{:>" + str(longest_col) + "}" for longest_col in longest_cols])
    for row in table:
        print(row_format.format(*row))


def computeFScore(tp, fp, fn, beta):
    p = float(tp)/(tp+fp) if fp else 1.0
    r = float(tp)/(tp+fn) if fn else 1.0
    f = float((1+(beta**2))*p*r)/(((beta**2)*p)+r) if p+r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def print_results(best, dt=False, ds=False, cse=False, cat=None, best_cats=None, beta=0.5):
    # Prepare output title.
    if dt: title = " Token-Based Detection "
    elif ds: title = " Span-Based Detection "
    elif cse: title = " Span-Based Correction + Classification "
    else: title = " Span-Based Correction "

    # Category Scores
    if cat:
        best_cats = processCategories(best_cats, cat)
        print("")
        print('{:=^66}'.format(title))
        print("Category".ljust(14), "TP".ljust(8), "FP".ljust(8), "FN".ljust(8),
            "P".ljust(8), "R".ljust(8), "F"+str(beta))
        for cat, cnts in sorted(best_cats.items()):
            cat_p, cat_r, cat_f = computeFScore(cnts[0], cnts[1], cnts[2], beta)
            print(cat.ljust(14), str(cnts[0]).ljust(8), str(cnts[1]).ljust(8),
                str(cnts[2]).ljust(8), str(cat_p).ljust(8), str(cat_r).ljust(8), cat_f)

    return list(computeFScore(best["tp"], best["fp"], best["fn"], beta))


def processCategories(cat_dict, setting):
    # Otherwise, do some processing.
    proc_cat_dict = {}
    for cat, cnt in cat_dict.items():
        if cat == "UNK":
            proc_cat_dict[cat] = cnt
            continue
        # M, U, R or UNK combined only.
        if setting == 1:
            if cat[0] in proc_cat_dict.keys():
                proc_cat_dict[cat[0]] = [x+y for x, y in zip(proc_cat_dict[cat[0]], cnt)]
            else:
                proc_cat_dict[cat[0]] = cnt
        # Everything without M, U or R.
        elif setting == 2:
            if cat[2:] in proc_cat_dict.keys():
                proc_cat_dict[cat[2:]] = [x+y for x, y in zip(proc_cat_dict[cat[2:]], cnt)]
            else:
                proc_cat_dict[cat[2:]] = cnt
        # All error category combinations
        else:
            return cat_dict
    return proc_cat_dict


def batch_multi_pre_rec_f1_errant(candidates, sources, system_edits, gold_edits, references, scorer, scorer_type,
                           max_unchanged_words=2, beta=0.5, ignore_whitespace_casing=False, verbose=False,
                           very_verbose=False):
    assert len(candidates) == len(sources) == len(gold_edits)
    stat_correct = 0.0
    stat_proposed = 0.0
    stat_gold = 0.0
    i = 0
    for candidate, source, refs, sys_set, golds_set in tqdm(zip(candidates, sources, references,
                                                                system_edits, gold_edits)):
        i = i + 1
        # Find measures maximizing current cumulative F1; local: curent annotator only
        sqbeta = beta * beta
        chosen_ann = -1
        f1_max = -math.inf

        argmax_correct = 0.0
        argmax_proposed = 0.0
        argmax_gold = 0.0
        max_stat_correct = -math.inf
        min_stat_proposed = math.inf
        min_stat_gold = math.inf
        for annotator, gold in golds_set.items():
            editSeq = sys_set
            correct = matchSeq(editSeq, gold, ignore_whitespace_casing, verbose)
            #gold = [(g[0], g[1], g[2], g[-1][0]) for g in gold]
            weight_edits = compute_weight_edits(editSeq, gold, source, candidate, refs[annotator], scorer_type, scorer)
            # local cumulative counts, P, R and F1
            stat_correct_local = stat_correct + sum(weight_edits[c] for c in correct)
            stat_proposed_local = stat_proposed + sum(weight_edits[e] for e in editSeq)
            stat_gold_local = stat_gold + sum(weight_edits[g] for g in gold)
            p_local = comp_p(stat_correct_local, stat_proposed_local)
            r_local = comp_r(stat_correct_local, stat_gold_local)
            f1_local = comp_f1(p_local, r_local, beta)

            if f1_max < f1_local or \
                    (f1_max == f1_local and max_stat_correct < stat_correct_local) or \
                    (
                            f1_max == f1_local and max_stat_correct == stat_correct_local and min_stat_proposed + sqbeta * min_stat_gold > stat_proposed_local + sqbeta * stat_gold_local):
                chosen_ann = annotator
                f1_max = f1_local
                max_stat_correct = stat_correct_local
                min_stat_proposed = stat_proposed_local
                min_stat_gold = stat_gold_local
                argmax_correct = sum(weight_edits[c] for c in correct)
                argmax_proposed = sum(weight_edits[e] for e in editSeq)
                argmax_gold = sum(weight_edits[g] for g in gold)

        if verbose:
            print(">> Chosen Annotator for line", i, ":", chosen_ann)
            print("")
        stat_correct += argmax_correct
        stat_proposed += argmax_proposed
        stat_gold += argmax_gold

    if stat_proposed:
        p = stat_correct / stat_proposed
    else:
        p = 1.0
    if stat_gold:
        r = stat_correct / stat_gold
    else:
        r = 1.0
    if beta * beta * p + r:
        f1 = (1.0 + beta * beta) * p * r / (beta * beta * p + r)
    else:
        f1 = 0.0

    if verbose:
        print("CORRECT EDITS  :", int(stat_correct))
        print("PROPOSED EDITS :", int(stat_proposed))
        print("GOLD EDITS     :", int(stat_gold))
        print("P =", p)
        print("R =", r)
        print("F_%.1f =" % beta, f1)
    return (p, r, f1)


def batch_multi_pre_rec_f1_sent_errant(candidates, sources, system_edits, gold_edits, references, scorer, scorer_type,
                           max_unchanged_words=2, beta=0.5, ignore_whitespace_casing=False, verbose=False,
                           very_verbose=False):
    assert len(candidates) == len(sources) == len(gold_edits) == len(system_edits)
    stat_correct = 0.0
    stat_proposed = 0.0
    stat_gold = 0.0
    i = 0
    for candidate, source, refs, editSeq, golds_set in zip(candidates, sources, references,
                                                                system_edits, gold_edits):
        i = i + 1
        # Find measures maximizing current cumulative F1; local: curent annotator only
        sqbeta = beta * beta
        chosen_ann = -1
        f1_max = -math.inf

        argmax_correct = 0.0
        argmax_proposed = 0.0
        argmax_gold = 0.0
        max_stat_correct = -math.inf
        min_stat_proposed = math.inf
        min_stat_gold = math.inf
        for annotator, gold in golds_set.items():
            correct = matchSeq(editSeq, gold, ignore_whitespace_casing, verbose)
            #gold = [(g[0], g[1], g[2], g[-1][0]) for g in gold]
            weight_edits = compute_weight_edits(editSeq, gold, source, candidate, refs[annotator], scorer_type, scorer, sent_level=True)
            # local cumulative counts, P, R and F1
            stat_correct_local = stat_correct + sum(weight_edits[c] for c in correct)
            stat_proposed_local = stat_proposed + sum(weight_edits[e] for e in editSeq)
            stat_gold_local = stat_gold + sum(weight_edits[g] for g in gold)
            p_local = comp_p(stat_correct_local, stat_proposed_local)
            r_local = comp_r(stat_correct_local, stat_gold_local)
            f1_local = comp_f1(p_local, r_local, beta)

            if f1_max < f1_local or \
                    (f1_max == f1_local and max_stat_correct < stat_correct_local) or \
                    (
                            f1_max == f1_local and max_stat_correct == stat_correct_local and min_stat_proposed + sqbeta * min_stat_gold > stat_proposed_local + sqbeta * stat_gold_local):
                chosen_ann = annotator
                f1_max = f1_local
                max_stat_correct = stat_correct_local
                min_stat_proposed = stat_proposed_local
                min_stat_gold = stat_gold_local
                argmax_correct = sum(weight_edits[c] for c in correct)
                argmax_proposed = sum(weight_edits[e] for e in editSeq)
                argmax_gold = sum(weight_edits[g] for g in gold)

        if verbose:
            print(">> Chosen Annotator for line", i, ":", chosen_ann)
            print("")
        stat_correct += argmax_correct
        stat_proposed += argmax_proposed
        stat_gold += argmax_gold

    if stat_proposed:
        p = stat_correct / stat_proposed
    else:
        p = 1.0
    if stat_gold:
        r = stat_correct / stat_gold
    else:
        r = 1.0
    if beta * beta * p + r:
        f1 = (1.0 + beta * beta) * p * r / (beta * beta * p + r)
    else:
        f1 = 0.0

    if verbose:
        print("CORRECT EDITS  :", int(stat_correct))
        print("PROPOSED EDITS :", int(stat_proposed))
        print("GOLD EDITS     :", int(stat_gold))
        print("P =", p)
        print("R =", r)
        print("F_%.1f =" % beta, f1)
    return (p, r, f1)