

import operator
import random
import math
import re

def smart_open(fname, mode = 'r'):
    if fname.endswith('.gz'):
        import gzip
        # Using max compression (9) by default seems to be slow.                                
        # Let's try using the fastest.                                                          
        return gzip.open(fname, mode, 1)
    else:
        return open(fname, mode, encoding="utf8")


def randint(b, a=0):
    return random.randint(a,b)

def uniq(seq, idfun=None):
    # order preserving                                                                          
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:                                                               
        # if seen.has_key(marker)                                                               
        # but in new ones:                                                                      
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result


def get_ref(edits, src):
    cnt = 0
    src = src.split()
    e_s = src

    for i in range(len(edits)):
        s_idx, e_idx, oral_tok, rep_tok = edits[i]
        if oral_tok == "":
            e_idx = s_idx
        s_idx = cnt + s_idx
        e_idx = cnt + e_idx
        e_s = e_s[:s_idx] + rep_tok.split() + e_s[e_idx:] if rep_tok else e_s[:s_idx] + e_s[e_idx:]
        cnt += len(rep_tok.split()) - len(oral_tok.split())
    return " ".join(e_s)


def compute_weight_edits(editSeq, gold, source, cand, ref, w_t, scorer=None, sent_level=False):
    weight_edits, filters = {}, {}
    editSeq = sorted(editSeq, key=lambda x: (x[0], x[1]))
    assert cand == get_ref(editSeq, source), f"src: {source}\nref: {cand}\nref_s: {get_ref(editSeq, source)}\nedits: {editSeq}"
    gold = sorted(gold, key=lambda x: (x[0], x[1]))
    assert ref == get_ref(gold, source), f"src: {source}\nref: {ref}\nref_s: {get_ref(gold, source)}\nedits: {gold}"
    edits = list(set(editSeq) | set(gold))
    edits = sorted(edits, key=lambda x: (x[0], x[1]))

    for i, edit in enumerate(edits):
        edit_s = [edit]
        ref_s = get_ref(edit_s, source)

        if w_t == "self":
            weight_edits[edit] = 1
        elif w_t == "bartscore":
            s1, s2 = scorer.score([ref, ref], [ref_s, source], batch_size=2)
            weight_edits[edit] = abs(s1 - s2)
        elif w_t == "bertscore":
            s1 = scorer.score([ref_s], [ref])[-1]
            s1 = s1[0].item()
            s2 = scorer.score([source], [ref])[-1]
            s2 = s2[0].item()
            weight_edits[edit] = abs(s1 - s2)
    if sent_level:
        w_sum = sum(v for v in weight_edits.values())
        if w_sum == 0:
            weight_edits = {k: 1 / len(weight_edits) for k in weight_edits.keys()}
    return weight_edits


def sort_dict(myDict, byValue=False, reverse=False):
    if byValue:
        items = myDict.items()
        items.sort(key = operator.itemgetter(1), reverse=reverse)
    else:
        items = sorted(myDict.items())
    return items

def max_dict(myDict, byValue=False):
    if byValue:
        skey=lambda x:x[1]
    else:
        skey=lambda x:x[0]
    return max(myDict.items(), key=skey)


def min_dict(myDict, byValue=False):
    if byValue:
        skey=lambda x:x[1]
    else:
        skey=lambda x:x[0]
    return min(myDict.items(), key=skey)

def paragraphs(lines, is_separator=lambda x : x == '\n', joiner=''.join):
    paragraph = []
    for line in lines:
        if is_separator(line):
            if paragraph:
                yield joiner(paragraph)
                paragraph = []
        else:
            paragraph.append(line)
    if paragraph:
        yield joiner(paragraph)


def isASCII(word):
    try:
        word = word.decode("ascii")
        return True
    except UnicodeEncodeError :
        return False
    except UnicodeDecodeError:
        return False


def intersect(x, y):
    return [z for z in x if z in y]



# Mapping Windows CP1252 Gremlins to Unicode
# from http://effbot.org/zone/unicode-gremlins.htm
cp1252 = {
    # from http://www.microsoft.com/typography/unicode/1252.htm
    u"\x80": u"\u20AC", # EURO SIGN
    u"\x82": u"\u201A", # SINGLE LOW-9 QUOTATION MARK
    u"\x83": u"\u0192", # LATIN SMALL LETTER F WITH HOOK
    u"\x84": u"\u201E", # DOUBLE LOW-9 QUOTATION MARK
    u"\x85": u"\u2026", # HORIZONTAL ELLIPSIS
    u"\x86": u"\u2020", # DAGGER
    u"\x87": u"\u2021", # DOUBLE DAGGER
    u"\x88": u"\u02C6", # MODIFIER LETTER CIRCUMFLEX ACCENT
    u"\x89": u"\u2030", # PER MILLE SIGN
    u"\x8A": u"\u0160", # LATIN CAPITAL LETTER S WITH CARON
    u"\x8B": u"\u2039", # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    u"\x8C": u"\u0152", # LATIN CAPITAL LIGATURE OE
    u"\x8E": u"\u017D", # LATIN CAPITAL LETTER Z WITH CARON
    u"\x91": u"\u2018", # LEFT SINGLE QUOTATION MARK
    u"\x92": u"\u2019", # RIGHT SINGLE QUOTATION MARK
    u"\x93": u"\u201C", # LEFT DOUBLE QUOTATION MARK
    u"\x94": u"\u201D", # RIGHT DOUBLE QUOTATION MARK
    u"\x95": u"\u2022", # BULLET
    u"\x96": u"\u2013", # EN DASH
    u"\x97": u"\u2014", # EM DASH
    u"\x98": u"\u02DC", # SMALL TILDE
    u"\x99": u"\u2122", # TRADE MARK SIGN
    u"\x9A": u"\u0161", # LATIN SMALL LETTER S WITH CARON
    u"\x9B": u"\u203A", # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    u"\x9C": u"\u0153", # LATIN SMALL LIGATURE OE
    u"\x9E": u"\u017E", # LATIN SMALL LETTER Z WITH CARON
    u"\x9F": u"\u0178", # LATIN CAPITAL LETTER Y WITH DIAERESIS
}

def fix_cp1252codes(text):
    # map cp1252 gremlins to real unicode characters
    if re.search(u"[\x80-\x9f]", text):
        def fixup(m):
            s = m.group(0)
            return cp1252.get(s, s)
        if isinstance(text, type("")):
            # make sure we have a unicode string
            text = unicode(text, "iso-8859-1")
        text = re.sub(u"[\x80-\x9f]", fixup, text)
    return text

def clean_utf8(text):
    return filter(lambda x : x > '\x1f' and x < '\x7f', text)

def pairs(iterable, overlapping=False):
    iterator = iterable.__iter__()
    token = iterator.next()
    i = 0
    for lookahead in iterator:
        if overlapping or i % 2 == 0: 
            yield (token, lookahead)
        token = lookahead
        i += 1
    if i % 2 == 0:
        yield (token, None)

def frange(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None:
        inc = 1.0

    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)
        
    return L

def softmax(values):
    a = max(values)
    Z = 0.0
    for v in values:
        Z += math.exp(v - a)
    sm = [math.exp(v-a) / Z for v in values]
    return sm
