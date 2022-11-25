

import sys
import re
import os
from util import *
from Tokenizer import PTBTokenizer


assert len(sys.argv) == 1


# main
# loop over sentences cum annotation
tokenizer = PTBTokenizer()
sentence = ''
for line in sys.stdin:
    line = line.decode("utf8").strip()
    if line.startswith("S "):
        sentence = line[2:]
        sentence_tok = "S " + ' '.join(tokenizer.tokenize(sentence))
        print sentence_tok.encode("utf8")
    elif line.startswith("A "):
        fields = line[2:].split('|||')
        start_end = fields[0]
        char_start, char_end = [int(a) for a in start_end.split()]
        # calculate token offsets
        prefix = sentence[:char_start]
        tok_start = len(tokenizer.tokenize(prefix))
        postfix = sentence[:char_end]
        tok_end = len(tokenizer.tokenize(postfix))
        start_end = str(tok_start) + " " + str(tok_end)
        fields[0] = start_end
        # tokenize corrections, remove trailing whitespace
        corrections = [(' '.join(tokenizer.tokenize(c))).strip() for c in fields[2].split('||')]
        fields[2] = '||'.join(corrections)
        annotation =  "A " + '|||'.join(fields)
        print annotation.encode("utf8")
    else:
        print line.encode("utf8")

