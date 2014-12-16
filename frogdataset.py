from storypy import Story
import glob
import codecs
import os
import sys

from frog import Frog, Word

frogger = Frog(int(sys.argv[1]))

def token_boundaries(tokens, original_text):
    curr_pos = 0
    for tok in tokens:
        start_pos = original_text.index(tok, curr_pos)
        # TODO: Check if we fail to find the token!
        end_pos = start_pos + len(tok)
        yield (start_pos, end_pos)
        curr_pos = end_pos

for filename in glob.glob(os.path.join(sys.argv[2], "SINVS*.ann")):
    if 'anomalies' in filename:
        continue
    story = Story.load(filename)
    if sys.argv[3] == 'chars':
        characters = {(start, end): (id, name)
                      for character in story.characters
                      for id, name, start, end in character.chain}
    else:
        characters = {(start, end): (id, name)
                      for location in story.locations
                      for id, name, start, end in location.chain}

    with codecs.open(filename.replace(".ann", ".txt"), encoding='utf-8') as f:
        orig_text = f.read()
        tokens = [word for sent in frogger.tag(orig_text) for word in sent]
        offsets = list(token_boundaries([t[0].decode('utf-8') for t in tokens], orig_text))
        found_characters = {(start, end): False for start, end in characters}
        for char_start, char_end in characters:
            start_found = False
            for i, (start, end) in enumerate(offsets):
                if (start == char_start or
                    (start < char_start < end) or
                    (char_start < start < char_end)):
                    start_found = True
                    tokens[i] = Word(*(tokens[i][:-1] + ('animate',)))
        for token in tokens:
            print '\t'.join(token)
        print '<FB/>'
