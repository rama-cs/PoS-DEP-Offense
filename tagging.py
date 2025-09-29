# -------------------------------
# file: src/data/tagging.py
# -------------------------------
from typing import List, Tuple

# Fixed tag vocabularies for POS/DEP to stable IDs
POS_VOCAB = {
    'ADJ':1,'ADP':2,'ADV':3,'AUX':4,'CCONJ':5,'DET':6,'INTJ':7,'NOUN':8,'NUM':9,
    'PART':10,'PRON':11,'PROPN':12,'PUNCT':13,'SCONJ':14,'SYM':15,'VERB':16,'X':17,'SPACE':18
}

# Common UD dependency relations subset
DEP_VOCAB = {
    'acl':1,'acl:relcl':2,'advcl':3,'advmod':4,'amod':5,'appos':6,'aux':7,'case':8,
    'cc':9,'ccomp':10,'clf':11,'compound':12,'conj':13,'cop':14,'csubj':15,'dep':16,
    'det':17,'discourse':18,'dislocated':19,'expl':20,'fixed':21,'flat':22,'goeswith':23,
    'iobj':24,'list':25,'mark':26,'nmod':27,'nsubj':28,'nummod':29,'obj':30,'obl':31,
    'orphan':32,'parataxis':33,'punct':34,'reparandum':35,'root':36,'vocative':37,'xcomp':38
}

POS_PAD_ID = 0
DEP_PAD_ID = 0
