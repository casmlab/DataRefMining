#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Basic Data Manipulation
import pandas as pd
import numpy as np
import re
# Information Extraction
from openie import StanfordOpenIE
# Calssify
from nltk.corpus import verbnet as vn
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import Matcher
from spacy.symbols import dobj, obj, pobj, acomp, ccomp, pcomp, xcomp, conj, acomp, ccomp, pcomp, xcomp, advmod, amod
from spacy.symbols import neg, det, aux, prep, poss, nsubj, nsubjpass, csubj, csubjpass, det, prt
from spacy.symbols import VERB, AUX, DET, ADP, ADV, ADJ, NOUN, PRON, PROPN, PART
from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex


# In[3]:


### Extract
# ref: https://github.com/philipperemy/stanford-openie-python
def extract_triple(text):
    with StanfordOpenIE() as client:
        return client.annotate(text)


# In[4]:


### Select
# for similar jsons, keep the shortest json 
# include triples with and without the dataset_prediction (for possible coref)

# Helper: count the "longest" object
def long_stuff(compare_stuff):  # a list of objects, return a list of longest objects
    stf = pd.DataFrame()
    stf["stuff"] = compare_stuff
    stf["len"] = stf.stuff.apply(lambda x: x.count(" "))  # a list of object length
    return stf.stuff[stf.len.idxmax()]  # we only interested in one object - the longest
    # later, we can extract more info from the long object


# Helper: get the stuff that are not a subject of others
def no_subset(A):  # a list of strings
    return list(set([x for x in A if not any(x in y and x != y for y in A)]))


# Helper: Using rules to select tri:
def select_triple(triplets):
    """
    Extract a equence of subject-verb-object (SVO) triples
    from a opie_tri fucntion acquired and processed doc,
    including both active and passive entities and actions.

    Args:
        triplets are lists of dictionaries;
        assume number of triplets >=2, i.e. len(triplets)>=2.

    Yields:
        List of dictionaries: the main/longest triplets from ``triplets``
        representing a (subject, verb, object) triple.
    """
    # initiate
    selected = []
    # only extract the longest subject
    compare_sub = list(map(lambda x: x["subject"], triplets))
    subjects = no_subset(compare_sub)
    for sub in subjects:
        # extract different unique relations
        tri_for_this_sub = [d for d in triplets if d["subject"] == sub]
        compare_rel = list(map(lambda x: x["relation"], tri_for_this_sub))
        relations = no_subset(compare_rel)
        # for each of the relation, extract the longest obeject
        for rel in relations:
            tri_for_this_rel = [d for d in tri_for_this_sub if d["relation"] == rel]
            compare_obj = list(map(lambda x: x["object"], tri_for_this_rel))
            this_object = long_stuff(compare_obj)
            selected.append({"subject": sub, "relation": rel, "object": this_object})

    # for the selected ones, if both subject and object are the same
    # we keep the one with the longest relation
    if len(selected) > 1:
        # initiate
        re_select = []

        # give group number
        group_list = [0] * len(selected)
        group_list[0] = 1  # initiate the first group
        group_num = 1
        pos = 0
        # if both subject and object are the same
        # assign the same group number
        # but avoid reassignment
        for i in range(len(selected)):
            pos = i
            for j in range(i + 1, len(selected)):
                pos += 1
                if group_list[pos] == 0:
                    if selected[i]["subject"] == selected[j]["subject"] and                             selected[i]["object"] == selected[j]["object"]:
                        group_list[pos] = group_num
                    else:
                        group_list[pos] = group_num + 1
            group_num += 1

        # for each group, find the longest relation
        numbers = list(set(group_list))
        selected_df = pd.DataFrame()
        selected_df["tri"] = selected
        selected_df["grp"] = group_list
        for num in numbers:
            # find all the triplets for this group
            tri_for_this_grp = selected_df[selected_df.grp == num].tri
            # acquire a list of relations in this group
            compare_rel = list(map(lambda x: x["relation"], tri_for_this_grp))
            # find the longest relation
            this_rel = long_stuff(compare_rel)
            # get the subjects and objects for this group
            # since these values are the same for each one of the values
            # we extract the first one
            this_subject = list(tri_for_this_grp)[0]["subject"]  # change a series to a list
            this_object = list(tri_for_this_grp)[0]["object"]  # change a series to a list
            # all the triplets in this group to the list
            re_select.append({"subject": this_subject, "relation": this_rel, "object": this_object})

        return re_select

    return selected


# In[6]:


### Classify

# Subjects and Objects
# identify "I" and "we" 
# detect (probably not full currently) dataset in part or in ful

def find_author(text):
    '''
    input: full sentence (text)
    output: bool of find or not
    '''
    text_lower = text.lower()
    words = text_lower.split()
    if ("i" in words or "we" in words):
        return True
    return False

data_keywords = ['data', 'data\s*(?:set|base)s?', 'corp(us|ora)', 'tree\s*bank', 
            '(?:train|test|validation|testing|trainings?)\s*(?:set)',
            'collections?', 'benchmarks?', 'surveys?', 'samples?', 'stud(y|ies)']
data_pattern= re.compile(r'\b(' + '|'.join(data_keywords) + r')\b', flags = re.IGNORECASE)

def find_dataset(text,data_name):
    '''
    input: full sentence (text) and dataset_prediction (data_name)
    output: bool of find or not
    '''
    # use predicted dataset names to find
    data_name_list = data_name.split()
    words = text.split()
    for data_name_token in data_name_list: # anything match counts
        if data_name_token in words:
            return True
    
    # use data citation pattern to find
    if re.search(data_pattern,text):
        return True
    
    return False


# In[22]:


# Relations

# ref: https://verbs.colorado.edu/verb-index/VerbNet_Guidelines.pdf
# https://docs.google.com/spreadsheets/d/18kn2z2df-M4ncUmoHPGqbs5nJyGL-k9R0d2slXdT830/edit?usp=sharing
verb_class_df = pd.read_csv("VerbNet_LF.csv")
verb_class = verb_class_df.drop("Verb Class",axis=1).set_index("Class Number").to_dict()['Verb Type']

def get_classes(classids):
    """
    input VerbNet classids (long)
    output a set of VerbNet classes (parenent-level, short)
    """
    classes = set()
    for classid in classids:
        # remove the word itself
        this_classid_long = classid.split("-")[1]
        # get the class -- the string upto the first non-digit
        this_classid_short = int(re.search(r'(\d+)',this_classid_long).group(1))
        classes.add(this_classid_short)
    return classes

def get_rel_classes(rel_nlp):
    rel_classes = set()
    for token in rel_nlp:
        token_lemma = token.lemma_
        token_classids = vn.classids(token_lemma)
        token_classes = get_classes(token_classids)
        rel_classes.update(token_classes)
    return rel_classes


# In[8]:


# AEO Categories (tentative)
# ref: https://github.com/lizhouf/semantic_triplets/blob/main/scr/add_aeo.py

# clauses are characterized by:
# - temporal organization (the order in which the subject narrates events and actions in the story), 
# - evaluative description (personal assessments made by the narrator), and 
# - contextual orientation (usually information provided by the narrator that helps orient the listener)

# ref: Labov and Waletsky 1997 Labov, William, and Joshua Waletzky. 1997. “Narrative Analysis: Oral Versions of Personal Experience.” Journal of Narrative & Life History 7 (1–4): 3–38.

# We have: 
# - Active Agency
# - Passive Agency
# - Possible Agency
# - Evaluative Description
# - Contextual Orientation


# Example Key Words
evaluation_verbs = ["feel","smell","taste","look","hear","see","think","know"]
orientation_verbs = ["remember","bear","grow","belong"]
imagine_verbs = ["want","should","would","could","can","might","may"]

def get_cat(this_rel, this_obj):
    '''
    input spaCy Spans this_rel, this_obj 
    output category result
    '''
    
    # initate category result
    
    this_cat = ""

    # initiate the rule components

    rel_has_evaluation = 0
    rel_has_orientation = 0
    rel_has_imagine = 0

    rel_has_be = 0
    rel_has_have = 0
    rel_has_to = 0

    rel_has_neg = 0

    rel_has_VBG = 0
    rel_num_verb = 0

    obj_is_adj = 0  # only adj, no NOUN+

    obj_has_no = 0
    
    # give value
    for rel in this_rel:

        # rel lemmas
        try:
            if rel.lemma_ in evaluation_verbs:
                rel_has_evaluation = 1
            if rel.lemma_ in imagine_verbs:
                rel_has_imagine = 1
            if rel.lemma_ in orientation_verbs:
                rel_has_orientation = 1
            if rel.lemma_ == "be":
                rel_has_be = 1
            if rel.lemma_ == "have":
                rel_has_have = 1
            if rel.lemma_ == "to":
                rel_has_to = 1
        except:  # avoid no lemma
            0

        # rel dep
        try:
            if rel.dep == neg:
                rel_has_neg = 1
        except:
            0

        # rel pos
        try:
            if (rel.pos == VERB or rel.pos == AUX):
                rel_num_verb = rel_num_verb + 1
        except:
            0

        # rel tag
        try:
            if rel.tag_ == "VBG":
                rel_has_VBG = 1
        except:
            0

    for obj in this_obj:
        
        # obj lemma
        try:
            if obj.lemma_ == "no":
                obj_has_no = 1
        except:
            0

    for obj in this_obj:  # seperate, want to break
        # obj pos
        try:
            if obj.pos == ADJ:
                obj_is_adj = 1
            if obj.pos in [NOUN,PRON,PROPN]:
                obj_is_adj = 0
                break
        except:
            0

    # judge:

    # fixed words
    if rel_has_evaluation and obj_is_adj:
        this_cat ="Evaluation"
    elif rel_has_imagine:
        this_cat ="Agency_Possible"
    elif rel_has_orientation:
        this_cat ="Orientation"

    # neg
    elif rel_has_neg or obj_has_no:
        this_cat ="Orientation"

    # have
    elif rel_has_have:
        if rel_has_to:
            this_cat ="Agency_Passive" # no longer coercive
        else:
            this_cat ="Orientation"

    # be
    elif rel_has_be:
        if obj_is_adj:
            this_cat ="Evaluation"
        elif rel_has_VBG:
            this_cat ="Agency_Active"
        elif rel_num_verb > 1:
            this_cat ="Agency_Passive"
        elif rel_num_verb == 1:
            this_cat ="Orientation"

    # if none of the above, then assign active:
    else:
        this_cat = "Agency_Active"

    return this_cat


# In[9]:


### Import and Manipulate Data
citations = pd.read_csv("/nfs/turbo/hrg/data_detection/outputs_pipeline/4_pubs_sents_preds_ids.csv")
citations_true = citations[~citations.dataset_prediction.isna()].reset_index(drop=True)

# In[23]:


# %%capture
# %timeit

# Initiate result df
df = pd.DataFrame(columns=["paper_id","sentence_text","dataset_prediction",
                           "subject","relation","object",
                           "subject_category","relation_categories","object_category","AEO_category"])

for i in range(len(citations_true)):
#for i in range(3): # test
    paper_id = citations_true.paper_id[i]
    sentence_text = citations_true.sentence_text[i]
    dataset_prediction = citations_true.dataset_prediction[i]
    # Extract triple
    triples = extract_triple(sentence_text)
    # Select triple
    triples_selected = select_triple(triples)
    
    # loop
    for triple in triples_selected:
        sub = str(triple["subject"]) # ensure string, not int or bool
        rel = str(triple["relation"])
        rel_nlp = nlp(rel)
        obj = str(triple["object"])
        # Subject
        Subject_Cat = "author" if find_author(sub) else "dataset" if find_dataset(sub,dataset_prediction) else "other"
        # Object
        Object_Cat = "author" if find_author(obj) else "dataset" if find_dataset(obj,dataset_prediction) else "other"
        # Relation
        Relation_Cats = get_rel_classes(rel_nlp)
        # AEO
        AEO_Cat = get_cat(rel_nlp,nlp(obj))
        # Append
        df.loc[len(df.index)] = [paper_id, sentence_text, dataset_prediction,
                                 sub, rel, obj, 
                                 Subject_Cat, Relation_Cats, Object_Cat, AEO_Cat] 


# In[24]:


df.to_csv("ICPSR_bib_data_citation_rhetoric_v01.csv",index=False)


# We are more interested in the columns with at least one author or dataset.
