#!/usr/bin/python
# -*- coding: utf-8 -*-

import re, glob, sys

from lxml import etree
from os import path, remove
from BeautifulSoup import BeautifulSoup

ROOT_FOLDER = '/Users/firasdib/exjobb/data/ldc/data/2015/training/source_docs/eng_out'
OUTPUT_FILE = '/Users/firasdib/exjobb/data/ldc/data/2015/training/source_docs/eng_out/tac2015.conll'
TAB_FILE = '/Users/firasdib/exjobb/data/ldc/data/2015/training/tac_kbp_2015_tedl_training_gold_standard_entity_mentions.tab'

tabFile = open(TAB_FILE).read().split('\n')
file2mention = {}

for i in range(len(tabFile)):
    split = tabFile[i].split('\t')

    if len(split) > 1:
        file = split[3]
        type = split[5]
        subtype = split[6]

        file_split = file.split(':')
        file_name = file_split[0]

        indices = file_split[1].split('-')
        start_idx = indices[0]
        end_idx = int(indices[1]) + 1

        try:
            file_mention_list = file2mention[file_name]
        except:
            file2mention[file_name] = []
            file_mention_list = file2mention[file_name]

        obj = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "entity": type + "-" + subtype
        }

        file_mention_list.append(obj)

        print obj




try:
    remove(OUTPUT_FILE)
except OSError:
    pass

write_file = open(OUTPUT_FILE, 'a+')

def parse_xml(file):
    tree = etree.parse(file)
    text = etree.tostring(tree, encoding='utf8', method='text')

    # Remove HTML elements
    #text = BeautifulSoup(text).text

    # Remove urls
    text = re.sub(r'\bhttps?://[^/]+(?:/\S*)?', '', text)

    # Remove multiple dots
    text = re.sub(r'\.{2,}', ' ', text)

    # Replace dates
    text = re.sub(r'\b\d+([/-])\d+\1\d+\b', ' <DATE> ', text)

    # Remove digits
    text = re.sub(r'\b\d+(?:[.,]\d+)?\b', ' <NUMBER> ', text)

    # Replace apostrophes
    text = re.sub(r'(?<=[a-z])(\'s)', ' \\1 ', text, 0, flags=re.IGNORECASE)
    text = re.sub(r'([a-z]\'[a-z])', ' \\1 ', text)

    # Replace punctuation -- see https://regex101.com/r/zlUuEx/1/
    #text = re.sub(ur'((?:(?:[a-z]{1,3}\.)+(?:[a-z]{1,3}|(?= +[a-z])))|[.()\[\]{},%£€#$=:;!?+*@"]|-+)', ' \\1 ', text, 0, flags=re.IGNORECASE)

    # Remove whitespace
    text = re.sub(r'(\s){2,}', '\\1', text)

    #if re.text.

    return text


all_files = glob.glob(path.join(ROOT_FOLDER, '**/*.xml'))
all_files_len = len(all_files)

print "Going to parse %s file(s)" % (all_files_len)

for index, xml_file in enumerate(all_files):
    parsed_xml_file = parse_xml(xml_file)

    sys.stdout.write("\rParsing file %i/%i" % (index, all_files_len))

    write_file.write(parsed_xml_file)


