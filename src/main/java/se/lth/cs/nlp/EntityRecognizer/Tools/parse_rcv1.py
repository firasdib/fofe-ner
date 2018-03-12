# -*- coding: utf-8 -*-

import re, glob

from lxml import etree
from os import path, remove

ROOT_FOLDER = '/Users/firasdib/exjobb/reuters/RCV/rcv1'
OUTPUT_FILE = '/Users/firasdib/exjobb/reuters/RCV/rcv1_parsed.txt'


try:
    remove(OUTPUT_FILE)
except OSError:
    pass

write_file = open(OUTPUT_FILE, 'a+')

def parse_xml(file):
    tree = etree.parse(file)
    text = etree.tostring(tree, encoding='utf8', method='text')

    # Cleanup the file a bit
    text = re.sub(r'\(c\) reuters limited \d+', '', text, 0, flags=re.IGNORECASE)

    # Replace dates
    text = re.sub(r'\b\d+([/-])\d+\1\d+\b', ' <DATE> ', text)

    # Remove digits
    text = re.sub(r'\b\d+(?:[.,]\d+)?\b', ' <NUMBER> ', text)

    # Replace apostrophes
    text = re.sub(r'(?<=[a-z])(\'s)', ' \\1 ', text, 0, flags=re.IGNORECASE)
    text = re.sub(r'([a-z]\'[a-z])', ' \\1 ', text)

    # Replace punctuation -- see https://regex101.com/r/zlUuEx/1/
    text = re.sub(ur'((?:(?:[a-z]{1,3}\.)+(?:[a-z]{1,3}|(?= +[a-z])))|[.()\[\]{},%£€#$=:;!?+*@"]|-+)', ' \\1 ', text, 0, flags=re.IGNORECASE)

    # Remove whitespace
    text = re.sub(r'(\s){2,}', '\\1', text)

    return text


all_files = glob.glob(path.join(ROOT_FOLDER, '**/*.xml'))
all_files_len = len(all_files)

print "Going to parse %s file(s)" % (all_files_len)

for index, xml_file in enumerate(all_files):
    parsed_xml_file = parse_xml(xml_file)

    if index % 1000 == 0:
        print "Parsing file %i/%i" % (index, all_files_len)

    write_file.write(parsed_xml_file)


