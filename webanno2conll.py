#Tranforma WebAnno para Flair

import os

path = 'annotation/'

directories = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for directory in d:
        if '.txt' in directory:
            directories.append(os.path.join(r, directory))
            
from flair.data import Sentence 

list_sentence = []

for d in directories:
  print(d)

  f = open(d + "/webanno.tsv", "r") 
  quedas = f.read()

  for text in quedas.split('\n\n'):
    if text.find('#Text=Evolução:') == 0:

      evolucao = text.split('\n')[1:]
      palavras = []

      for p in evolucao:
        parts = p.split('\t')
        if len(parts) > 3:
          palavras.append(parts[2])

      sentence = Sentence(' '.join(palavras))

      for i, p in enumerate(evolucao):
        parts = p.split('\t')
        if len(parts) > 3:
          if parts[3].find('Tipo de Queda') == 0:
            sentence[i].add_tag('ner', 'queda')

      if len(sentence.tokens) > 0:
        list_sentence.append(sentence)
