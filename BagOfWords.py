import re
import requests
import spacy
from spacy import displacy
import pandas as pd
from bs4 import BeautifulSoup
import re

'''
Trabalho de Bag of Words
Nome: Pedro B. de Quadros
ENUNCIADO:
    Sua tarefa será  gerar a matriz termo documento, dos documentos recuperados da internet e 
imprimir esta matriz na tela. Para tanto:

    a) Considere que todas as listas de sentenças devem ser transformadas em listas de vetores, 
onde cada item será uma das palavras da sentença. 
    b) Todos  os  vetores  devem  ser  unidos  em  um  corpus  único  formando  uma  lista  de  vetores, 
onde cada item será um lexema.  
    c) Este único corpus será usado para gerar o vocabulário. 
    d) O  resultado  esperado  será  uma  matriz  termo  documento  criada  a  partir  da  aplicação  da 
técnica bag of Words em todo o corpus.

'''

def corpus():
  nlp=spacy.load("en_core_web_sm")

  # Urls usadas
  urls = ["https://www.ibm.com/cloud/learn/natural-language-processing",
          "https://en.wikipedia.org/wiki/Natural_language_processing",
          "https://www.techtarget.com/searchenterpriseai/definition/natural-language-processing-NLP",
          "https://www.sas.com/en_us/insights/analytics/what-is-natural-language-processing-nlp.html",
          "https://www.engati.com/glossary/natural-language-processing"
          ]

  # Todos os textos
  textosWeb = []
  # Pegando as urls e extraindo os textos dos htmls dessas paginas
  for url in urls:
      html = requests.get(url).content
      soup = BeautifulSoup(html, features="html.parser")

      for script in soup(["script", "style"]):
          script.extract()

      texto = soup.get_text()

      linhas = (line.strip() for line in texto.splitlines())
      chunks = (phrase.strip() for line in linhas for phrase in line.split("  "))
      texto = "".join(chunk for chunk in chunks if chunk)

      buffer = []

      for token in re.split("[.?!]", texto):
          if token != "":
              buffer.append(token)
      textosWeb.append(buffer)

  # Aplicando o NPL nas sentenças
  doc = []

  # Esse for está criando uma matriz com todos os arrays gerados pelo npl
  for text in textosWeb:
      for sentence in text:
          doc.append(nlp(sentence))

  #print(df.head(75))
  return doc

def preencheContador(vocabulario):
    tabelaDePalavras = []
    for i in range(len(vocabulario)):
        tabelaDePalavras.append(0)
    return tabelaDePalavras

def criaVocabulario(corpus):
    vocabulario = set()
    for x in corpus:
        separacao = x.text.split()
        for y in separacao:
            vocabulario.add(y)
    return sorted(vocabulario)

# Pegando o corpus
def bagOfWords():
    #Obtendo o corpus
    docIncorpus = corpus()
    bagOfWords = []

    vocabulario = criaVocabulario(docIncorpus)

    for x in docIncorpus:
        separacao = x.text.split()
        tabelaDePalavras = preencheContador(vocabulario)
        for y in separacao:
            if vocabulario.index(y):
                tabelaDePalavras[vocabulario.index(y)] += 1
            
        bagOfWords.append(tabelaDePalavras)
    
    return bagOfWords

bagOfWords()
