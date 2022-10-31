import requests
import spacy
from spacy import displacy
import pandas as pd
from bs4 import BeautifulSoup
import re

# Pedro Bianchini de Quadros

'''
Sua  tarefa  será  transformar  um  conjunto  de  5  sites,  sobre  o  tema  de  processamento  de 
linguagem natural em um conjunto de cinco listas distintas de sentenças. Ou seja, você fará uma função 
que, usando a biblioteca Beautifull Soap, faça a requisição de uma url, e extrai todas as sentenças desta 
url. Duas condições são importantes:  
  a) A página web (url) deve apontar para uma página web em inglês contendo, não menos que 
1000 palavras.  
  b) O texto desta página deverá ser transformado em um array de senteças.  
 
Para separar as sentenças você pode usar os sinais de pontuação ou as funções da biblibioteca 
Spacy.
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

#corpus()