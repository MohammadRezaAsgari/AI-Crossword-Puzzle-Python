# -*- coding: utf-8 -*-

import beautifultable
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer,util
import csv
import requests
from bs4 import BeautifulSoup




def SERACH(des):
  des = des.split(' ')
  des = '-'.join(des)
  link = f'https://abadis.ir/fatofa/{des}/'
  response = requests.get(link)
  soup = BeautifulSoup(response.text,features="html.parser")
  txt = soup.find('div',{'t':'مترادف ها'})

  if txt==None:
      txt = soup.find('div',{'t':'جدول کلمات'})
      if txt==None:
          return 0
      l = txt.find_all('div',{'class':"boxBd"})
      return l[0].text
  else:
      txt = txt.find_all('div',{'class':None})
      txt = [i.text.split('،') for i in txt]
      newtxt = txt[0]
      for i in range(1,len(txt)):
          for j in txt[i]:
              newtxt.append(j)
      return newtxt



def SERACH2(des):
  des = des.split('#')
  return SERACH(des[0]),SERACH(des[1])




def Classify(des):
  global df
  global data
  model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
  vec_des = model.encode(des)
  vec_des = vec_des.reshape(1, -1)
  compare_scores = util.dot_score(vec_des, df['vectorised'])
  similarity_array = torch.Tensor.tolist(compare_scores)[0]
  sim_indx = similarity_array.index(max(similarity_array))
  groupe = df.iloc[[sim_indx]].values[0][1]
  return data[groupe-1]




def CSP_ROW(sr, indx, table, row, column, fir_sec):
  if isinstance(sr,str):
    sr = [sr]
  word = []
  word2 = []
  i = column-1
  lengh_desired = 0

  lis = [i for i in range(0,column)]
  lis = lis[::-1]
  flag = False
  for i in lis:
    w = table.rows[indx][i]
    if w=='#' and i==column-1:
      continue
    elif w=='#':
      break
    else:
      word.append(w)
      lengh_desired += 1
    
  if fir_sec==2:
    lengh_desired = column - (lengh_desired+1)

  new_list = [i for i in sr if len(i)==lengh_desired]

  if len(new_list)!=1 and fir_sec==1:
    for i in word:
      if i!=' ':
        tmp = word.index(i)
        new_list = [j for j in new_list if j[tmp]==i]

  elif len(new_list)!=1 and fir_sec==2:
    for i in word2:
      if i!=' ':
        tmp = word.index(i)
        new_list = [j for j in new_list if j[tmp]==i]
  
  if len(new_list)!=1:
    return 0
  return new_list[0]





def CSP_COL(sr, ind, table, row, column, fir_sec):
  word = []
  word2 = []
  indx = column - ind -1
  lengh_desired = 0

  flag = False
  for i in range(row):
    w = table.rows[i][indx]
    if w=='#' and not flag:
      flag = True
    elif w=='#' and flag:
      break
    elif w!='#':
      word.append(w)
      lengh_desired += 1

    
  if fir_sec==2:
    lengh_desired = row - (lengh_desired+1)
      
  new_list = [i for i in sr if len(i)==lengh_desired]

  if len(new_list)!=1 and fir_sec==1:
    for i in word:
      if i!=' ':
        tmp = word.index(i)
        new_list = [j for j in new_list if j[tmp]==i]

  elif len(new_list)!=1 and fir_sec==2:
    for i in word2:
      if i!=' ':
        tmp = word.index(i)
        new_list = [j for j in new_list if j[tmp]==i]
  
  if len(new_list)!=1:
    return 0
  return new_list[0]



def create_table(row , column, rows_shape):
  table_shape = []
  for i in range(row):
      table_shape.append([])
      for j in rows_shape[i]:
          if j=="0":
              table_shape[i].append(" ")
          else:
              table_shape[i].append("#")

  table = beautifultable.BeautifulTable()
  for i in range(row):
      table.append_row(table_shape[i])
  return table




def fill_row(word, indx , table, row, column):
  if word=='-':
    return 0
  elif '-' in word:
    new_word = word.split('#')
    new_word = [i for i in new_word if i!='-']
    if len(new_word)==0:
      return 0
    word = new_word[0]
  
  ss = word
  counter = 0

  lis = [i for i in range(0,column)]
  lis = lis[::-1]

  for j in lis:
    if counter+1>len(ss):
        break
    elif table.rows[indx][j]==' ':
      table.rows[indx][j] = ss[counter]
      counter += 1
    elif table.rows[indx][j]!='#' and table.rows[indx][j]!=' ':
      counter += 1
    elif table.rows[indx][j]=='#' and ss[counter]=='#':
      counter += 1
    elif table.rows[indx][j]=='#' and j!=column-1:
      break




def fill_col(word, ind, table, row, column):
  if word=='-':
    return 0
  elif '-' in word:
    new_word = word.split('#')
    new_word = [i for i in new_word if i!='-']
    word = new_word[0]

  ss = word
  counter = 0
  indx = column - 1 - ind
  
  for j in range(row):
    if counter+1>len(ss):
        break
    elif table.rows[j][indx]==' ':
      table.rows[j][indx] = ss[counter]
      counter += 1
    elif table.rows[j][indx]!='#' and table.rows[j][indx]!=' ':
      counter += 1
    elif table.rows[j][indx]=='#' and ss[counter]=='#':
      counter += 1
    elif table.rows[j][indx]=='#' and j!=0:
      break



def func(des, indx, bol, table, row, column, fir_sec, des_or_sr):
  if des=='-':
    return ' '
  elif isinstance(des, list):
    if bol:
      return CSP_ROW(des, indx, table, row, column, fir_sec)
    else:
      return CSP_COL(des, indx, table, row, column, fir_sec)
  elif des_or_sr:
    if bol:
      return CSP_ROW(des, indx, table, row, column, fir_sec)
    else:
      return CSP_COL(des, indx, table, row, column, fir_sec)
  else:
    new_sr = Classify(des)
    if bol:
      return CSP_ROW(new_sr, indx, table, row, column, fir_sec)
    else:
      return CSP_COL(new_sr, indx, table, row, column, fir_sec)



def func2(des, indx, bol, table, row, column):
  sr1 , sr2 = SERACH2(des)
  if sr1==0 and sr2==0:
    des = des.split('#')
    res1,res2 = func(des[0], indx, bol, table, row, column, 1,0),func(des[1], indx, bol, table, row, column, 2,0)
  elif sr1==0:
    des = des.split('#')
    res1,res2 = func(des[0], indx, bol, table, row, column, 1,0),func(sr2, indx, bol, table, row, column, 2,1)
  elif sr2==0:
    des = des.split('#')
    res1,res2 = func(sr1, indx, bol, table, row, column, 1,1),func(des[1], indx, bol, table, row, column, 2,0)
  else:
    res1,res2 = func(sr1, indx, bol, table, row, column, 1,1),func(sr2, indx, bol, table, row, column, 2,1)
  
  if res1==0:
    res1='-'
  if res2==0:
    res2='-'
  return res1+'#'+res2


"""
//////////////////////
Main:

"""
path = 'C:/Users/Roham/Desktop/ProjAI/class_data.txt'
df = pd.read_csv(path, encoding = 'utf-8', error_bad_lines=False, names=('Description', 'Group'))
sent = df['Description']
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
embeddings = model.encode(sent)
df['vectorised'] = embeddings.tolist()

path = 'C:/Users/Roham/Desktop/ProjAI/pot_ans.txt'
file = open(path, "r",encoding="utf8")
data = list(csv.reader(file, delimiter=","))
file.close()



print('Enter table scale: ')
scale = input()
row = int(scale.split()[0])
column = int(scale.split()[1])

print('Enter table shape: ')
input_string = input()[::-1]
chunks  = [input_string[i:i+column] for i in range(0, len(input_string), column)]
rows_shape = [i[::-1] for i in chunks]
table = create_table(row , column, rows_shape)

print(table)

print('Enter descriptions: ')
input_string = input()[1:-1].split('@')


rows_des = input_string[0:row]
columns_des = input_string[row:]

print('rows_des',rows_des)
print('columns_des',columns_des)


for des in rows_des:
  if des=='-':
    continue
  elif '#' in des:
    word = func2(des, rows_des.index(des), 1, table, row, column)
    if word==0:
      continue
    fill_row(word, rows_des.index(des), table, row,column)
  else: 
    sr = SERACH(des)
    if sr==0:
      word = func(des, rows_des.index(des), 1, table, row, column, 1,0)
    else:
      word = func(sr, rows_des.index(des), 1, table, row, column, 1,1)
    if word==0:
      continue
    fill_row(word, rows_des.index(des), table, row,column)


for des in columns_des:
  if des=='-':
    continue
  elif '#' in des:
    word = func2(des, columns_des.index(des), 0, table, row, column)
    if word==0:
      continue
    fill_col(word, columns_des.index(des), table, row,column)
  else: 
    sr = SERACH(des)
    if sr==0:
      word = func(des, columns_des.index(des), 0, table, row, column,1,0)
    else:
      word = func(sr, columns_des.index(des), 0, table, row, column,1,1)
    if word==0:
      continue    
    fill_col(word, columns_des.index(des), table, row,column)

print(table)