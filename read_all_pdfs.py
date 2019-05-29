import PyPDF2
import pandas as pd
import os

directory = '/Users/ghazalghalebandi/PycharmProjects/pdf-parser/pdf/main'

list_of_class = []
list_of_text = []



for filename in os.listdir(directory):
    # print(filename.endswith(".pdf"))
    if filename.endswith(".pdf"):
        path = (directory + '/' + filename)
        f = open(path, mode='rb')
        # print(f)
        pdfReader = PyPDF2.PdfFileReader(f)

        pageObj = pdfReader.getPage(0)
        content = pageObj.extractText()

        remove_class = content.replace('class:', '')
        remove_text = remove_class.replace('text:', '')
        remove_dot = remove_text.replace("." , '')

        list_of_words = remove_dot.split()
        class_content = list_of_words[0]
        list_of_class.append(class_content)

        text_content = list_of_words[1]
        list_of_text.append(text_content)
        continue
    else:
        continue


data = pd.DataFrame()
data['label'] = list_of_class
data['text'] = list_of_text
print('row and col', data.shape)

# print(data.head)
export_csv = data.to_csv ('/Users/ghazalghalebandi/PycharmProjects/pdf-parser/data.csv', index = None, header=True)
# print(data.groupby('class').count())

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2)
print('train row and col' , train.shape)
print('test row and col' , test.shape)

export_train = train.to_csv ('/Users/ghazalghalebandi/PycharmProjects/pdf-parser/train.csv', index = None, header=True)
export_test = test.to_csv ('/Users/ghazalghalebandi/PycharmProjects/pdf-parser/test.csv', index = None, header=True)
