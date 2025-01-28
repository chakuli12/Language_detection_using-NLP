# Language_detection_using-NLP

Creating a language detection project using Natural Language Processing (NLP) involves using algorithms to identify the language of a given text based on its features. You can build a language detection system in Python, for example, using popular libraries like langdetect, langid, or custom models built with machine learning techniques.      

1.What is Language Detection.

Language detection is the process of identifying the language of a given text.

It’s essential for many applications in natural language processing (NLP), machine learning, and AI systems.

2.Why Is Language Detection Important?

Improves communication in multilingual environments.

Enables translation services (e.g., Google Translate).

Enhances content moderation and filtering.

Optimizes SEO and user experience. 

3.Common Methods of Language Detection.

N-gram Models: Analyze text by breaking it down into sequences of 'n' characters or words.

Statistical Methods: Identify languages based on letter frequencies and word patterns.

4.Challenges in Language Detection.

Short Texts: Limited data can make accurate detection difficult.

Similar Languages: For example, Spanish vs. Portuguese.

Code-Switching: Multiple languages used within the same text.

Dialects: Variations in a language across regions. 

Accuracy of Language Detection.
Detection is more accurate with longer texts.
Modern models can handle ambiguity, but challenges remain.
Models improve with more data and training. 

Language Detection Applications

Content Moderation: Detect inappropriate language.
Translation Services: Automatic detection before translating.
Search Engines: Personalized search results based on language preference.
Data Analysis: Grouping data by language for better insights. 

Real-World Example
1.Google Translate uses language detection to identify the 
input language and provide accurate translations. 
2.Microsoft Translator:
It helps you translate images, screenshots, texts, and voice translations 
for more than 60 languages ranging from Hindi to Spanish, and Urdu 
to French. FYI all of this can be downloaded for offline use as well.
3.iTranslate : The iTranslate app by a company named Sonico Mobile,
 helps you translate text, websites or lookup words with
 meanings and even verb conjugations in more than 100 languages.
4.Linguee : Linguee is a web-based service launched in 2009 that 
helps you to translate singular words or sentences in place of paragraphs 
and supports more than 25 dialects. Similar to any other language translator,
 you can use it offline as well, at times of poor internet connection.

Tools and Libraries for Language Detection
Python Libraries: 
1.Langdetect-For the detect the language.
2.Matplotlib - Matplotlib is a low level graph plotting library in python that serves as a visualization utility.
3.Pandas- Pandas is a Python library used for working with data sets
4.Numpy- NumPy is a Python library used for working with arrays.
5.Seaborn - Seaborn is a library that uses Matplotlib underneath to plot graphs. 
Streamlit is an open-source Python library that makes it easy to create and share custom web apps for machine learning and data science.

Future of Language Detection
More accurate models using deep learning.
Better handling of multilingual, dialectal, and regional variations.
Expansion in AI-driven communication tools and virtual assistants. 

Conclusion
Language detection is an evolving field that plays 
a crucial role in technology and communication. By understanding its methods, challenges, and applications, we can better utilize it in various domains.

#Implementation
#Importing libraries and dataset

#import all the libraries like pandas ,string,numpy,re,matplotlib and seaborn

import string                        #strings in Python are arrays of bytes representing unicode characters.
import pandas as pd                  #Pandas is a Python library used for working with data sets
import numpy as np                   #NumPy is a Python library used for working with arrays.
import re                            #A RegEx, or Regular Expression, is a sequence of characters that forms a search pattern. eg. findall,search,split,sub
import matplotlib.pyplot as plt      #Matplotlib is a low level graph plotting library in python that serves as a visualization utility.
import seaborn as sns                #Seaborn is a library that uses Matplotlib underneath to plot graphs. It will be used to visualize random distributions.


# Loading the dataset

df=pd.read_csv("C:\\Users\\User\\Downloads\\Language Detection.csv~\\Language Detection.csv")
df

#there are 10337 rows × 2 columns
#In this datasets there are text and language. In this all over languages like english,hindi,french,arabic,kannada,tamil,telugu,gurjati etc.


#As I told you earlier this dataset contains text details for 17 different languages. So let’s count the value count for each language.

df["Language"].value_counts()

Language
English       1385
French        1014
Spanish        819
Portugeese     739
Italian        698
Russian        692
Sweedish       676
Malayalam      594
Dutch          546
Arabic         536
Turkish        474
German         470
Tamil          469
Danish         428
Kannada        369
Greek          365
Hindi           63
Name: count, dtype: int64
0         Nature in the broadest sense is the natural p...
1        Nature can refer to the phenomena of the physi...
2        The study of nature is a large if not the only...
3        Although humans are part of nature human activ...
4        1 The word nature is borrowed from the Old Fre...
                               ...                        
10332    ನಿಮ್ಮ ತಪ್ಪು ಏನು ಬಂದಿದೆಯೆಂದರೆ ಆ ದಿನದಿಂದ ನಿಮಗೆ ಒ...
10333    ನಾರ್ಸಿಸಾ ತಾನು ಮೊದಲಿಗೆ ಹೆಣಗಾಡುತ್ತಿದ್ದ ಮಾರ್ಗಗಳನ್...
10334    ಹೇಗೆ  ನಾರ್ಸಿಸಿಸಮ್ ಈಗ ಮರಿಯನ್ ಅವರಿಗೆ ಸಂಭವಿಸಿದ ಎಲ...
10335    ಅವಳು ಈಗ ಹೆಚ್ಚು ಚಿನ್ನದ ಬ್ರೆಡ್ ಬಯಸುವುದಿಲ್ಲ ಎಂದು ...
10336    ಟೆರ್ರಿ ನೀವು ನಿಜವಾಗಿಯೂ ಆ ದೇವದೂತನಂತೆ ಸ್ವಲ್ಪ ಕಾಣು...
Name: Text, Length: 10337, dtype: object
0         Nature in the broadest sense is the natural p...
1        Nature can refer to the phenomena of the physi...
2        The study of nature is a large if not the only...
3        Although humans are part of nature human activ...
4        1 The word nature is borrowed from the Old Fre...
                               ...                        
10332    ನಿಮ್ಮ ತಪ್ಪು ಏನು ಬಂದಿದೆಯೆಂದರೆ ಆ ದಿನದಿಂದ ನಿಮಗೆ ಒ...
10333    ನಾರ್ಸಿಸಾ ತಾನು ಮೊದಲಿಗೆ ಹೆಣಗಾಡುತ್ತಿದ್ದ ಮಾರ್ಗಗಳನ್...
10334    ಹೇಗೆ  ನಾರ್ಸಿಸಿಸಮ್ ಈಗ ಮರಿಯನ್ ಅವರಿಗೆ ಸಂಭವಿಸಿದ ಎಲ...
10335    ಅವಳು ಈಗ ಹೆಚ್ಚು ಚಿನ್ನದ ಬ್ರೆಡ್ ಬಯಸುವುದಿಲ್ಲ ಎಂದು ...
10336    ಟೆರ್ರಿ ನೀವು ನಿಜವಾಗಿಯೂ ಆ ದೇವದೂತನಂತೆ ಸ್ವಲ್ಪ ಕಾಣು...
Name: Text, Length: 10337, dtype: object


#here we print top 5 row using this head method
df.head()
#this is top 5 rows from datasets

#we can also print the last 5 rows using this method which tail.
df.tail()

#Text Preprocessing
#This is a dataset created using scraping the Wikipedia, so it contains many unwanted symbols, 
#numbers which will affect the quality of our model. So we should perform text preprocessing techniques.
#Now, In this part we remove the punctuation like comma,semicolon,full stop.
#here we create def fuction and apply for loop and remove all the unneccessary information like punctuation.

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text
    
#In this we create df variable and pass the varibale text 

df["Text"] = df["Text"].apply(remove_punctuations)
df["Text"]

#after applying thsi method we got data neat and clean which is remove all the punctuation.

Nature in the broadest sense is the natural p...
1        Nature can refer to the phenomena of the physi...
2        The study of nature is a large if not the only...
3        Although humans are part of nature human activ...
4        1 The word nature is borrowed from the Old Fre...
                               ...                        
10332    ನಿಮ್ಮ ತಪ್ಪು ಏನು ಬಂದಿದೆಯೆಂದರೆ ಆ ದಿನದಿಂದ ನಿಮಗೆ ಒ...
10333    ನಾರ್ಸಿಸಾ ತಾನು ಮೊದಲಿಗೆ ಹೆಣಗಾಡುತ್ತಿದ್ದ ಮಾರ್ಗಗಳನ್...
10334    ಹೇಗೆ  ನಾರ್ಸಿಸಿಸಮ್ ಈಗ ಮರಿಯನ್ ಅವರಿಗೆ ಸಂಭವಿಸಿದ ಎಲ...
10335    ಅವಳು ಈಗ ಹೆಚ್ಚು ಚಿನ್ನದ ಬ್ರೆಡ್ ಬಯಸುವುದಿಲ್ಲ ಎಂದು ...
10336    ಟೆರ್ರಿ ನೀವು ನಿಜವಾಗಿಯೂ ಆ ದೇವದೂತನಂತೆ ಸ್ವಲ್ಪ ಕಾಣು...
Name: Text, Length: 10337, dtype: object


# sklearn.model_selection a module that provides tools for splitting datasets into training and testing sets, performing cross-validation, 
# and generally managing the process of selecting the best model parameters for machine learning tasks 

#The train_test_split() method is used to split our data into train and test sets

from sklearn.model_selection import train_test_split

#Separating Independent and Dependent features
#Now we can separate the dependent and independent variables, here text data is the independent variable and 
#the language name is the dependent variable.

X=df.iloc[:,0]
Y=df.iloc[:,1]

X

0         Nature in the broadest sense is the natural p...
1        Nature can refer to the phenomena of the physi...
2        The study of nature is a large if not the only...
3        Although humans are part of nature human activ...
4        1 The word nature is borrowed from the Old Fre...
                               ...                        
10332    ನಿಮ್ಮ ತಪ್ಪು ಏನು ಬಂದಿದೆಯೆಂದರೆ ಆ ದಿನದಿಂದ ನಿಮಗೆ ಒ...
10333    ನಾರ್ಸಿಸಾ ತಾನು ಮೊದಲಿಗೆ ಹೆಣಗಾಡುತ್ತಿದ್ದ ಮಾರ್ಗಗಳನ್...
10334    ಹೇಗೆ  ನಾರ್ಸಿಸಿಸಮ್ ಈಗ ಮರಿಯನ್ ಅವರಿಗೆ ಸಂಭವಿಸಿದ ಎಲ...
10335    ಅವಳು ಈಗ ಹೆಚ್ಚು ಚಿನ್ನದ ಬ್ರೆಡ್ ಬಯಸುವುದಿಲ್ಲ ಎಂದು ...
10336    ಟೆರ್ರಿ ನೀವು ನಿಜವಾಗಿಯೂ ಆ ದೇವದೂತನಂತೆ ಸ್ವಲ್ಪ ಕಾಣು...
Name: Text, Length: 10337, dtype: object

Y

0        English
1        English
2        English
3        English
4        English
          ...   
10332    Kannada
10333    Kannada
10334    Kannada
10335    Kannada
10336    Kannada
Name: Language, Length: 10337, dtype: object

#Train Test Splitting
#We preprocessed our input and output variable. The next step is to create the training set, for training the model and test set, 
#for evaluating the test set. For this process, we are using a train test split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2)



X_train,X_test,Y_train,Y_test
#X_train


(6256    В октябре 2007 года статья Reuters озаглавленн...
 9429                         أنا بخير ولكن شكرا على العرض
 4029         et tu es comme oh ça lexplique qui lexplique
 9806                                           Vorschläge
 1062    to tell me which of these ten words was your f...
                               ...                        
 1681    ആരംഭിക്കുന്നതിനുള്ള ഏറ്റവും മികച്ച വീഡിയോ ഇതാണ...
 7402    Il data mining sfrutta i metodi dellapprendime...
 2243    நீங்கள் வீட்டிற்கு வந்தீர்கள் என்று எந்த கவலைய...
 7102    du bliver nødt til at spise alt dette jeg troe...
 9667                                      Es tut mir leid
 Name: Text, Length: 8269, dtype: object,
 4344    In september 2008 ontving Wikipedia de Quadrig...
 9724    Dies ist ein Kinderspiel was bedeutet dass es ...
 7256    Le varie edizioni di Wikipedia facenti capo a ...
 6248    Другие могут опираться на утверждение Википеди...
 6797    i frankrig blev rendezvous tidligere brugt af ...
                               ...                        
 104     It is not known if Titans lakes are fed by riv...
 3483    Les mêmes principes fondateurs de rédaction so...
 9240            لذلك تأكد واحدًا تلو الآخر من نطقك للكلمة
 8774                                 jag ställde till det
 7279    I redattori stessi di Wikipedia sono stati piu...
 Name: Text, Length: 2068, dtype: object,
 6256      Russian
...
 3483      French
 9240      Arabic
 8774    Sweedish
 7279     Italian
 Name: Language, Length: 2068, dtype: object

 X_test

 4344    In september 2008 ontving Wikipedia de Quadrig...
9724    Dies ist ein Kinderspiel was bedeutet dass es ...
7256    Le varie edizioni di Wikipedia facenti capo a ...
6248    Другие могут опираться на утверждение Википеди...
6797    i frankrig blev rendezvous tidligere brugt af ...
                              ...                        
104     It is not known if Titans lakes are fed by riv...
3483    Les mêmes principes fondateurs de rédaction so...
9240            لذلك تأكد واحدًا تلو الآخر من نطقك للكلمة
8774                                 jag ställde till det
7279    I redattori stessi di Wikipedia sono stati piu...
Name: Text, Length: 2068, dtype: object

Y_train

6256      Russian
9429       Arabic
4029       French
9806       German
1062      English
          ...    
1681    Malayalam
7402      Italian
2243        Tamil
7102       Danish
9667       German
Name: Language, Length: 8269, dtype: object

Y_test

4344       Dutch
9724      German
7256     Italian
6248     Russian
6797      Danish
          ...   
104      English
3483      French
9240      Arabic
8774    Sweedish
7279     Italian
Name: Language, Length: 2068, dtype: object

#Model Training and Prediction 
#The sklearn. feature_extraction module can be used to extract features in a format supported by machine learning 
#algorithms from datasets consisting of formats such as text and image.

from sklearn import feature_extraction

#Term frequency-inverse document frequency (TF-IDF) is a feature vectorization method widely used in text mining to reflect the 
#importance of a term to a document in the corpus. Denote a term by t , a document by d , and the corpus by D .

vec=feature_extraction.text.TfidfVectorizer(ngram_range=(1,2),analyzer='char')


#linear_model is a class of the sklearn module if contain different functions for performing machine learning with linear models

from sklearn import pipeline
from sklearn import linear_model

model_pipe=pipeline.Pipeline([('vec',vec),('clf',linear_model.LogisticRegression())])

#model fitting 

model_pipe.fit(X_train,Y_train)

#clases 

model_pipe.classes_

array(['Arabic', 'Danish', 'Dutch', 'English', 'French', 'German',
       'Greek', 'Hindi', 'Italian', 'Kannada', 'Malayalam', 'Portugeese',
       'Russian', 'Spanish', 'Sweedish', 'Tamil', 'Turkish'], dtype=object)



# Prediction 

predict_val=model_pipe.predict(X_test)

#Model Evaluation

#import metrics 
from sklearn import metrics

#Predict the accuracy

metrics.accuracy_score(Y_test,predict_val)*100
97.48549323017409


#confusion matrix A confusion matrix is a table that is used to define the performance of a classification algorithm. 
#A confusion matrix visualizes and summarizes the performance of a classification algorithm.

metrics.confusion_matrix(Y_test,predict_val)

array([[114,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,  80,   0,   4,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   2,   0,   0],
       [  0,   0, 101,   4,   1,   0,   0,   0,   0,   0,   0,   2,   0,
          1,   0,   0,   0],
       [  0,   1,   0, 265,   1,   0,   0,   0,   1,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   1, 189,   1,   0,   0,   1,   0,   0,   0,   0,
          0,   0,   0,   1],
       [  0,   1,   3,   0,   0,  87,   0,   0,   0,   0,   0,   0,   0,
          0,   1,   0,   0],
       [  0,   0,   0,   0,   0,   0,  66,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,  13,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   1,   0,   0,   0,   0,   0, 126,   0,   0,   1,   0,
          6,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,  70,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 121,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   1,   0,   3,   0,   0,   1,   0,   0, 148,   0,
          1,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 137,
...
          0, 134,   0,   0],
          [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0, 101,   0],
       [  0,   0,   0,   1,   0,   0,   0,   0,   1,   0,   0,   0,   0,
          0,   0,   0, 117]], dtype=int64)

#Model testing  
model_pipe.predict(["माझे नाव पूजा आहे"])

#It is predicted the value 
array(['Hindi'], dtype=object)

model_pipe.predict(["வாருங்கம் "])

array(['Tamil'], dtype=object)

import pickle
new_file=open('model.pckl','wb')
pickle.dump(model_pipe,new_file)
new_file.close()

import os

ls

Volume in drive C has no label.
 Volume Serial Number is 587A-8C5C

 Directory of C:\Users\User

01/27/2025  12:25 PM    <DIR>          .
01/27/2025  12:25 PM    <DIR>          ..
01/25/2025  01:19 PM    <DIR>          .anaconda
01/27/2025  12:27 PM    <DIR>          .conda
01/25/2025  01:20 PM               146 .condarc
01/25/2025  01:19 PM    <DIR>          .continuum
12/04/2024  03:21 PM    <DIR>          .idlerc
01/27/2025  12:22 PM    <DIR>          .ipynb_checkpoints
01/22/2025  04:17 PM    <DIR>          .ipython
01/27/2025  11:42 AM    <DIR>          .jupyter
01/13/2025  12:19 PM    <DIR>          .matplotlib
12/04/2024  03:11 PM                 7 .python_history
01/27/2025  12:25 PM    <DIR>          .streamlit
12/04/2024  02:42 PM    <DIR>          .vscode
12/07/2019  01:13 PM    <DIR>          3D Objects
01/25/2025  01:34 PM    <DIR>          anaconda_projects
01/25/2025  01:22 PM    <DIR>          anaconda3
01/27/2025  12:22 PM               425 app.py
12/07/2019  01:13 PM    <DIR>          Contacts
01/22/2025  04:14 PM    <DIR>          Desktop
...
01/27/2025  12:20 PM                 0 untitled.txt
08/05/2024  08:08 PM    <DIR>          Videos
              13 File(s)      1,604,867 bytes
              27 Dir(s)  103,582,601,216 bytes free

              
os.listdir()

['.anaconda',
 '.conda',
 '.condarc',
 '.continuum',
 '.idlerc',
 '.ipynb_checkpoints',
 '.ipython',
 '.jupyter',
 '.matplotlib',
 '.python_history',
 '.streamlit',
 '.vscode',
 '3D Objects',
 'anaconda3',
 'anaconda_projects',
 'app.py',
 'AppData',
 'Application Data',
 'Contacts',
 'Cookies',
 'Desktop',
 'Documents',
 'Downloads',
 'Favorites',
 'IntelGraphicsProfiles',
...
 'Templates',
 'Untitled.ipynb',
 'untitled.py',
 'untitled.txt',
 'Videos']



App.py

# function for predicting language


#import this libries 

import pickle                #Pickle is a Python module that converts Python objects into byte streams, which can then be stored in files or databases. 
import string                # #strings in Python are arrays of bytes representing unicode characters.
import streamlit as st       #Streamlit is an open-source Python library that makes it easy to create and share custom web apps for machine learning and data science.
import webbrowser            #webbrowser module is a convenient web browser controller. It provides a high-level interface that allows displaying Web-based documents to users. 

global Lrdetect_model
LrdetectFile=open('model.pckl','rb')
Lrdetect_model=pickle.load(LrdetectFile)
LrdetectFile.close()

st.title('Language Detection Tool')
input_test=st.text_input("Provide your text input here","Hello My name is Pooja ")

button_clicked=st.button("Get Language Name")  
if button_clicked:
  st.text(Lrdetect_model.predict([input_test]))




















