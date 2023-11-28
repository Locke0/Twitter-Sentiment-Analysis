MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Welcome to our CIS 545 Project! Introduction: Our goal of this project
was to examine public opinion and sentiments about the Ukraine War
collected between Feburary 9th to March 31st. We used a following set of
keywords to �lter the Ukraine war-related tweets: 'ukraine','ukrainian',
'russia', 'russian','putin' 'zelensky'. Also, we looked at how Twitter
in�uencers (e.g., mainstream news channels, veri�ed Twitter accounts)
are related to their followers via sentiment and NLP predictions. We
incorporated both supervised machine learning models (BERT) and
dictionary-based sentiment analyses (Lexicone) to predict sentiments
expressed in each tweet and compare similarity/difference across tweets
of different in�uence networks. We end with graph/network visualizations
to understand the structural differences among in�uential social media
in�uencers. Data Collection: We used Wharton/Annenberg Twitter Database
that provides a datset of historical Tweets, representing about 1% of
the total Twitter volume tweeted each day. By using SQL queries, we
obtained the columns that we need for our analyses. Our SQL query looked
as below:

The SQL Query using Wharton DB gives us \~1,000,000 random subseted of
tweets tweeted each day from Feb 9th - March 31st. The main columns that
we have used for our analysis was created\_at, which indicates the date
when each tweet was posted, user\_id, the IDs of each Twitter user
provided by o�cial Twitter API, text, the main text of the tweets,
user\_location, the location information provided by each Twitter user
where they lcoate at. 1 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Data Combining: From the SQL Query above we varied the dates, and
collected multilple CSV �les. We then combined all of the CVS �les into
one CVS �le in the notebook titled:
CIS545\_Combinding\_Relevant\_data.ipynb We �lter the CSV �les and
combine rows that have our key words. Key words are:
\['ukraine','ukrainian', 'russia', 'russian','putin' 'zelensky'\]
Cleaning Raw Data and Text Filtering After collecting only the Ukraine
War-related tweets from the CSV �les, we cleaned the text before working
with NLP/sentiment analyses. The text �ltering process can be found in
the notebook titled: CIS545\_prepross.ipynb.

Importing and Installing Necessary Python Packages import pandas as pd
import nltk from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np import matplotlib.pyplot as plt
nltk.download('punkt') nltk.\_\_version\_\_ from os import listdir from
os.path import isfile, join import seaborn as sns from google.colab
import drive import json !pip install tensorflow\_text \#!pip install
tensorflow-gpu==2.3.0 import tensorflow as tf import tensorflow\_hub as
hub import tensorflow\_text as text from sklearn.model\_selection import
train\_test\_split \#!pip install tf-models-official \#from official
import nlp !pip install NRCLex from nrclex import NRCLex

drive.mount('/content/drive')

2 of 69

1)  Exploratory Data Analysis (EDA) Time

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

1)  Exploratory Data Analysis (EDA) Time

df = pd.read\_json('/content/drive/MyDrive/CIS 545
project/Ukrain\_Russia\_tweets\_Feb\_Mar df \#466547 rows × 33 columns
created\_at

id

user\_id

user\_name

0

2022-02-09 18:57:04+00:00

1491486324000000000 1376954378595180544

droptown.io

3

2022-02-09 18:11:08+00:00

1491474764000000000 1170848942784995328

Alexander

5

2022-02-09 01:16:35+00:00

1491219444000000000 1174009099824128000

James Rath

7

2022-02-09 06:53:19+00:00

1491304186000000000 1473016153467506688

Limited USD (LUSD)

13

2022-02-09 06:26:19+00:00

1491297391000000000

92895963

...

...

...

...

1867809

2022-03-31 14:28:17+00:00

1509538075000000000

27567038

� Russell Hayward

1867811

2022-03-31 05:29:58+00:00

1509402603000000000

2208091680

Alex True

1867814

2022-03-31 07:41:44+00:00

1509435764000000000

300812441

Matthias Schulze

1867815

2022-03-31 18:29:32+00:00

1509598788000000000

29223860

Renee Jarrett

1867829

2022-03-31 18:01:16+00:00

1509591674000000256 1046799801256464384

Edward Aviles

Samar Hashemi

466547 rows × 33 columns

1.1) Lets check that the Tweets we collected from Wharton Twitter Data
base was equally sampled per day! 3 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Lets make sure we are sampling our tweets evenly!
=================================================

index = np.logical\_not(np.isnan(df\['day'\].unique())) days =
df\['day'\].unique()\[index\] dates = \[\] for day in days: year =
str(day)\[0:2\] month = str(day)\[2:4\] day = str(day)\[4:6\]
string\_day = str(month) + '/' + str(day) + '/' + str(year)
dates.append(string\_day) index =
np.logical\_not(np.isnan(df\['total\_day\_tweets\_collected'\].unique()))
Num\_tweets\_sampled =
df\['total\_day\_tweets\_collected'\].unique()\[index\]

plt.figure(figsize=(13,6)) sns.barplot(x = dates,y =
Num\_tweets\_sampled) plt.title('Barplot of Total number of Tweets
Sampled Per Day', fontsize = 16) plt.ylabel('Number of Tweets
Sampled',labelpad=15,fontsize = 14) plt.xlabel('Dates', labelpad = 15,
fontsize = 14) plt.xticks(rotation=90) plt.show()

From this plot we see that we have around 1.2 to 1.6 million tweets per
day! So we are 4 of 69

con�dent we are not oversampling for a given day!

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

1.2) From the Data Frame (df) we see we have geographical information!
Lets create a tokenizer to get User's locations and see where our data
is from! We expect english speaking countries due to our Query inputs!
De�ne tokenized\_locations which tokenizes user\_location from df

def tokenized\_locations(content,additional\_stop\_words = \[\]):
Token\_list = nltk.word\_tokenize(content) filtered\_list = \[\] for
word in Token\_list: word = word.lower() if not word.isalpha(): continue
filtered\_list.append(word)

return filtered\_list

import us\_state and country dictionaries and �nd what country and what
USA state each user comes from! \# Lets see where our data is comming
from!

with open('us\_state\_to\_abbrev.json') as json\_file:
us\_state\_to\_abbrev = json.load(json\_file) with
open('country\_and\_abv.json') as json\_file: country\_and\_abv =
json.load(json\_file)

lower\_names = \[x.lower() for x in us\_state\_to\_abbrev.keys()\]
lower\_abrv = \[x.lower() for x in us\_state\_to\_abbrev.values()\]
country\_lower\_names = \[x.lower() for x in
country\_and\_abv.values()\] 5 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Locations = df\['user\_location'\] countries = \[\] states = \[\] for
area in Locations: tok\_loc = tokenized\_locations(str(area)) holder = 0
for entry in tok\_loc: if entry in lower\_names: countries.append('USA')
states.append(lower\_abrv\[lower\_names.index(entry)\].upper()) holder =
1 break if entry in lower\_abrv: countries.append('USA')
states.append(lower\_abrv\[lower\_abrv.index(entry)\].upper()) holder =
1 break if entry in \['usa','us'\]: countries.append('USA') holder = 1
break if entry in country\_lower\_names:
countries.append(country\_lower\_names\[country\_lower\_names.index(entry)\])
holder = 1 break if holder == 0: countries.append('Not Found')

Some fixes, since the above approach fails with multiple word countries!
========================================================================

Locations\_df = df\[\['user\_location'\]\].copy()
Locations\_df\['filt'\] = countries Index =
Locations\_df\['user\_location'\] == 'United Kingdom'
Locations\_df\['filt'\]\[Index\] = 'united kingdom' Index =
Locations\_df\['user\_location'\] == 'United States'
Locations\_df\['filt'\]\[Index\] = 'USA' Index = Locations\_df\['filt'\]
== 'england' Locations\_df\['filt'\]\[Index\] = 'united kingdom'

Visualize Geographical Locations 6 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

import plotly import plotly.express as px from plotly.subplots import
make\_subplots

Location\_freq =
list(Locations\_df.groupby(by='filt').count().iloc\[1:,0\])
Country\_list =
list(Locations\_df.groupby(by='filt').count().iloc\[1:,0\].index)
Location\_freq\_df =
pd.DataFrame({'Location\_freq':Location\_freq,'Country':Country\_list
Location\_freq\_df

fig = px.choropleth(Location\_freq\_df,locations = 'Country'
,locationmode='country name fig.update\_layout(title\_text='Frequency of
Ukraine-Russian Tweets by Country', title\_x
fig.layout.coloraxis.colorbar.title = 'Freq.' fig.show()

/usr/local/lib/python3.7/dist-packages/distributed/config.py:20:
YAMLLoadWarning: defaults = yaml.load(f)

Frequency of Ukrain

As we can see we have collected mostly from USA! So lets see what the
map looks without USA. Also this is expected since we queried for the
English language! 7 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Remove USA!
===========

fig = px.choropleth(Location\_freq\_df.iloc\[1:,:\],locations =
'Country' ,locationmode= fig.update\_layout(title\_text='Frequency of
Ukraine-Russian Tweets by Country (U fig.layout.coloraxis.colorbar.title
= 'Freq.'

fig.show()

Frequency of Ukraine-Russian T

WOOOOOW! We still see mostly English speaking states. BUT look at
UKRAINE!! We got alot of tweets from there! This is kind of expected
since we are looking at the Ukraine-Russian War! Now lets look at the
USA states! \# Now lets look at states! States\_df =
pd.DataFrame({'states':states,'count':states}).groupby(by='states'
States\_df.reset\_index(inplace=True) 8 of 69

fig = px.choropleth(States\_df,locations = 'states'
,locationmode='USA-states', color= 11/28/23, 15:32
fig.update\_layout(title\_text='Frequency of Ukraine-Russian Tweets by
USA States

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

fig.update\_layout(title\_text='Frequency of Ukraine-Russian Tweets by
USA States fig.layout.coloraxis.colorbar.title = 'Freq.' fig.show()

Frequency of Ukraine

As expected, tweets are mostly from USA states with higher population
(e.g., California, Texas, Flordia, etc.). Now we will explore how many
tweets about the Ukraine War are tweeted per day!

1.3) Lets witness how the frequency of tweets related to UkraineRussia
con�ict vary per day!

Lets plot the Occurance of Ukraine-Russian Related Tweets for the Feb 9th - March 30
====================================================================================

day\_count = np.array(df.groupby(by='day').count()\['id'\])
plt.figure(figsize=(13, 6)) plt.plot(dates,day\_count,'o',color = 'm',
label = '*nolegend*') plt.plot(dates,day\_count,color = 'm')
plt.title('Frequency of Ukraine-Russian Related Tweets vs Date',
fontsize = 16 9 of 69 plt.ylabel('Frequency of Ukraine-Russian Related
Tweets',fontsize = 14, 11/28/23, 15:32 labelpad=

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

plt.xlabel('Date',fontsize = 14, labelpad= 15) plt.xticks(rotation = 90)
\# Rotates X-Axis Ticks by 90-degrees
plt.axvline(x=15,color='red',linestyle='dashed')
plt.legend(\['Ukraine-Russian Tweets Related','Day of Invasion'\])
plt.show() np.sum(day\_count)

466547

WOW! we see a huge increase when Russia invaded Ukraine, and it looks
like the frequency of tweets are slowly decreasing as we move away from
the invasion date!

1.4) From our data lets witness the top 150 words, and visualize the
words in a wordcloud! Download stop words as well as
twitter\_stop\_words! Create new function tokenized\_content which will
tokenize the Filtered\_extended\_text in df! \# Lets discover what words
are the most common! from nltk.corpus import stopwords
nltk.download('stopwords')

10 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

nltk.download('stopwords') stopwords = set(stopwords.words('english'))
twitter\_txt = open("twitter\_stop\_words.txt", "r") twitter\_stop =
\[\] for word in twitter\_txt: twitter\_stop.append(word.strip())
\[nltk\_data\] Downloading package stopwords to /root/nltk\_data...
\[nltk\_data\] Unzipping corpora/stopwords.zip.

Tokenizer here:

def tokenized\_content(content,additional\_stop\_words = \[\]):
Token\_list = nltk.word\_tokenize(content) filtered\_list = \[\] for
word in Token\_list: word = word.lower() if word in stopwords: continue
if not word.isalpha(): continue if word in twitter\_stop: continue if
word in additional\_stop\_words: continue filtered\_list.append(word)

return filtered\_list

Create a list called top\_tokens where we have a list of all relavent
words from our data! We also generated a list of additional stop words
to remove unnecessary texts that do not help us with our analyses (e.g.,
rt, https, says).

top\_tokens = \[\] additional\_stop\_words =
\['rt','https','says','one','would','nan','get','new','said'
Word\_series = df\['extended\_full\_text'\] for string in Word\_series:
11 of 69 11/28/23, 15:32 filtered\_list =
tokenized\_content(str(string),additional\_stop\_words = additional\_sto

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N... filtered\_list =
tokenized\_content(str(string),additional\_stop\_words = additional\_sto
for word in filtered\_list: top\_tokens.append(word)

Find the most frequent words and use wordcloud to visualize! We do this
by counting the frequency of words that appear in top\_tokens! Remember
top\_tokens is a list of all the tokenized words from all the �ltered
tweets we collected! import collections top\_most\_common\_counter =
collections.Counter() for word in top\_tokens:
top\_most\_common\_counter\[word\] += 1 from wordcloud import WordCloud
top\_100 = top\_most\_common\_counter.most\_common(150) wc\_top =
WordCloud(background\_color='black',height=1000,width=1400, max\_words=
wc\_top.generate\_from\_frequencies(dict(top\_100))
plt.figure(figsize=(13, 6)) plt.imshow(wc\_top) plt.title('Top 150 words
from Ukraine-Russian Related Tweets', fontsize = 16,pad =
plt.axis('off') plt.show()

COOOL! We see that the top words are Ukraine, Russia (or relating to
their respective 12 of 69

nationalities). Moreover, we see intresting words such as help, trump,
putin, oil,

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

nationalities). Moreover, we see intresting words such as help, trump,
putin, oil, humanitarism, stopputin, nuclear, china, money! Now lets see
how the top 20 words vary per day!

1.5) Using the top 150 words, lets witness how the top 20 words varied
per day from Feb 9th- March 31st! Once again we tokenize the text and
count how each of the 20 words vary per day: \# From the top 20, lets
see how the top 20 words varied per day! top\_20 =
dict(top\_most\_common\_counter.most\_common(20)) list\_top\_20 =
top\_20.keys() dict\_top\_20\_per\_day = {} \#print(list\_top\_20) for
candidate in list\_top\_20: dict\_top\_20\_per\_day\[candidate\] = \[\]

for candidate in list\_top\_20: \#print(candidate) for day in days:
index = df\['day'\] == day day\_df = df\[index\]

extended\_text = day\_df\['extended\_full\_text'\] counter = 0 for
string in extended\_text: filtered\_list =
tokenized\_content(str(string),additional\_stop\_words = additional for
word in filtered\_list: if candidate == word: counter += 1
dict\_top\_20\_per\_day\[candidate\].append(counter)

plt.figure(figsize=(13, 6)) colors =
\['b','g','r','c','m','y','k','cyan','\#68228B','\#FFE4B5','\#9AFF9A','\#33A1C9'
counter = 0 for i in dict\_top\_20\_per\_day.keys(): color =
colors\[counter\] valz = dict\_top\_20\_per\_day\[i\] 13 of 69
plt.plot(dates,valz,color = color)

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

plt.plot(dates,valz,'o',color =color,label='*nolegend*') counter +=1
plt.legend(list(dict\_top\_20\_per\_day.keys()),loc=(1.01,.02),title =
'Buzz Words' plt.title('Frequency of Top 20 Ukraine-Russian Buzz Words
vs Date', fontsize = plt.ylabel('Frequency of Top 20 Ukraine-Russian
Buzz Words',fontsize = 14, labelpad= plt.xlabel('Date',fontsize = 14,
labelpad= 15) plt.xticks(rotation = 90) plt.show()

Above is a plot that visualizes the top 20 frequent words (i.e., 'buzz
words') that have appeared in the twitter dataset we have analyzed so
far. In line with the previous frequency plot we have generated, the
buzz words appear much more frequently when the war actually begins. We
can conlcude that each word follows a similar trend, however the peak
differs. END OF EDA

Starting the Model Process

2)  Intercoder Reliability 14 of 69

What is Intercoder Reliability

11/28/23, 15:32

MAIN.ipynb - Colaboratory

What is Intercoder Reliability

https://colab.research.google.com/drive/1NQq5N...

Before we predict the opinions/sentiments of each tweet using BERT
models, we begin with getting a manually labeled dataset of each opinion
and sentiment we are trying to predict. This manually labeled dataset is
based on individual labeling by each of our team member. Before labeling
a large share of tweets, we �rst tested whether our team comes up with a
good score of intercoder reliability, that is, whether us three labels
important variables we are trying to predict in the same way. To test
the inter-rater reliability, we randomly sampled 50 tweets from the
whole dataset, labeled each of the variable individually, and then
compared whether we labeled every variable in the same direction. Please
look at notebook: CIS5454\_Intercoder\_reliability.ipynb to see how we
got the Intercoder\_reliability csv! We explored total 11 variables to
estimate the opinions and sentiments surrounding the Ukraine War. We
took each tweet as a face value. 1) War\_related. If a tweet is related
ot the Ukraine War (i.e., the Russian invasion of Ukraine since February
2022), then 1; otherwise 0. If the tweet does ont directly mention the
war/invasion (or something related to this con�ict), it should be
counted as 0. 2) Pro\_ukraine. If a tweet expresses positive opinions or
sentiments toward Ukraine, Zelensky (or Ukrainian political elites),
Ukrainian people, or anything related to Ukraine, then 1; otherwise 0.
If a tweet only delivers certain facts about what is happening in
Ukraine/Russia, not particular opinions or subjective evaluations about
the war, it should be coded as 0. Also, the same tweet can be both
pro-Ukraine and pro-Russian. 3) anti\_ukraine. If a tweet expresses
negative opinions or sentiments toward Ukraine, Zelensky (or Ukrainian
political elites), Ukrainian people, or anything related to Ukraine,
then 1; otherwise 0. If a tweet only delivers certain facts about what
is happening in Ukraine/Russia, not particular opinions or subjective
evaluations about the war, it should be coded as 0. Also, the same tweet
can be both pro-Ukraine and pro-Russian. 4) pro\_russia. If a tweet
expresses positive opinions or sentiments toward Russia, Putin (or
Russian political elites), Russian people, or anything related to
Russia, then 1; otherwise 0. 5) anti\_russia. If a tweet expresses
negative opinions or sentiments toward Russia, Putin (or Russian
political elites), Russian people, or anything related to Russia, then
1; otherwise 0. 6) pro\_involve. If a tweet expresses support for
Western countries' or organizations' involvement in the Ukraine War
(e.g., 'NATO should more intervene in Ukraine,' 'more countries should
close the sky above Ukraine,' 'European countries should provide more
help to Ukraine'), then 1; otherwise 0. If the tweet does not mention
anything about other countries' or organizations' involvement in the
war, it should be coded as 0. 7) anti\_involve. If a tweet expresses
opposition to Western countries' or organizations' 15 of 69involvement
in the Ukraine War, then 1; otherwise 0.

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

8)  pro\_democrate. If a tweet expresses positive opinions or sentiments
    toward the US Democratic Party, Biden (or any other Democratic Party
    politicians), Biden's decisions, Democratic Party's policies,
    Democrats, or anything related to the US Democratic Party, then 1;
    otherwise 0. If a tweet is only saying something positive about the
    Republican Party but says nothing about the Democratic Party, it
    should be coded as 0. If a tweet only delivers certain facts about
    what the Democratic Party or Biden has done, and does not carry
    particular opinions or subjective evaluations about what they did,
    it should be coded as 0.
9)  anti\_democrate. If a tweet expresses negative opinions or
    sentiments toward the US Democratic Party, Biden (or any other
    Democratic Party politicians), Biden's decisions, Democratic Party's
    policies, Democrats, or anything related to the US Democratic Party,
    then 1; otherwise 0.
10) pro\_republican. If a tweet expresses positive opinions or
    sentiments toward the US Republican Party, Trump (or any other
    Republican Party politicians), Trump's decisions, Republican Party's
    policies, Republicans, or anything related to the US Republican
    Party, then 1; otherwise 0.
11) anti\_republican. If a tweet expresses negative opinions or
    sentiments toward the US Republican Party, Trump (or any other
    Republican Party politicians), Trump's decisions, Republican Party's
    policies, Republicans, or anything related to the US Republican
    Party, then 1; otherwise 0. Below, we provide how we calculated the
    intercoder reliability based on Krippendorf's alpha, which is
    largely used in social sciences to measure inter-rater reliability
    of manual coded variables. The code to calculate the Krippendorf's
    was done in R: please see R script intercoder\_reliability\_script.R
    if intrested or want to see the code in more detail.

16 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Results from Krippendorf's Alpha:

We see from the Krippendorf's Alpha we get high accuracies (above 80%)
for each label! However, lets visualize the differences in our
intercoder reliability for each person! This is done below: 1) Below we
load in each intercoder 17 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

2)  Plot each intercoder as a 50 x 11 heat map: 50 is the number of
    tweets and 11 is the labels for the respective tweet!
3)  Subtract Chloe Intercoder - .5(Rohan Intercoder + Locke Intercoder)
    and plot it as Differences of Intercoder \# Model Training
    Intercoder Reliabliliy Chloe\_intercoder =
    np.genfromtxt('labels\_chloe.csv',delimiter=',',skip\_header=
    Rohan\_intercoder =
    np.genfromtxt('labels\_rohan.csv',delimiter=',',skip\_header=
    Locke\_intercoder =
    np.genfromtxt('labels\_locke.csv',delimiter=',',skip\_header=

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(15,10)) map1 =
ax1.imshow(Chloe\_intercoder, cmap = 'Reds') ax1.set\_title('Chloe
Intercoder Labels',fontsize = 16, pad = 10)
ax1.set\_ylabel('Samples',fontsize = 14, labelpad = 10)
ax1.set\_xlabel('Labels',fontsize = 14,labelpad=10)
ax1.set\_xticks(\[\]) ax1.set\_yticks(\[\]) map2 =
ax2.imshow(Rohan\_intercoder, cmap = 'Reds') ax2.set\_title('Rohan
Intercoder Labels',fontsize = 16, pad = 10)
ax2.set\_ylabel('Samples',fontsize = 14, labelpad = 10)
ax2.set\_xlabel('Labels',fontsize = 14,labelpad=10)
ax2.set\_xticks(\[\]) ax2.set\_yticks(\[\]) map3 =
ax3.imshow(Locke\_intercoder, cmap = 'Reds') ax3.set\_title('Locke
Intercoder Labels',fontsize = 16, pad = 10)
ax3.set\_ylabel('Samples',fontsize = 14, labelpad = 10)
ax3.set\_xlabel('Labels',fontsize = 14,labelpad=10)
ax3.set\_xticks(\[\]) ax3.set\_yticks(\[\])

Difference\_matrix = (Chloe\_intercoder - .5\*(Rohan\_intercoder +
Locke\_intercoder map4 = ax4.imshow(Difference\_matrix, cmap = 'Reds')
ax4.set\_title('Differences of Intercoder',fontsize = 16, pad = 10)
ax4.set\_ylabel('Samples',fontsize = 14, labelpad = 10)
ax4.set\_xlabel('Labels',fontsize = 14,labelpad=10)
ax4.set\_xticks(\[\]) ax4.set\_yticks(\[\])

fig.colorbar(map1, ax = ax1) fig.colorbar(map2, ax = ax2)
fig.colorbar(map3, ax = ax3) fig.colorbar(map4, ax = ax4) 18 of 69

plt.show()

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

plt.show()

So we can see each of our intercoder heatmaps here! But we get most of
our information from the Differences of Intercoder heatmap! Since we did
Chloe - .5(Rohan + Locke) we know these rules: A) 0 are where we all
Agree! And this is the Majority :) B) 1 or -1 is where Locke and Rohan
agree and Chloe disagrees C) .5 or -.5 is where Locke and Rohan disagree
and either one of us agree with Chloe

3)  TIME TO TRAIN OUR BERT MODELS! Lets load in our dataset where we
    have 11 labels as described in the Intercoder Reliability section:

4)  War\_related 19 of 69

5)  Pro\_ukraine

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

2)  Pro\_ukraine
3)  anti\_ukraine
4)  pro\_russia
5)  anti\_russia
6)  pro\_involve
7)  anti\_involve
8)  pro\_democrate
9)  anti\_democrate
10) pro\_republican
11) anti\_republican

3.1) Initial Bert Model based on our labeled Tweets Load in training Now
that we know the intercoder is good! We created a Training/Testing
Dataset of 2400 random tweets and we manually labeled them! To see how
we created the random dataset please look at
CIS5454\_Intercoder\_reliability.ipynb sections training/testing and
down! df\_train = pd.read\_csv('TRAINING\_TEST\_Labels\_real.csv')
df\_train Filtered\_extended\_text

war\_related

pro\_ukraine

anti\_ukraine

0

funny Russia dont speak much just action

0

0

0

1

Leaving Teacher retirement investments to a Ru...

0

0

0

2

\#SuperHotDeal - EE Sim Plans - Free Calls, Tex...

0

0

0

3

\#Russia issues threat to \#Donbass ceaseﬁre vi...

1

0

0

4

Yeah... when? We've been hearing this shit for...

1

0

0

...

...

...

...

...

"The evacuation vehicle sent to Volnovakha in ...

1

1

0

2405 20 of 69

I propose we shame them

pro\_russia

11/28/23, 15:32

MAIN.ipynb - Colaboratory

2406

https://colab.research.google.com/drive/1NQq5N...

I propose we shame them in public at every opp...

1

1

0

\#WorldWarIII

Split our training/testing dataset into training and testing! We use
test 2407 \#UkraineRussiaCrisis 1 1 size of 20% 0 PEOPLES CAN ...
X\_train, X\_test, Justy\_train, like those y\_test 13( 82 in=
train\_test\_split(df\_train\['Filtered\_extended\_text' 2408 1 0 0
reality) soldiers fr... Russian tanks stuck in 2409 1 by Plotting 0 out
our Class 0 Check the quality of the data we have labeled

Balances. \# Lets check class balances Training\_balance =
dict(y\_train.sum()/len(y\_train)) Testing\_balance =
dict(y\_test.sum()/len(y\_test)) f, (ax1, ax2) = plt.subplots(1,
2,figsize=(15,5))
ax1.bar(Training\_balance.keys(),Training\_balance.values())
ax1.set\_xticklabels(Training\_balance.keys(), rotation=90, fontsize =
12) ax1.set\_title('Class Balances for Training',fontsize = 16, pad =
10) ax1.set\_xlabel('Labels',fontsize = 14,labelpad=10)
ax1.set\_ylabel('Percentage',fontsize = 14, labelpad = 10)

ax2.bar(Testing\_balance.keys(),Testing\_balance.values())
ax2.set\_xticklabels(Testing\_balance.keys(), rotation=90, fontsize =
12) ax2.set\_title('Class Balances for Test',fontsize = 16, pad = 10)
ax2.set\_xlabel('Labels',fontsize = 14,labelpad=10)
ax2.set\_ylabel('Percentage',fontsize = 14, labelpad = 10)

plt.show()

21 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

As the two �gures above demonstrate, we observe a huge class imbalance.
For example, there are much less tweets related to the
Democratic/Republican Party or contain AntiUkraine or Pro-Russian
opinions compared to others. In order to deal with this imabalance, we
tried removing labels with low percentages, and sub-sample from high
percentages!

Corrected Class Imbalances by Removing Labels with Low Percentages and
sub\_sampling from High Percentages Correct for class imbalance by
removing labels with high majority, and subsetting labels with
majorities labels\_to\_keep =
\['Filtered\_extended\_text','pro\_ukraine','anti\_ukraine','pro\_russia'
df\_train\_update = df\_train\[df\_train\['war\_related'\] ==
1\]\[labels\_to\_keep\] Min\_samples =
min(df\_train\_update.iloc\[:,1:6\].sum())

df\_class\_balanced = pd.DataFrame() for label in labels\_to\_keep: if
label == 'Filtered\_extended\_text': continue index =
df\_train\_update\[label\] == 1 df\_train\_update\[index\]
df\_class\_balanced =
df\_class\_balanced.append(df\_train\_update\[index\].sample(Min\_sampl
index = df\_train\_update\[label\] == 0 df\_class\_balanced =
df\_class\_balanced.append(df\_train\_update\[index\].sample(Min\_sampl
remove\_n = 150 drop\_indices1 =
np.random.choice(df\_class\_balanced\[df\_class\_balanced\['pro\_ukraine'
drop\_indices2 =
np.random.choice(df\_class\_balanced\[df\_class\_balanced\['anti\_russia'
drop\_indices = np.concatenate((drop\_indices1,drop\_indices2))
df\_subset = df\_class\_balanced.drop(drop\_indices)

X\_train, X\_test, y\_train, y\_test =
train\_test\_split(df\_subset\['Filtered\_extended\_text'

Recheck Balances \# Lets re - check class balances Training\_balance =
dict(y\_train.sum()/len(y\_train)) Testing\_balance =
dict(y\_test.sum()/len(y\_test)) 22 of 69

f

(ax1

ax2) = plt.subplots(1

2 figsize=(15 5))

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

f, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))
ax1.bar(Training\_balance.keys(),Training\_balance.values())
ax1.set\_xticklabels(Training\_balance.keys(), rotation=90, fontsize =
12) ax1.set\_title('New Class Balances for Training',fontsize = 16, pad
= 10) ax1.set\_xlabel('Labels',fontsize = 14,labelpad=10)
ax1.set\_ylabel('Percentage',fontsize = 14, labelpad = 10)

ax2.bar(Testing\_balance.keys(),Testing\_balance.values())
ax2.set\_xticklabels(Testing\_balance.keys(), rotation=90, fontsize =
12) ax2.set\_title('New Class Balances for Test',fontsize = 16, pad =
10) ax2.set\_xlabel('Labels',fontsize = 14,labelpad=10)
ax2.set\_ylabel('Percentage',fontsize = 14, labelpad = 10)

plt.show()

Ahhh!!! Much more balanced! However we did lose a lot of the data we had
previously! Initially we had a training size of 1920, but now we have
around 700. We lose a lot of data in order to balance.

3.2) Constructing our Initial Bert Model Build Text Tokenizer and
Prepocesser function 23 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Buling the Preprocesser A preprocessor converts raw text to the numeric
input tensors expected by the encoder. The encoder then interprets the
vectorized raw text, and is able to perform operations on it!
tfhub\_handle\_preprocess =
"https://tfhub.dev/tensorflow/bert\_en\_cased\_preprocess/3" def
make\_bert\_preprocess\_model(sentence\_features, seq\_length= 310):
\"\""Returns Model mapping string features to BERT inputs. Args:
sentence\_features: a list with the names of string-valued features.
seq\_length: an integer that defines the sequence length of BERT inputs.
Returns: A Keras Model that can be called on a list or dict of string
Tensors (with the order or names, resp., given by sentence\_features)
and returns a dict of tensors for input to BERT."\"\" input\_segments =
\[ tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft) for ft in
sentence\_features\] \# Tokenize the text to word pieces.
bert\_preprocess = hub.load(tfhub\_handle\_preprocess) tokenizer =
hub.KerasLayer(bert\_preprocess.tokenize, name='tokenizer') segments =
\[tokenizer(s) for s in input\_segments\] \# Optional: Trim segments in
a smart way to fit seq\_length. \# Simple cases (like this example) can
skip this step and let \# the next step apply a default truncation to
approximately equal lengths. truncated\_segments = segments \# Pack
inputs. The details (start/end token ids, dict of output tensors) \# are
model-dependent, so this gets loaded from the SavedModel. packer =
hub.KerasLayer(bert\_preprocess.bert\_pack\_inputs,
arguments=dict(seq\_length=seq\_length), name='packer') model\_inputs =
packer(truncated\_segments) return tf.keras.Model(input\_segments,
model\_inputs)

Call Preprocesser and Encoder from Tensor�owHub bert\_preprocess =
make\_bert\_preprocess\_model(\['my\_input1'\]) bert\_encoder =
hub.KerasLayer(\"https://tfhub.dev/tensorflow/bert\_en\_cased\_L-12\_H

Build Initial Bert Model 24 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

text\_input = tf.keras.layers.Input(shape=(), dtype=tf.string,
name='text') preprocessed\_text = bert\_preprocess(text\_input) outputs
= bert\_encoder(preprocessed\_text) \# Neural network layers l =
tf.keras.layers.Dropout(0.01,
name="dropout")(outputs\['pooled\_output'\]) l =
tf.keras.layers.Dense(6, activation='softmax', name="output")(l) \# Use
inputs and outputs to construct a final model model =
tf.keras.Model(inputs=\[text\_input\], outputs = \[l\])

Here we bulid our simple model! We have the input being raw text --\>
vectorized by the bert processor --\> transformed by the encoder --\>
sent into the Bert Layers (Transfer learning) --\> 1% dropout --\>
classi�cation nodes! Note we also use Adam optimizer, and Softmax
activation because we are multiclassi�cation. For loss we use
categorical\_crossentropy!

Train and evaluate model.compile(optimizer='adam',
loss='categorical\_crossentropy', metrics=\['accuracy'\]) hist =
model.fit(X\_train, y\_train, epochs=20)

25 of 69

Epoch 1/20 20/20 \[==============================\] - 35s 2s/step -
loss: 2.4532 - accuracy: 0 Epoch 2/20 20/20
\[==============================\] - 31s 2s/step - loss: 2.2439 -
accuracy: 0 Epoch 3/20 20/20 \[==============================\] - 31s
2s/step - loss: 2.2194 - accuracy: 0 Epoch 4/20 20/20
\[==============================\] - 31s 2s/step - loss: 2.2056 -
accuracy: 0 Epoch 5/20 20/20 \[==============================\] - 31s
2s/step - loss: 2.1950 - accuracy: 0 Epoch 6/20 20/20
\[==============================\] - 31s 2s/step - loss: 2.2107 -
accuracy: 0 Epoch 7/20 20/20 \[==============================\] - 31s
2s/step - loss: 2.2172 - accuracy: 0 Epoch 8/20 20/20
\[==============================\] - 31s 2s/step - loss: 2.1754 -
accuracy: 0 Epoch 9/20 20/20 \[==============================\] - 31s
2s/step - loss: 2.1664 - accuracy: 0 Epoch 10/20 20/20
\[==============================\] - 31s 2s/step - loss: 2.1480 -
accuracy: 0 Epoch 11/20 20/20 \[==============================\] - 31s
2s/step - loss: 2.1535 - accuracy: 0 Epoch 12/20 20/20
\[==============================\] - 31s 2s/step - loss: 2.1449 -
accuracy: 11/28/23, 15:32 0

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Epoch 13/20 20/20 \[==============================\] - 31s 2s/step -
loss: 2.1294 - accuracy: 0 Epoch 14/20 20/20
\[==============================\] - 31s 2s/step - loss: 2.1695 -
accuracy: 0 Epoch 15/20 20/20 \[==============================\] - 31s
2s/step - loss: 2.1401 - accuracy: 0 Epoch 16/20 20/20
\[==============================\] - 31s 2s/step - loss: 2.1376 -
accuracy: 0 Epoch 17/20 20/20 \[==============================\] - 31s
2s/step - loss: 2.1120 - accuracy: 0 Epoch 18/20 20/20
\[==============================\] - 31s 2s/step - loss: 2.1096 -
accuracy: 0 Epoch 19/20 20/20 \[==============================\] - 31s
2s/step - loss: 2.1326 - accuracy: 0 Epoch 20/20 20/20
\[==============================\] - 31s 2s/step - loss: 2.1085 -
accuracy: 0

Visualize Accuracy and Loss Initial\_model\_acc =
hist.history\['accuracy'\] Initial\_model\_loss = hist.history\['loss'\]

f, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))
ax1.plot(Initial\_model\_acc, color = 'g')
ax1.plot(Initial\_model\_acc,'go') ax1.set\_title('Accuracy vs Epochs
for Initial Model',fontsize = 16, pad = 10)
ax1.set\_xlabel('Epochs',fontsize = 14,labelpad=10)
ax1.set\_ylabel('Accuracy',fontsize = 14, labelpad = 10)

ax2.plot(Initial\_model\_loss, color = 'r')
ax2.plot(Initial\_model\_loss,'ro') ax2.set\_title('Loss vs Epochs for
Initial Model',fontsize = 16, pad = 10)
ax2.set\_xlabel('Epochs',fontsize = 14,labelpad=10)
ax2.set\_ylabel('Loss',fontsize = 14, labelpad = 10)

plt.show()

26 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

NOT GOOD due to the class imbalance we lost most of our training data,
and our model can not learn! model.evaluate(X\_test,y\_test) \# The
evaluation support the previous claims

5/5 \[==============================\] - 9s 2s/step - loss: 2.2410 -
accuracy: 0.28 \[2.2410428524017334, 0.2848101258277893\]

Lets plot a Confusion Matrix to see what our model is predictiing \#
Confusion Matrix y\_predicted = pd.DataFrame(model.predict(X\_test))
y\_predicted\[0\] = y\_predicted\[0\].apply(lambda x: 1 if x \> (1/6)
else 0) y\_predicted\[1\] = y\_predicted\[1\].apply(lambda x: 2 if x \>
(1/6) else 0) y\_predicted\[2\] = y\_predicted\[2\].apply(lambda x: 3 if
x \> (1/6) else 0) y\_predicted\[3\] = y\_predicted\[3\].apply(lambda x:
4 if x \> (1/6) else 0) y\_predicted\[4\] =
y\_predicted\[4\].apply(lambda x: 5 if x \> (1/6) else 0)
y\_predicted\[5\] = y\_predicted\[5\].apply(lambda x: 6 if x \> (1/6)
else 0) y\_predicted\_flat = np.array(y\_predicted).flatten()
y\_test\['pro\_ukraine'\] = y\_test\['pro\_ukraine'\].apply(lambda x: 1
if x \> (1/6) else y\_test\['anti\_ukraine'\]
=y\_test\['anti\_ukraine'\].apply(lambda x: 2 if x \> (1/6) else
y\_test\['pro\_russia'\] =y\_test\['pro\_russia'\].apply(lambda x: 3 if
x \> (1/6) else y\_test\['anti\_russia'\]
=y\_test\['anti\_russia'\].apply(lambda x: 4 if x \> (1/6) else
y\_test\['pro\_involve'\] =y\_test\['pro\_involve'\].apply(lambda x: 5
if x \> (1/6) else y\_test\['anti\_involve'\]
=y\_test\['anti\_involve'\].apply(lambda x: 6 if x \> (1/6) else
y\_test\_flat = np.array(y\_test).flatten()

from sklearn.metrics import confusion\_matrix labels =
\['no\_label','pro\_ukraine','anti\_ukraine','pro\_russia','anti\_russia','pro\_invo
CM\_\_bert\_initial = pd.DataFrame(confusion\_matrix(y\_test\_flat,
y\_predicted\_flat plt.figure(figsize=(13,10))
sns.heatmap(CM\_\_bert\_initial,cmap = 'Reds',annot=False) \# Note I do
not want to use annot, makes the graph harder to understand
plt.title('Confusion Matrix of Initial BERT Model',fontsize = 16, pad =
20) plt.xlabel('Predicted Label From BERT',labelpad=15,fontsize = 14)
plt.ylabel('Ground Truth Label',labelpad=15,fontsize = 14) plt.show() 27
of 69 11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Looks like the model learns to just guess zero! Or that there is no
label. This can be due to having a class imbalance of no labels for
majority of the text, or the text are neutral! If We want to improve
this model we can: 1) Get more data 2) Make sure our classes are truely
balanced 3) Optimize the Model Architectural 4) Adjust the weights of
the model to accomodate the class imabalances 5) Train for some more
epochs (this is not garenteed to improve the model) 6) use other
optimizers (not garenteed to improve) 28 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

3.3) New Model based on positive and negative labeled Tweets Since our
previous model did not work due to class imbalances and a limited data
set we sought out to �nd a new data set We came across a Kaggle Dataset
for Labeled Positive and Negative tweets:
https://www.kaggle.com/datasets/kazanova/sentiment140 Using this Dataset
we want to train a model and then evaluate on our data re-labeled as
Positive or Negative and evaluate the generalizability of the model! We
believe this will work �ne for our data because we have seen that
Positive tweets in our data set are symomous for Defending Ukraine, and
negative tweets are for Not defending Ukraine. Load in Kaggle Data

df\_train =
pd.read\_csv('training.1600000.processed.noemoticon.csv',encoding='ISO-8859-

Split Training and Testing Note we also only use 5% of the data! This is
because this data set has 1.6 million tweets. We dont wanna train
forever! sample\_size = int(len(df\_train)\*0.05) sampleDf =
df\_train.sample(sample\_size, random\_state=23) sampleDf\['positive'\]
= sampleDf\['target'\].apply(lambda x: 1 if x == 4 else 0)
sampleDf\['negative'\] = sampleDf\['target'\].apply(lambda x: 1 if x ==
0 else 0) x\_train, x\_test, y\_train, y\_test =
train\_test\_split(sampleDf\['text'\],sampleDf\[\[

Lets check class balances \# Lets check class balances Training\_balance
= dict(y\_train.sum()/len(y\_train)) Testing\_balance =
dict(y\_test.sum()/len(y\_test)) f, (ax1, ax2) = plt.subplots(1,
2,figsize=(15,5))
ax1.bar(Training\_balance.keys(),Training\_balance.values())
ax1.set\_xticklabels(Training\_balance.keys(), rotation=90, fontsize =
12) ax1.set\_title('Class Balances for Training',fontsize = 16, pad =
10) ax1.set\_xlabel('Labels',fontsize = 14,labelpad=10)
ax1.set\_ylabel('Percentage',fontsize = 14, labelpad = 10) 29 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

ax2.bar(Testing\_balance.keys(),Testing\_balance.values())
ax2.set\_xticklabels(Testing\_balance.keys(), rotation=90, fontsize =
12) ax2.set\_title('Class Balances for Test',fontsize = 16, pad = 10)
ax2.set\_xlabel('Labels',fontsize = 14,labelpad=10)
ax2.set\_ylabel('Percentage',fontsize = 14, labelpad = 10)

plt.show()

Much more Balanced! We dont need to intervine! Build Model text\_input =
tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed\_text = bert\_preprocess(text\_input) outputs =
bert\_encoder(preprocessed\_text) \# Neural network layers l =
tf.keras.layers.Dropout(0.01,
name="dropout")(outputs\['pooled\_output'\]) l =
tf.keras.layers.Dense(2, activation='softmax', name="output")(l) \# Use
inputs and outputs to construct a final model model\_pos\_neg =
tf.keras.Model(inputs=\[text\_input\], outputs = \[l\])

Train and Evaluate model\_pos\_neg.compile(optimizer='adam',
loss='categorical\_crossentropy'

30 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...
loss='categorical\_crossentropy', metrics=\['accuracy'\])

epochs = 3 batch\_size = 16 history = model\_pos\_neg.fit(x\_train,
y\_train, epochs=epochs, batch\_size = 16, validation\_data=(x\_test,
y\_test), verbose=1)

Save the model so we can use it again!
======================================

\#model\_pos\_neg.save("/content/drive/MyDrive/CIS 545
project/twitter\_BERT\_rohan")

Epoch 1/3 4000/4000 \[==============================\] - 2152s
535ms/step - loss: 0.5863 - ac Epoch 2/3 4000/4000
\[==============================\] - 2141s 535ms/step - loss: 0.5504 -
ac Epoch 3/3 4000/4000 \[==============================\] - 2120s
530ms/step - loss: 0.5423 - ac

Visualize Accuracy and Loss New\_model\_acc =
history.history\['accuracy'\] New\_model\_loss =
history.history\['loss'\]

f, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))
ax1.plot(New\_model\_acc, color = 'g') ax1.plot(New\_model\_acc,'go')
ax1.set\_title('Accuracy vs Epochs for New BERT Model',fontsize = 16,
pad = 10) ax1.set\_xlabel('Epochs',fontsize = 14,labelpad=10)
ax1.set\_ylabel('Accuracy',fontsize = 14, labelpad = 10)

ax2.plot(New\_model\_loss, color = 'r') ax2.plot(New\_model\_loss,'ro')
ax2.set\_title('Loss vs Epochs for New BERT Model',fontsize = 16, pad =
10) ax2.set\_xlabel('Epochs',fontsize = 14,labelpad=10)
ax2.set\_ylabel('Loss',fontsize = 14, labelpad = 10)

plt.show()

31 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Evaluate the New Model on its Testing and Training Set \# evaluate on
this data set model\_pos\_neg.evaluate(x\_test,y\_test)

500/500 \[==============================\] - 422s 843ms/step - loss:
0.5359 - accur \[0.5359355211257935, 0.7329375147819519\]

Note if we want to improve this model we can 1) optimize the bert
structure by potentially adding more dense layers 2) Increase the data
we use (right now its 5%) 3) Train for more Epochs (This is not
garenteed to improve) 4) Get more data 5) use other optimizers (not
garenteed to improve) NOW Evaluate the New Model on our RE-labeled Data
Load in RE-labeled Data \# evaluate on our data set with re-labels!
df\_evaluate =
pd.read\_csv('TRAINING\_TEST\_Labels\_NEW\_MODEL\_eval.csv') X\_train =
df\_evaluate\['Filtered\_extended\_text'\] Y\_eval =
df\_evaluate.iloc\[:,\[1,2\]\] df\_evaluate

0 32 of 69

1

Filtered\_extended\_text

Defending Ukraine (Positive)

Negative (Not Defending Ukraine)

funny Russia dont speak much just action

0

1

Leaving Teacher retirement

0

11/28/23, 15:32

1

MAIN.ipynb - Colaboratory

1

Leaving Teacher retirement investments to a Ru...

2

https://colab.research.google.com/drive/1NQq5N...

0

1

\#SuperHotDeal - EE Sim Plans Free Calls, Tex...

0

1

3

\#Russia issues threat to \#Donbass ceaseﬁre vi...

1

0

4

Yeah... when? We've been hearing this shit for...

0

1

...

...

...

...

"The evacuation vehicle sent to Volnovakha in ...

1

0

1

0

2405

2406

I propose we shame them in public

Evaluate \# Eval model\_pos\_neg.evaluate(X\_train,Y\_eval)

76/76 \[==============================\] - 65s 844ms/step - loss: 0.5112
- accuracy \[0.5111690163612366, 0.8082987666130066\]

WOW we get approx 80% accuracy! This leads us to believe the model is
generalizable to our data! To be even more sure lets plot a Confusion
Matrix! y\_pred = model\_pos\_neg.predict(X\_train)

from sklearn.metrics import confusion\_matrix y\_pred\_df =
pd.DataFrame(y\_pred) y\_pred\_df\[0\] = y\_pred\_df\[0\].apply(lambda
x: 1 if x \> .5 else 0) y\_pred\_flatt =
np.array(y\_pred\_df\[0\]).flatten() ground\_truth\_flat =
np.array(df\_evaluate\['Defending Ukraine (Positive)'\]).flatten labels
= \['(Positive) Defending Ukraine', '(Negative) Not Defending Ukraine'\]
CM\_\_bert\_new = pd.DataFrame(confusion\_matrix(ground\_truth\_flat,
y\_pred\_flatt), plt.figure(figsize=(13,10))
sns.heatmap(CM\_\_bert\_new,cmap = 'Reds',annot=True) \# Note I do not
want to use annot, makes the graph harder to understand
plt.title('Confusion Matrix of New BERT Model',fontsize = 16, pad = 20)
plt.xlabel('Predicted Label From BERT',labelpad=15,fontsize = 14)
plt.ylabel('Ground Truth Label',labelpad=15,fontsize = 14) 33 of
69plt.show() 11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Amazing From the Confusion Matrix we can see that we get a strong
diagonal, meaning that the predictions are matching with the ground
truth! We do have a small False positive and decent Flase Negative rate,
but for our applications it is tolerable!

3)  Lexicone approach over than Bert due to Neutral Label Even though
    our Bert Model is generalizable to our data set, we noticed the
    training labels are binary, we only get Positive (Defending Ukraine)
    or Negative (Not Defending Ukraine). It is largely due to the fact
    that we have labeled each variable as 0/1, and we have trained BERT
    models based on logistic regression classi�ers. 34 of 69

11/28/23, 15:32

Nonetheless, we realized that there were quite many neutral tweets like
news reports, factual

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Nonetheless, we realized that there were quite many neutral tweets like
news reports, factual statements, or breaking news about what is
happening in Ukraine/Russia. These tweets were largely left intact in
our BERT models as we have not made a distinct category of neutral
opinions. Since we noticed many nutrual tweets while labeling, we
decided to go for a Lexicone Approach. The Lexicon-based approach uses
pre-prepared sentiment dictionary to score a document by aggregating the
sentiment scores of all the words in the document. The pre-prepared
sentiment lexicon should contain a word and corresponding sentiment
score to it. Thus Lexicone approaches have the power to identify
Postive, Negative, and NEUTRAl tweets, can not read the context of the
sentence like a bert model. While losing the context of each tweet
reduces the amount of interpretation we can make out of NLP predictions,
sentiment scores are still useful to gauge the overall distribution of
opinion sentiments related to the War as they give us a more nuanced
picture of how the tweets are discussed in terms of continuous variable
(not binary predictors, 0/1). Load in Lexicone Dictionary:
nltk.download('vader\_lexicon') sentiments =
SentimentIntensityAnalyzer() \[nltk\_data\] Downloading package
vader\_lexicon to /root/nltk\_data...

Use the Lexicone Dictionary to look at each tweet and predict sentiments
of: 1) Positive 2) Negative 3) Neutral We call this the Tweet
Connotations! Add the predictions and user\_id and �ltered\_test to a
new df called predicted\_df pos = \[\] neg = \[\] neutral = \[\] for i
in df\["Filtered\_extended\_text"\]: score =
sentiments.polarity\_scores(i) del score\['compound'\] max\_value =
max(score, key=score.get) \#print(max\_value) if max\_value == 'neg':
neg.append(1) pos.append(0) 35 of 69 neutral.append(0)

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

neutral.append(0) elif max\_value == 'pos': neg.append(0) pos.append(1)
neutral.append(0) elif max\_value == 'neu': neg.append(0) pos.append(0)
neutral.append(1) predicted\_df =
df\[\['user\_id','Filtered\_extended\_text'\]\].copy()
predicted\_df\['nuetral'\] = neutral predicted\_df\['negative'\] = neg
predicted\_df\['positive'\] = pos predicted\_df user\_id

Filtered\_extended\_text

nuetral

negative

0

1376954378595180544

The Russian government and the Bank of Russia ...

1

0

3

1170848942784995328

Life Under the Specter of War: Images From Ukr...

1

0

5

1174009099824128000

I honestly don't think anything can beat the o...

1

0

7

1473016153467506688

Hi Pomp, at community we just created the Lim...

1

0

13

92895963

Another possible doping situation for Russia? ...

1

0

...

...

...

...

...

1867809

27567038

Look like Russian invaders to me

1

0

1867811

2208091680

The IAEA has 172 employees, 40% of whom are Ru...

1

0

positiv

4)  Predicting Emotions via a lexicon approach! Similarly we can use a
    lexicon approach to predict emotions! We load in a new lexicone
    emotion dictionary from NRCLex and use it for to predict emotions
    for each tweet! Emotions consist of:
5)  Fear
6)  Trust 36 of 693) Anger

11/28/23, 15:32

MAIN.ipynb - Colaboratory

3)  Anger

https://colab.research.google.com/drive/1NQq5N...

4)  Anticipation
5)  Sadness
6)  Suprise
7)  Disgust
8)  Joy We call these Emotions Add the predictions and user\_id and
    �ltered\_test to a new df called predicted\_df emotion\_dic =
    {'fear':\[\], 'trust':\[\], 'anger':\[\], 'anticipation':\[\],
    'sadness' \# Note Joy is also classified as hopful in the
    Documentation! for i in df\["Filtered\_extended\_text"\]: emotion =
    NRCLex(i.lower()) score = emotion.raw\_emotion\_scores emotion\_list
    = np.array(list(score.keys())) freq\_list =
    np.array(list(score.values())) try: max\_freq = max(freq\_list)
    except: max\_freq = 0 space = np.linspace(0,max\_freq) thresh =
    np.quantile(space,.5) top\_emotions = emotion\_list\[freq\_list \>=
    thresh\] if ('anger' in top\_emotions) and ('joy' in top\_emotions):
    index = top\_emotions != 'joy' top\_emotions =
    top\_emotions\[index\]

for emotion in emotion\_dic.keys(): if emotion in top\_emotions:
emotion\_dic\[emotion\].append(1) else:
emotion\_dic\[emotion\].append(0)

for emotion in emotion\_dic.keys(): predicted\_df\[emotion\] =
emotion\_dic\[emotion\] 37 of 69

predicted\_df

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

predicted\_df

/usr/local/lib/python3.7/dist-packages/ipykernel\_launcher.py:23:
FutureWarning: e
/usr/local/lib/python3.7/dist-packages/ipykernel\_launcher.py:29:
FutureWarning: e user\_id

Filtered\_extended\_text

nuetral

negative

0

1376954378595180544

The Russian government and the Bank of Russia ...

1

0

3

1170848942784995328

Life Under the Specter of War: Images From Ukr...

1

0

5

1174009099824128000

I honestly don't think anything can beat the o...

1

0

7

1473016153467506688

Hi Pomp, at community we just created the Lim...

1

0

13

92895963

Another possible doping situation for Russia? ...

1

0

...

...

...

...

...

1867809

27567038

Look like Russian invaders to me

1

0

1867811

2208091680

The IAEA has 172 employees, 40% of whom are Ru...

1

0

positiv

Google: Russian phishing

As we see, predicted\_df is a df with the User\_id, Filtered text, and
predictions!

5)  Now that we have predicted labels for Tweets lets take a deeper
    look! 5.1) Lets looks at the Percentage of Tweet Connotations and
    Emotions Tweet\_status =
    dict(predicted\_df.iloc\[:,2:5\].sum()/len(predicted\_df))
    Tweet\_emotion =
    dict(predicted\_df.iloc\[:,5:13\].sum()/len(predicted\_df))

f, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,7))
ax1.bar(Tweet\_status.keys(),Tweet\_status.values())
ax1.set\_xticklabels(Tweet\_status.keys(), rotation=90, fontsize = 12)
ax1.set\_title('Tweet Status (Connotation)',fontsize = 20, pad = 10)
ax1.set\_xlabel('Connotation',fontsize = 16,labelpad=10)
ax1.set\_ylabel('Percentage',fontsize = 16, labelpad = 10) 38 of 69

ax2.bar(Tweet\_emotion.keys(),Tweet\_emotion.values())

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

ax2.bar(Tweet\_emotion.keys(),Tweet\_emotion.values())
ax2.set\_xticklabels(Tweet\_emotion.keys(), rotation=90, fontsize = 12)
ax2.set\_title('Tweet Emotion',fontsize = 20, pad = 10)
ax2.set\_xlabel('Emotions',fontsize = 16,labelpad=10)
ax2.set\_ylabel('Percentage',fontsize = 16, labelpad = 10) plt.show()

The �gures above show the Connotations and Emotions of the Ukraine War
related tweets we have collected. The left panel, which visualizes the
percentage of neutral/negative/positive emotions of the total tweets,
shows that most of the tweets are neither completely negative nor
postivie but mostly on the middle ground, that is, they are mostly
moderate about the War. The right panel shows that the War-related
tweets frequently show emotions like fear, trust, anger, and
anticipation. Below, we provide word cloud images to visualize how each
emotions are expressed via speci�c words in a tweet.

5.2) Lets look at the most common words in each label! 39 of 69

11/28/23, 15:32 word\_label\_dic =
{'nuetral':\[\],'negative':\[\],'positive':\[\],'fear':\[\],
'trust':\[\],

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N... word\_label\_dic =
{'nuetral':\[\],'negative':\[\],'positive':\[\],'fear':\[\],
'trust':\[\], additional\_stop\_words =
\['rt','https','says','one','would','nan','get','new','said'

N = len(predicted\_df)

for i in range(N): curr\_row = predicted\_df.iloc\[i\] string =
curr\_row\['Filtered\_extended\_text'\] Index = (predicted\_df.iloc\[i\]
== 1) Relavent\_labels = predicted\_df.iloc\[0\].index\[Index\]
filtered\_list = tokenized\_content(string,additional\_stop\_words =
additional\_stop\_wor for word in filtered\_list: for r\_lab in
Relavent\_labels: word\_label\_dic\[r\_lab\].append(word) \# Time to
count the frequencies for each label! word\_label\_counter\_dic = {} for
lab in word\_label\_dic.keys(): temp\_counter = collections.Counter()
for word in word\_label\_dic\[lab\]: temp\_counter\[word\] += 1
word\_label\_counter\_dic\[lab\] = temp\_counter

visualizing the top words from Connotations fig, axs = plt.subplots(1,
3, figsize = (90,70)) x = 0 for lab in
list(word\_label\_counter\_dic.keys())\[0:3\]: temp =
dict(word\_label\_counter\_dic\[lab\].most\_common(150)) \# Deleting the
top 50 most common from EDA for key in
list(dict(top\_100).keys())\[0:50\]: try: del temp\[key\] except:
continue wc\_top =
WordCloud(background\_color='black',height=1000,width=1400, max\_words=
wc\_top.generate\_from\_frequencies(temp) axs\[x\].imshow(wc\_top)
axs\[x\].set\_title('Top words from Label:' + lab.upper(), fontsize =
75,pad = axs\[x\].set\_xticks(\[\]) axs\[x\].set\_yticks(\[\]) x +=1
plt.show()

40 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Its really cool to see how each connotations have such different
words!!!

Visualizing top words from Emotions fig, axs = plt.subplots(2, 4,
figsize = (100,40))

Plotting the first 4 emotions
=============================

x = 0 y = 0 for lab in list(word\_label\_counter\_dic.keys())\[3:11\]:
temp = dict(word\_label\_counter\_dic\[lab\].most\_common(150)) \#
Deleting the top 50 most common from EDA for key in
list(dict(top\_100).keys())\[0:50\]: try: del temp\[key\] except:
continue wc\_top =
WordCloud(background\_color='black',height=1000,width=1400, max\_words=
wc\_top.generate\_from\_frequencies(temp) axs\[y,x\].imshow(wc\_top)
axs\[y,x\].set\_title('Top words from Label:' + lab.upper(), fontsize =
75,pad = axs\[y,x\].set\_xticks(\[\]) axs\[y,x\].set\_yticks(\[\]) if x
== 3: x = -1 y = 1 x +=1 fig.show()

41 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Likewise!! The different emotions have such different words assoicated
with them! How COOL IS THIS!!!

6)  Connecting Followers to Following 6.1) Do Followers share the same
    opinion?
7)  Lets �nd a time range in which the Ukraine-Russian Tweets are high!
    (2/22-3/15)
8)  Lets then take the top N veri�ed followers from this list, and link
    them to their followers! We can link via Unveri�ed\_user\_following
    jsons �les! These �les showcase unveri�ed followers from our dataset
    and who they follow. Will we see that the follower's opinions for
    each label will match the Veri�ed user?
9)  lets �rst compile ALL opinions from Veri�ed-Following relationships
    and see the percentage that Agree and Disagree!
10) Is there a particular label that people disagree or agree with?

42 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

5)  lets take a deeper look and look at how each opinions for the top 5
    Veri�ed-Following Relationship compare to their followers

Visualizing Time Range of Intrest! \# Visualization of the time frame we
are looking at! day\_count =
np.array(df.groupby(by='day').count()\['id'\])

plt.figure(figsize=(13, 6)) plt.plot(dates,day\_count,'o',color = 'm',
label = '*nolegend*') plt.plot(dates,day\_count,color = 'm')
plt.title('Frequency of Ukraine-Russian Related Tweets vs Date',
fontsize = 16 plt.ylabel('Frequency of Ukraine-Russian Related
Tweets',fontsize = 14, labelpad= plt.xlabel('Date',fontsize = 14,
labelpad= 15) plt.xticks(rotation = 90) \# Rotates X-Axis Ticks by
90-degrees plt.axvline(x=15,color='red',linestyle='dashed')
plt.axvspan(13, 34, alpha=0.5, color='skyblue')
plt.legend(\['Ukraine-Russian Tweets Related','Day of
Invasion','Time-Range of Intrest' plt.show() np.sum(day\_count)

466547

43 of 69

Connecting Veri�ed with Followers

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Connecting Veri�ed with Followers

Load in json �les where the keys are users and their values is the list
of poeple they follow. This data was Scraped using o�cial Twitter APIs!
Please check out notebook: Follower\_info.ipynb to see scraping code! \#
Loading in a subsample of unverified user's following! with
open('Unverified\_user\_following\_final.json') as json\_file:
unver\_following = json.load(json\_file)

with open('Unverified\_user\_following\_2\_Rando\_final.json') as
json\_file: unver\_following\_2 = json.load(json\_file)

unver\_following.update(unver\_following\_2)

Create a dataframe with veri�ed people from our data, and their user\_id
Index

= (df\['user\_verified'\] == 1) & (df\['day'\] \<= 220315) &
(df\['day'\] \>= 220222

Verified\_df =
df\[Index\]\[\['user\_id','user\_screen\_name','Filtered\_extended\_text'
Verified\_df = Verified\_df.sort\_values(by = 'user\_followers\_count',
ascending=False user\_id = Verified\_df\['user\_id'\].unique()
user\_name = Verified\_df\['user\_screen\_name'\].unique() id\_name\_df
= pd.DataFrame({'user\_id':user\_id,'user\_name':user\_name})
id\_name\_df user\_id

user\_name

0

428333

cnnbrk

1

759251

CNN

2

807095

nytimes

3

5402612

BBCBreaking

4

742143

BBCWorld

...

...

...

6191 1127143755499225088

TUCIOﬃcial

6192 1354375054109401088

CityIndexSG

6193 44 of 69

839312384657338368 Bill\_4\_Congress

6194 1303584169143001088

UKRinALB

11/28/23, 15:32

MAIN.ipynb - Colaboratory

6195 1474043237912530944

https://colab.research.google.com/drive/1NQq5N...

sportsbriefcom

6196 rows × 2 columns

Iteratively look to see if a veri�ed user is in an unveri�ed following
list. If so then make a new Dict called Veri�ed\_follower\_dic\_all
where the key is the veri�ed user\_id, and the values are a list of
users that follow! Verified\_follower\_dic\_all = {} ver\_user\_ids =
list(id\_name\_df\['user\_id'\]) for ver\_user in ver\_user\_ids:
Verified\_follower\_dic\_all\[ver\_user\] = \[\] for key in
unver\_following.keys(): array = unver\_following\[key\] if ver\_user in
array: Verified\_follower\_dic\_all\[int(ver\_user)\].append(int(key))

Compile ALL opinions from Veri�ed-Following relationships and see the
percentage that Agree and Disagree! Combine the percentages from each
relationship to get a global Agree % and Disagree %! Using
Veri�ed\_follower\_dic\_all �nd the Veri�ed user's opinions and see how
their opinions correspond to their follower. Then aggregate for all
relationships into a global Agree or Disagree agree = 0 disagree = 0 for
Verified in Verified\_follower\_dic\_all: index =
predicted\_df\['user\_id'\] == Verified df\_of\_verified =
predicted\_df\[index\] A = (df\_of\_verified.iloc\[:,2:13\].sum() \>= 1)
\* 1 for entry in Verified\_follower\_dic\_all\[Verified\]: index =
predicted\_df\['user\_id'\] == entry df\_of\_user =
predicted\_df\[index\] B = (df\_of\_user.iloc\[:,2:13\].sum() \>= 1) \*
1 disagree += sum((A - B) != 0) agree += sum((A - B) == 0) \# Should add
to 11 since we have 11 labels

45 of 69tot = agree + disagree

percent\_agree\_or\_dis = \[agree/tot

11/28/23, 15:32

disagree/tot\]

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

percent\_agree\_or\_dis = \[agree/tot, disagree/tot\]

plt.figure(figsize=(13,6)) plt.bar(\['Agree with Verified','Disagree
with Verified'\],percent\_agree\_or\_dis) plt.title('Percentage of Users
that Agree or Disagree with a Verified User they Follow
plt.xlabel('Agree or Disagree with Verified Follower',labelpad = 15,
fontsize = plt.ylabel('Percent Agree or Disagree',labelpad = 15,
fontsize = 14) plt.show()

AWESOME! From this we see that most followers agree with the opinion of
the veri�ed user they follow! But do you think that speci�c labels might
have different outcomes??

Lets see if any speci�c Labels have higher percentage of Agree or
Disagree! Using Veri�ed\_follower\_dic\_all �nd the Veri�ed user's
opinions and see how their opinions correspond to their follower
ACCORDING to label. Then aggregate for all relationships into a global
Agree or Disagree per label! Agree\_label\_dic =
{'nuetral':0,'negative':0,'positive':0,'fear':0, 'trust':0,
Disagree\_label\_dic = {'nuetral':0,'negative':0,'positive':0,'fear':0,
'trust': label\_list = list(Agree\_label\_dic.keys()) for Verified in
Verified\_follower\_dic\_all: index = predicted\_df\['user\_id'\] ==
Verified df\_of\_verified = predicted\_df\[index\] 46 of 69 A =
(df\_of\_verified.iloc\[:,2:13\] sum() \>= 1) \* 1

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

A = (df\_of\_verified.iloc\[:,2:13\].sum() \>= 1) \* 1 for entry in
Verified\_follower\_dic\_all\[Verified\]: index =
predicted\_df\['user\_id'\] == entry df\_of\_user =
predicted\_df\[index\] B = (df\_of\_user.iloc\[:,2:13\].sum() \>= 1) \*
1 Agree\_Index = np.where(((A - B) == 0) == True)\[0\] for i in
Agree\_Index: key = label\_list\[i\] Agree\_label\_dic\[key\] += 1
Disagree\_Index = np.where(((A - B) != 0) == True)\[0\] for i in
Disagree\_Index: key = label\_list\[i\] Disagree\_label\_dic\[key\] += 1

Visualizing for Connotation \# Connotation fig, axs = plt.subplots(1, 3,
figsize = (33,6)) connotations = label\_list\[0:3\] counter = 0 for con
in connotations: tot = Agree\_label\_dic\[con\] +
Disagree\_label\_dic\[con\] sub\_percent =
\[Agree\_label\_dic\[con\]/tot,Disagree\_label\_dic\[con\]/tot\]
axs\[counter\].bar(\['Agree with Verified','Disagree with
Verified'\],sub\_percent axs\[counter\].set\_title('Users Consensus with
a Verified User They Follow for' axs\[counter\].set\_ylabel('Percent
Agree or Disagree',labelpad = 15, fontsize =
axs\[counter\].tick\_params(labelsize = 16) counter += 1 plt.show()

47 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Visualizing for Emotions \# Emotions fig, axs = plt.subplots(2, 4,
figsize = (105,35))

Plotting the first 4 emotions
=============================

x = 0 y = 0 connotations = label\_list\[3:11\] for con in connotations:
tot = Agree\_label\_dic\[con\] + Disagree\_label\_dic\[con\]
sub\_percent =
\[Agree\_label\_dic\[con\]/tot,Disagree\_label\_dic\[con\]/tot\]

axs\[y,x\].bar(\['Agree with Verified','Disagree with
Verified'\],sub\_percent) axs\[y,x\].set\_title('Users Consensus with a
Verified User They Follow for' + con.upp axs\[y,x\].set\_ylabel('Percent
Agree or Disagree',labelpad = 25, fontsize = 43
axs\[y,x\].tick\_params(labelsize = 42) if x == 3: x = -1 y = 1 x += 1
plt.tick\_params(axis='both', which='major', labelsize=40) plt.show()

48 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

WOW OH WOW! What we see from here is that per label, most users agree
with the Connotaiton, HOWEVER when it comes to emotion it is a toss up!

Now for the top 5 Veri�ed-Following Relationships (determined by highest
veri�ed follower) compare their follower's opinion to their own opinion.
Do this per label as well! 1) Plot the network of the Veri�ed user to
their followers 2) Under their network have bar graphs for each label
where you show the percentage of Users that have that label's opinion
Remember these are only users that follow the respective Veri�ed User.
3) For each bar graph have a Red Star over the opinion the veri�ed user
agreed with! 4) Split this for Connotation and Emotions so the graphs
are not too big! In the cell below we are iteratively look to see if a
veri�ed user is in an unveri�ed following list. If so then make a new
Dict called Veri�ed\_follower\_dic where the key is the veri�ed
user\_id, and the values are a list of users that follow! We do this for
only the top 5 most followed users that are veri�ed \# Top 5 verified
accounts and their followers is in the dict below
Verified\_follower\_dic = {} ver\_user\_ids =
list(id\_name\_df\['user\_id'\]) for ver\_user in ver\_user\_ids\[0:5\]:
Verified\_follower\_dic\[ver\_user\] = \[\] for key in
unver\_following.keys(): array = unver\_following\[key\] if ver\_user in
array: Verified\_follower\_dic\[int(ver\_user)\].append(int(key)) 49 of
69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

We can split this into two viualizations: One for Connotations
(Positive, Nuetral, Negative) and another for Emotions! Connotation
Visualization \# Connotations import networkx as nx ycounter = 0 fig,
axs = plt.subplots(4, 5, figsize = (50,35)) for Verified in
Verified\_follower\_dic: User\_name =
str(id\_name\_df\[id\_name\_df\['user\_id'\] ==
Verified\]\['user\_name'\].iloc \# Making our Network Here df\_test =
pd.DataFrame({'Source':Verified\_follower\_dic\[Verified\]})
df\_test\['Target'\] = Verified df\_test\['Type'\] = 'Unidirected'

G = nx.from\_pandas\_edgelist(df\_test,source='Source',target =
'Target') color\_map = \[\] for node in G: if node == Verified:
color\_map.append('red') else: color\_map.append('blue')

Time to get Unverified OPINION!
===============================

\#Num\_opinion =
{'nuetral':\[\],'negative':\[\],'positive':\[\],'fear':\[\],
'trust':\[\], 'ang Num\_opinion =
{'nuetral':\[\],'negative':\[\],'positive':\[\]} Percent\_opinion = {}
tot\_tweets = 0 for entry in Verified\_follower\_dic\[Verified\]: index
= predicted\_df\['user\_id'\] == entry df\_of\_user =
predicted\_df\[index\] tot\_tweets += len(df\_of\_user) user\_opinions =
dict(df\_of\_user.iloc\[:,2:13\].sum()) for op in user\_opinions.keys():
try: Num\_opinion\[op\].append(user\_opinions\[op\]) except: None

50 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

for opinion in Num\_opinion.keys(): Percent\_opinion\[opinion\] =
sum(Num\_opinion\[opinion\])/tot\_tweets \# Get the Percent Opinion for
the Verified \#Num\_opinion\_ver =
{'nuetral':\[\],'negative':\[\],'positive':\[\],'fear':\[\],
'trust':\[\], Num\_opinion\_ver =
{'nuetral':\[\],'negative':\[\],'positive':\[\]} Percent\_opinion\_ver =
{}

tot\_tweets = 0 index = predicted\_df\['user\_id'\] == Verified
df\_of\_user = predicted\_df\[index\] tot\_tweets += len(df\_of\_user)
user\_opinions = dict(df\_of\_user.iloc\[:,2:13\].sum()) for op in
user\_opinions.keys(): try:
Num\_opinion\_ver\[op\].append(user\_opinions\[op\]) except: None

for opinion in Num\_opinion\_ver.keys():
Percent\_opinion\_ver\[opinion\] = 1 if
sum(Num\_opinion\_ver\[opinion\])/tot\_tweets \>

nx.draw(G,node\_color=color\_map,with\_labels = False,
ax=axs\[0,ycounter\]) axs\[0,ycounter\].set\_title(User\_name, fontsize
= 50,pad = 50) counter = 1 for i in Percent\_opinion.keys(): opinion =
Percent\_opinion\[i\] anti = 1 - opinion ver\_opinion =
Percent\_opinion\_ver\[i\] if ver\_opinion == 1: x = 0 y = 1.1 else: x =
1 y = 1.1 axs\[counter,ycounter\].bar(\[i,'not' + i\],\[opinion,anti\])
axs\[counter,ycounter\].set\_ylabel(i + ' percentage',fontsize =
20,labelpad = axs\[counter,ycounter\].plot(x,y,'r\*',markersize = 30)
axs\[counter,ycounter\].tick\_params(labelsize = 20)

counter +=1 ycounter +=1 51 of 69

plt.show()

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

plt.show()

52 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

We can see for each of the 5 Veri�ed Followers and their following
networks, the Veri�ed User and their followers have similar opinions for
Connotations! This is similar of what we found per label in the previous
sections \# Emotions import networkx as nx ycounter = 0 fig, axs =
plt.subplots(9, 5, figsize = (50,65)) for Verified in
Verified\_follower\_dic: User\_name =
str(id\_name\_df\[id\_name\_df\['user\_id'\] ==
Verified\]\['user\_name'\].iloc \# Making our Network Here df\_test =
pd.DataFrame({'Source':Verified\_follower\_dic\[Verified\]})
df\_test\['Target'\] = Verified df\_test\['Type'\] = 'Unidirected'

G = nx.from\_pandas\_edgelist(df\_test,source='Source',target =
'Target') color\_map = \[\] for node in G: if node == Verified:
color\_map.append('red') else: color\_map.append('blue') \# Time to get
Unverified OPINION! Num\_opinion = {'fear':\[\], 'trust':\[\],
'anger':\[\], 'anticipation':\[\], 'sadness' Percent\_opinion = {}
tot\_tweets = 0 for entry in Verified\_follower\_dic\[Verified\]: index
= predicted\_df\['user\_id'\] == entry df\_of\_user =
predicted\_df\[index\] tot\_tweets += len(df\_of\_user) user\_opinions =
dict(df\_of\_user.iloc\[:,2:13\].sum()) for op in user\_opinions.keys():
try: Num\_opinion\[op\].append(user\_opinions\[op\]) except: 53 of 69
None

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

None

for opinion in Num\_opinion.keys(): Percent\_opinion\[opinion\] =
sum(Num\_opinion\[opinion\])/tot\_tweets \# Get the Percent Opinion for
the Verified Num\_opinion\_ver = {'fear':\[\], 'trust':\[\],
'anger':\[\], 'anticipation':\[\], 'sadness' Percent\_opinion\_ver = {}

tot\_tweets = 0 index = predicted\_df\['user\_id'\] == Verified
df\_of\_user = predicted\_df\[index\] tot\_tweets += len(df\_of\_user)
user\_opinions = dict(df\_of\_user.iloc\[:,2:13\].sum()) for op in
user\_opinions.keys(): try:
Num\_opinion\_ver\[op\].append(user\_opinions\[op\]) except: None

for opinion in Num\_opinion\_ver.keys():
Percent\_opinion\_ver\[opinion\] = 1 if
sum(Num\_opinion\_ver\[opinion\])/tot\_tweets \>

nx.draw(G,node\_color=color\_map,with\_labels = False,
ax=axs\[0,ycounter\]) axs\[0,ycounter\].set\_title(User\_name, fontsize
= 50,pad = 50) counter = 1 for i in Percent\_opinion.keys(): opinion =
Percent\_opinion\[i\] anti = 1 - opinion ver\_opinion =
Percent\_opinion\_ver\[i\] if ver\_opinion == 1: x = 0 y = 1.1 else: x =
1 y = 1.1 axs\[counter,ycounter\].bar(\[i,'not' + i\],\[opinion,anti\])
axs\[counter,ycounter\].set\_ylabel(i + ' percentage',fontsize =
20,labelpad = axs\[counter,ycounter\].plot(x,y,'r\*',markersize = 30)
axs\[counter,ycounter\].tick\_params(labelsize = 20)

54 of 69

counter +=1 ycounter +=1

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

ycounter +=1 plt.show()

55 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

We can see for each of the 5 Veri�ed Followers and their following
networks, the Veri�ed User and their followers DONT have similar
opinions for Emotions! This is similar of what we found per label in the
previous sections

7)  Summary Our �ndings based on a series of sentiment analyses and
    language predictions suggest that, while individual users share
    similar opinions toward the Ukraine War with the prominent Twitter
    accounts they follow (i.e., useres' opinion leaders), the ways of
    expressing their emotions about the War do not necessarily align
    with the opinion leaders they follow. As the �gures above indicate,
    Twitter users show similar stances toward the Ukraine War (i.e.,
    connotations) with their opinion leaders, but how they express their
    stances via different emotions ranging from anger to joy/hopefulness
    diverge across users.

56 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Thank you PANDAS for helping our assignment!!!!!!!

8)  Appendix 1: Another Unused Bert Model For Negative/Positive Emotion
    Prediction. In this section we optimized our BERT model in section
    3.3 and we used it to predict postive and negative Tweets. We saw
    that the model predicted around 50/50 for positive/negative. This
    made us worried, since we have many nuetral tweets. This appendex
    helped us realize a Lexicon approach is much better!

8.1) Model Performance See bert model training notebook in our project
folder for the original code and more details

57 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

8.2) Environment Setup gpu\_info = !nvidia-smi gpu\_info =
'`\n`{=tex}'.join(gpu\_info) if gpu\_info.find('failed') \>= 0:
print('Not connected to a GPU') else: print(gpu\_info)

58 of 69

Sun May 1 00:17:02 2022
+-----------------------------------------------------------------------------+
\| NVIDIA-SMI 460.32.03 Driver Version: 460.32.03 CUDA Version: 11.2 \|
\|-------------------------------+----------------------+----------------------+
\| GPU Name Persistence-M\| Bus-Id Disp.A \| Volatile Uncorr. ECC \| \|
Fan Temp Perf Pwr:Usage/Cap\| Memory-Usage \| GPU-Util Compute M. \| \|
\| \| MIG M. \|
\|===============================+======================+======================\|
\| 0 Tesla P100-PCIE... Off \| 00000000:00:04.0 Off \| 0 \| \| N/A 33C
P0 27W / 250W \| 0MiB / 16280MiB \| 0% Default \| \| \| \| N/A \|
11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

+-------------------------------+----------------------+----------------------+
+-----------------------------------------------------------------------------+
\| Processes: \| \| GPU GI CI PID Type Process name GPU Memory \| \| ID
ID Usage \|
\|=============================================================================\|
\| No running processes found \|
+-----------------------------------------------------------------------------+

from psutil import virtual\_memory ram\_gb = virtual\_memory().total /
1e9 print('Your runtime has {:.1f} gigabytes of available
RAM`\n`{=tex}'.format(ram\_gb)) if ram\_gb \< 20: print('Not using a
high-RAM runtime') else: print('You are using a high-RAM runtime!') Your
runtime has 13.6 gigabytes of available RAM Not using a high-RAM runtime

!pip install -q tf-models-official==2.4.0 !pip install -q -U
\"tensorflow-text==2.8.\*\" !pip install tensorflow\_hub !pip install
keras tf-models-official pydot graphviz !git clone
https://github.com/tensorflow/models.git

59 of 69

\|████████████████████████████████\| 1.1 MB 4.2 MB/s
\|████████████████████████████████\| 237 kB 68.7 MB/s
\|████████████████████████████████\| 1.1 MB 24.2 MB/s
\|████████████████████████████████\| 43 kB 2.2 MB/s
\|████████████████████████████████\| 1.2 MB 49.7 MB/s
\|████████████████████████████████\| 99 kB 11.0 MB/s
\|████████████████████████████████\| 47.8 MB 83.9 MB/s
\|████████████████████████████████\| 596 kB 34.5 MB/s
\|████████████████████████████████\| 352 kB 72.8 MB/s
\|████████████████████████████████\| 462 kB 63.1 MB/s Building wheel for
py-cpuinfo (setup.py) ... done Building wheel for seqeval (setup.py) ...
done \|████████████████████████████████\| 4.9 MB 4.3 MB/s Requirement
already satisfied: tensorflow\_hub in /usr/local/lib/python3.7/dist-pa
Requirement already satisfied: protobuf\>=3.8.0 in
/usr/local/lib/python3.7/dist-p Requirement already satisfied:
numpy\>=1.12.0 in /usr/local/lib/python3.7/dist-pac Requirement already
satisfied: six\>=1.9 in /usr/local/lib/python3.7/dist-packages
Requirement already satisfied: keras in
/usr/local/lib/python3.7/dist-packages (2 Requirement already satisfied:
tf-models-official in /usr/local/lib/python3.7/dis Requirement already
satisfied: pydot in /usr/local/lib/python3.7/dist-packages (1
Requirement already satisfied: graphviz in
/usr/local/lib/python3.7/dist-packages Requirement already satisfied:
tensorflow-hub\>=0.6.0 in /usr/local/lib/python3.7/ Requirement already
satisfied: pandas\>=0.22.0 in /usr/local/lib/python3.7/dist-pa
Requirement already satisfied: dataclasses in
/usr/local/lib/python3.7/dist-packa Requirement already satisfied:
pyyaml\>=5.1 in /usr/local/lib/python3.7/dist-packa Requirement already
satisfied: opencv-python-headless in /usr/local/lib/python3.7
Requirement already satisfied: tensorflow-addons in
/usr/local/lib/python3.7/dist Requirement already satisfied:
tf-slim\>=1.1.0 in /usr/local/lib/python3.7/dist-pa 11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Requirement already satisfied: pycocotools in
/usr/local/lib/python3.7/dist-packa Requirement already satisfied:
sentencepiece in /usr/local/lib/python3.7/dist-pac Requirement already
satisfied: seqeval in /usr/local/lib/python3.7/dist-packages Requirement
already satisfied: oauth2client in /usr/local/lib/python3.7/dist-pack
Requirement already satisfied: tensorflow-model-optimization\>=0.4.1 in
/usr/local Requirement already satisfied: gin-config in
/usr/local/lib/python3.7/dist-packag Requirement already satisfied:
py-cpuinfo\>=3.3.0 in /usr/local/lib/python3.7/dist Requirement already
satisfied: google-api-python-client\>=1.6.7 in /usr/local/lib/
Requirement already satisfied: Cython in
/usr/local/lib/python3.7/dist-packages ( Requirement already satisfied:
six in /usr/local/lib/python3.7/dist-packages (fro Requirement already
satisfied: numpy\>=1.15.4 in /usr/local/lib/python3.7/dist-pac
Requirement already satisfied: Pillow in
/usr/local/lib/python3.7/dist-packages ( Requirement already satisfied:
tensorflow-datasets in /usr/local/lib/python3.7/di Requirement already
satisfied: matplotlib in /usr/local/lib/python3.7/dist-packag
Requirement already satisfied: psutil\>=5.4.3 in
/usr/local/lib/python3.7/dist-pac Requirement already satisfied:
scipy\>=0.19.1 in /usr/local/lib/python3.7/dist-pac Requirement already
satisfied: google-cloud-bigquery\>=0.31.0 in /usr/local/lib/py
Requirement already satisfied: kaggle\>=1.3.9 in
/usr/local/lib/python3.7/dist-pac Requirement already satisfied:
tensorflow\>=2.4.0 in /usr/local/lib/python3.7/dist Requirement already
satisfied: uritemplate\<4dev,\>=3.0.0 in /usr/local/lib/python3
Requirement already satisfied: google-auth-httplib2\>=0.0.3 in
/usr/local/lib/pyth Requirement already satisfied:
google-api-core\<3dev,\>=1.21.0 in /usr/local/lib/py Requirement already
satisfied: google-auth\<3dev,\>=1.16.0 in /usr/local/lib/python
Requirement already satisfied: httplib2\<1dev,\>=0.15.0 in
/usr/local/lib/python3.7 Requirement already satisfied: packaging\>=14.3
in /usr/local/lib/python3.7/dist-p Requirement already satisfied:
googleapis-common-protos\<2.0dev,\>=1.6.0 in /usr/lo Requirement already
satisfied: pytz in /usr/local/lib/python3.7/dist-packages (fr
Requirement already satisfied: requests\<3.0.0dev,\>=2.18.0 in
/usr/local/lib/pytho Requirement already satisfied: protobuf\>=3.12.0 in
/usr/local/lib/python3.7/distRequirement already satisfied:
setuptools\>=40.3.0 in /usr/local/lib/python3.7/dis Requirement already
satisfied: pyasn1-modules\>=0.2.1 in /usr/local/lib/python3.7/ import
pandas as pd import numpy as np import matplotlib.pyplot as plt import
networkx as nx import os os.environ\['PYTHONPATH'\] +=
":/content/models" import sys sys.path.append("/content/models") import
IPython import numpy as np import pandas as pd import tensorflow as tf
import tensorflow\_hub as hub from keras.utils import np\_utils import
official.nlp.bert.bert\_models import official.nlp.bert.configs import
official.nlp.bert.run\_classifier import official.nlp.bert.tokenization
as tokenization 60 of 69from official.modeling import tf\_utils

11/28/23, 15:32

MAIN.ipynb - Colaboratory

official.modeling import tf\_utils from official import nlp from
official.nlp import bert

https://colab.research.google.com/drive/1NQq5N...

from sklearn.model\_selection import train\_test\_split from
sklearn.preprocessing import LabelEncoder import matplotlib.pyplot as
plt gpus = tf.config.experimental.list\_physical\_devices('GPU') if
gpus: try: \# Currently, memory growth needs to be the same across GPUs
for gpu in gpus: tf.config.experimental.set\_memory\_growth(gpu, True)
logical\_gpus = tf.config.experimental.list\_logical\_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical\_gpus), "Logical GPUs")
except RuntimeError as e: \# Memory growth must be set before GPUs have
been initialized print(e)

print("Version:", tf.\_\_version\_\_) print("Eager mode:",
tf.executing\_eagerly()) print("Hub version:", hub.\_\_version\_\_)
print("GPU is", "available" if tf.config.list\_physical\_devices('GPU')
else \"NOT AVAILA 1 Physical GPUs, 1 Logical GPUs Version: 2.8.0 Eager
mode: True Hub version: 0.12.0 GPU is available

8.3) Load In The Dataset and The Bert Model

df = pd.read\_json('/content/drive/MyDrive/CIS
545/Ukrain\_Russia\_tweets\_Feb\_March\_FULLY df.head() created\_at

61 of 69

id

user\_id

user\_name

0

2022-02-09 18:57:04+00:00

1491486324000000000 1376954378595180544 droptown.io

3

2022-02-09 18:11:08+00:00

1491474764000000000 1170848942784995328

Alexander

5

2022-02-09 01:16:35+00:00

1491219444000000000 1174009099824128000

James Rath

7

2022-02-09 06:53:19+00:00

1491304186000000000 1473016153467506688

Limited USD (LUSD) 11/28/23, 15:32

MAIN.ipynb - Colaboratory

13

2022-02-09 06:26:19+00:00

https://colab.research.google.com/drive/1NQq5N...

1491297391000000000

92895963

Samar Hashemi

5 rows × 33 columns

model\_fname = 'twitter\_BERT' my\_wd = '/content/drive/MyDrive/CIS 545'
bert\_model = tf.keras.models.load\_model(os.path.join(my\_wd,
model\_fname)) bert\_model.summary()

Model: "model"
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
Layer (type) Output Shape Param \# Connected to
=================================================================================
input\_word\_ids (InputLayer) \[(None, 240)\] 0 \[\] input\_mask
(InputLayer)

\[(None, 240)\]

0

\[\]

segment\_ids (InputLayer)

\[(None, 240)\]

0

\[\]

keras\_layer (KerasLayer)

\[(None, 768), (None, 240, 768)\]

177853441

\['input\_word\_ids 'input\_mask\[0\]\[ 'segment\_ids\[0\]

dropout (Dropout)

(None, 768)

0

\['keras\_layer\[0\]

output (Dense)

(None, 2)

1538

\['dropout\[0\]\[0\]'

=================================================================================
Total params: 177,854,979 Trainable params: 177,854,978 Non-trainable
params: 1
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

tf.keras.utils.plot\_model(bert\_model, show\_shapes=True, dpi=48)

8.4)Text Preprocessing 62 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

Removing special characters, white space and tabs x\_test =
df\['Filtered\_extended\_text'\] x\_test = x\_test.reset\_index() import
re def remove\_extra\_whitespace\_tabs(text): \#pattern =
r'\^`\s`{=tex}+$|\s+$' pattern = r'\^`\s*`{=tex}\|`\s`{=tex}`\s*`{=tex}'
return re.sub(pattern, ' ', text).strip() \# function to remove special
characters def remove\_special\_characters(text): \# define the pattern
to keep text = text.replace('-',' ') pat =
r'\[\^a-zA-z0-9.,!?/:;\\\"\\\'\\s\]' return re.sub(pat, '', text)
x\_test\['Filtered\_extended\_text'\] =
x\_test\['Filtered\_extended\_text'\].apply(lambda
x\_test\['Filtered\_extended\_text'\] =
x\_test\['Filtered\_extended\_text'\].apply(lambda x\_test index

Filtered\_extended\_text

0

0

The Russian government and the Bank of Russia ...

1

3

Life Under the Specter of War: Images From Ukr...

2

5

ani\_obsessive I honestly don't think anything ...

3

7

APompliano Hi Pomp, at community we just creat...

4

13

Another possible doping situation for Russia? ...

...

...

...

466542

1867809

thereality031 Real\_Jayant RT\_com Look like Rus...

466543

1867811 The IAEA has 172 employees, 40 of whom are Rus...

466544

1867814

Google: Russian phishing attacks target NATO, ...

466545

1867815

telegram Could you tell the world why youve re...

466546

1867829

RepLizCheney I do appreciate that you have tak...

466547 rows × 2 columns

Tokenization tokenizerSaved = bert.tokenization.FullTokenizer(
vocab\_file=os.path.join(my\_wd model\_fname 'assets/vocab.txt'),

63 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...
vocab\_file=os.path.join(my\_wd, model\_fname, 'assets/vocab.txt'),
do\_lower\_case=False)

tokenizedTweet =
tokenizerSaved.tokenize(x\_test\['Filtered\_extended\_text'\]\[0\]) for
i in tokenizedTweet: print(i,
tokenizerSaved.convert\_tokens\_to\_ids(\[i\]))

64 of 69

The \[10117\] Russian \[13463\] government \[12047\] and \[10111\] the
\[10105\] Bank \[13533\] of \[10108\] Russia \[14664\] look \[25157\] to
\[10114\] have \[10529\] c \[171\] \#\#ry \[10908\] \#\#pt \[14971\]
\#\#o \[10133\] regulated \[106125\] as \[10146\] cu \[10854\] \#\#rren
\[46111\] \#\#cies \[18320\] , \[117\] local \[11436\] reports \[26610\]
have \[10529\] suggested \[27675\] . \[119\] Russia \[14664\] is
\[10124\] set \[11847\] to \[10114\] formally \[43082\] re \[11639\]
\#\#co \[10812\] \#\#gni \[27170\] \#\#se \[10341\] c \[171\] \#\#ry
\[10908\] \#\#pt \[14971\] \#\#oc \[25125\] \#\#urre \[97235\] \#\#ncies
\[21512\] as \[10146\] cu \[10854\] \#\#rren \[46111\] \#\#cies
\[18320\] , \[117\] news \[14424\] reports \[26610\] out \[10950\] of
\[10108\] the \[10105\] country \[12723\] suggest \[56874\]

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

suggest \[56874\] . \[119\] Cry \[33909\] \#\#pt \[14971\] \#\#o
\[10133\]

x\_test\['Filtered\_extended\_text'\]\[0\] 'The Russian government and
the Bank of Russia look to have crypto regulat ed as currencies, local
reports have suggested. Russia is set to formally recognise
cryptocurrencies as currencies, news reports out of the country
bert\_layer =
hub.KerasLayer(\"https://tfhub.dev/tensorflow/bert\_multi\_cased\_L-1
trainable=True) vocab\_file =
bert\_layer.resolved\_object.vocab\_file.asset\_path.numpy()
do\_lower\_case = bert\_layer.resolved\_object.do\_lower\_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab\_file, do\_lower\_case)
tokenizer.convert\_tokens\_to\_ids(\['\[CLS\]', '\[SEP\]'\]) \[101,
102\]

def encode\_names(n): tokens = list(tokenizer.tokenize(n))
tokens.append('\[SEP\]') \# seperation token. Would be much more useful
if you had a return tokenizer.convert\_tokens\_to\_ids(tokens) tweets =
tf.ragged.constant(\[ encode\_names(n) for n in
x\_test\['Filtered\_extended\_text'\]\])

cls =
\[tokenizer.convert\_tokens\_to\_ids(\['\[CLS\]'\])\]\*tweets.shape\[0\]
input\_word\_ids = tf.concat(\[cls, tweets\], axis=-1) \_ =
plt.pcolormesh(input\_word\_ids\[0:10\].to\_tensor())

def encode\_names(n, tokenizer): tokens = list(tokenizer.tokenize(n))
tokens.append('\[SEP\]') return
tokenizer.convert\_tokens\_to\_ids(tokens) 65 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

def bert\_encode(string\_list, tokenizer, max\_seq\_length):
num\_examples = len(string\_list) string\_tokens = tf.ragged.constant(\[
encode\_names(n, tokenizer) for n in np.array(string\_list)\]) cls =
\[tokenizer.convert\_tokens\_to\_ids(\['\[CLS\]'\])\]\*string\_tokens.shape\[0\]
input\_word\_ids = tf.concat(\[cls, string\_tokens\], axis=-1)

input\_mask = tf.ones\_like(input\_word\_ids).to\_tensor(shape=(None,
max\_seq\_length type\_cls = tf.zeros\_like(cls) type\_tokens =
tf.ones\_like(string\_tokens) input\_type\_ids = tf.concat( \[type\_cls,
type\_tokens\], axis=-1).to\_tensor(shape=(None, max\_seq\_length))
inputs = { 'input\_word\_ids': input\_word\_ids.to\_tensor(shape=(None,
max\_seq\_length)), 'input\_mask': input\_mask, 'input\_type\_ids':
input\_type\_ids} return inputs encoder\_fname = 'twitter\_classes.npy'
my\_wd = '/content/drive/MyDrive/CIS 545' encoder = LabelEncoder()
encoder.classes\_ = np.load(os.path.join(my\_wd, encoder\_fname),
allow\_pickle=True twt = \['i hate you'\] input =
bert\_encode(string\_list=list(twt), tokenizer=tokenizerSaved,
max\_seq\_length=240) prediction1 = bert\_model.predict(input)

8.5) Model Predicting twts =
x\_test\['Filtered\_extended\_text'\].iloc\[:1000\] prediction = \[\]
for i in twts: twt = \[i\] input = bert\_encode(string\_list=list(twt),
tokenizer=tokenizerSaved, max\_seq\_length=240) prediction1 =
bert\_model.predict(input) 66 of 69 if
encoder.classes\_\[np.argmax(prediction1)\]==4:
prediction.append('positive')

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

prediction.append('positive') else: prediction.append('negative')
prediction

67 of 69

\['negative', 'positive', 'positive', 'positive', 'negative',
'negative', 'positive', 'positive', 'negative', 'positive', 'negative',
'positive', 'positive', 'negative', 'negative', 'negative', 'negative',
'positive', 'negative', 'positive', 'positive', 'positive', 'negative',
'positive', 'negative', 'negative', 'negative', 'positive', 'positive',
'negative', 'positive', 'negative', 'positive', 'positive', 'negative',
'negative', 'negative', 'positive', 'negative', 'positive', 'negative',
'positive', 'negative', 'negative', 'positive', 'negative', 'negative',
'negative', 'positive', 'positive', 'negative', 'positive', 'negative',
'negative',

11/28/23, 15:32

MAIN.ipynb - Colaboratory

https://colab.research.google.com/drive/1NQq5N...

'negative', 'negative', 'negative', 'negative', 'positive', 'negative',

8.6) Visualization \# label = pd.DataFrame({'predicted\_emotion':
prediction}) emo\_df = df\[\['user\_id', 'user\_verified', 'day',
'favorite\_count'\]\].iloc\[:1000\] emo\_df\['predicted\_emotion'\] =
prediction from matplotlib import pyplot as plt

import seaborn as sns plt.figure(figsize=(16,8)); ax =
sns.countplot(x="predicted\_emotion",data=emo\_df) plt.ylim(400, 550);
plt.title('Predicted Emotion');

68 of 69

11/28/23, 15:32

MAIN.ipynb - Colaboratory

69 of 69

https://colab.research.google.com/drive/1NQq5N...

11/28/23, 15:32


