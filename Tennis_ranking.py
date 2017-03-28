
# coding: utf-8

# In[1]:

import json
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:

import re
 
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
        
    return tokens


# In[ ]:




# In[ ]:




# In[3]:

tweets_data_path = 'source.json'

tweets_data = []
tokens=[]
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
        
        tokens.append(preprocess(tweet['text']))
        
    except:
        continue


# In[4]:

print len(tweets_data)


# In[ ]:




# In[5]:

def populate_tweet_df(tweets_data):
    tweets = pd.DataFrame()
 
    tweets['text'] = map(lambda tweet: tweet['text'], tweets_data)
    tweets['lang'] = map(lambda tweet: tweet['lang'], tweets_data)
 
    tweets['location'] = list(map(lambda tweet: tweet['user']['location'], tweets_data))
    tweets['country'] = map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data)
 
    tweets['country_code'] = list(map(lambda tweet: tweet['place']['country_code']
                                  if tweet['place'] != None else '', tweets_data))
 
    tweets['long'] = list(map(lambda tweet: tweet['coordinates']['coordinates'][0]
                        if tweet['coordinates'] != None else 'NaN', tweets_data))
 
    tweets['latt'] = list(map(lambda tweet: tweet['coordinates']['coordinates'][1]
                        if tweet['coordinates'] != None else 'NaN', tweets_data))
 
    return tweets


# In[6]:

tweets=populate_tweet_df(tweets_data)


# In[10]:

from nltk.corpus import stopwords
import string
 
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via', 'RT','ud83d','u2026','u1000']


# In[11]:

import operator 
import json
from collections import Counter

tweets_file = open(tweets_data_path, "r")
count_all = Counter()
for line in tweets_file:
    tweet = json.loads(line)
        # Create a list with all the terms
    terms_stop = [term for term in preprocess(tweet['text']) if term not in stop]
       # Update the counter
    count_all.update(terms_stop)
    # Print the first 5 most frequent words
print(count_all.most_common(10))


# In[14]:

from collections import defaultdict
# remember to include the other import from the previous post
 
com = defaultdict(lambda : defaultdict(int))
tweets_file = open(tweets_data_path, "r")
count_terms_only = Counter()
for line in tweets_file: 
 
    tweet = json.loads(line)
    terms_only = [term for term in preprocess(tweet['text']) 
                  if term not in stop 
                  and not term.startswith(('#', '@'))]
    count_terms_only.update(terms_only)
 
    # Build co-occurrence matrix
    for i in range(len(terms_only)-1):            
        for j in range(i+1, len(terms_only)):
            w1, w2 = sorted([terms_only[i], terms_only[j]])                
            if w1 != w2:
                com[w1][w2] += 1


# In[13]:

com_max = []
# For each term, look for the most common co-occurrent terms
for t1 in com:
    t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
    for t2, t2_count in t1_max_terms:
        com_max.append(((t1, t2), t2_count))
# Get the most frequent co-occurrences
terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
print(terms_max[:5])


# In[ ]:




# In[15]:

tweets=populate_tweet_df(tweets_data)


# In[ ]:




# In[16]:

tweets_by_lang = tweets['lang'].value_counts()

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Languages', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('Top 5 languages', fontsize=15, fontweight='bold')
tweets_by_lang[:5].plot(ax=ax, kind='bar', color='red')


# In[17]:

tweets_by_country = tweets['country'].value_counts()

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Countries', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('Top 10 countries', fontsize=15, fontweight='bold')
tweets_by_country[:10].plot(ax=ax, kind='bar', color='blue')


# In[18]:

get_ipython().magic(u'pylab inline')


# In[19]:

import re


# In[20]:

def word_in_text(word, text):
   
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    


    if match:
        return True
    return False


# In[27]:


tweets['Roger'] = tweets['text'].apply(lambda tweet: word_in_text('roger|Federer|fed', tweet))
tweets['Rafa'] = tweets['text'].apply(lambda tweet: word_in_text('nadal|rafa|rafael', tweet))




# In[28]:

print tweets['Rafa'].value_counts()


# In[ ]:




# In[29]:

print tweets['Rafa'].value_counts()[True]
print tweets['Roger'].value_counts()[True]


# In[30]:

prg_langs = ['Rafa', 'Roger']
tweets_by_prg_lang = [tweets['Rafa'].value_counts()[True], tweets['Roger'].value_counts()[True]]

x_pos = list(range(len(prg_langs)))
width = 0.8
fig, ax = plt.subplots()
plt.bar(x_pos, tweets_by_prg_lang, width, alpha=1, color='br')

# Setting axis labels and ticks
ax.set_ylabel('Number of tweets', fontsize=15)
ax.set_title('Ranking: DRafa vs Roger ', fontsize=10, fontweight='bold')
ax.set_xticks([p + 0.4 * width for p in x_pos])
ax.set_xticklabels(prg_langs)
plt.grid()


# In[32]:

def extract_link(text):
    regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    match = re.search(regex, text)
    if match:
        return match.group()
    return ''


# In[33]:

tweets['link'] = tweets['text'].apply(lambda tweet: extract_link(tweet))


# In[34]:


tweets_relevant_with_link = tweets[tweets['link'] != '']


# In[37]:

print tweets_relevant_with_link[tweets_relevant_with_link['Roger'] == True]['link']
print tweets_relevant_with_link[tweets_relevant_with_link['Rafa'] == True]['link']


# In[39]:

print tweets_relevant_with_link[tweets_relevant_with_link['Roger'] == True]['link']


# In[ ]:




# In[ ]:




# In[41]:

temp1 = tweets['Roger'].value_counts()
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Roger Federer')
ax1.set_ylabel('Count ')
ax1.set_title("nummber of tweets for Federer")
temp1.plot(kind='bar')



# In[42]:

tweets.apply(lambda x: sum(x.isnull()),axis=0) 


# In[ ]:




# In[43]:

from mpl_toolkits.basemap import Basemap
 
# plot the blank world map
my_map = Basemap(projection='merc', lat_0=50, lon_0=-100,
                     resolution = 'l', area_thresh = 5000.0,
                     llcrnrlon=-140, llcrnrlat=-55,
                     urcrnrlon=160, urcrnrlat=70)
# set resolution='h' for high quality
 
# draw elements onto the world map
my_map.drawcountries()
#my_map.drawstates()
my_map.drawcoastlines(antialiased=False,
                      linewidth=0.005)
 
# add coordinates as red dots
longs = list(tweets.loc[(tweets.long != 'NaN')].long)
latts = list(tweets.loc[tweets.latt != 'NaN'].latt)
x, y = my_map(longs, latts)
my_map.plot(x, y, 'ro', markersize=6, alpha=0.5)
 


# In[15]:


import vincent

word_freq = count_terms_only.most_common(20)
labels, freq = zip(*word_freq)
data = {'data': freq, 'x': labels}
bar = vincent.Bar(data, iter_idx='x')
bar.to_json('term_freq.json')


# In[16]:

bar.to_json('term_freq.json', html_out=True, html_path='chart.html')


# In[ ]:



