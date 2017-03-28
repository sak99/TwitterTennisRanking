
# coding: utf-8

# In[4]:

import json
tweets_data_path = 'F:\Data Analysis\PythonProgs\Twitfeed\Tennis\old_tennis2.json'

tweets_file = open(tweets_data_path, "r")

geo_data = {
        "type": "FeatureCollection",
        "features": []
    }
for line in tweets_file:
    tweet = json.loads(line)
    if tweet['coordinates']:
        geo_json_feature = {
                "type": "Feature",
                "geometry": tweet['coordinates'],
                "properties": {
                    "text": tweet['text'],
                    "created_at": tweet['created_at']
                }
            }
        geo_data['features'].append(geo_json_feature)
 

with open('F:\Data Analysis\PythonProgs\Twitfeed\Tennis\geo_data.json', 'w') as fout:
    fout.write(json.dumps(geo_data, indent=4))


# In[ ]:



