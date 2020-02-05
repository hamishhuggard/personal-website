---
title: "The Shape of Stories"
published: True
---

<img src="/assets/images/2020-01-31-shape-of-stories/kurt_vonnegart.png" alt="Vonnegart" width="80%"/>

There’s a great clip of Kurt Vonnegart giving a lecture on [“the shape of stories”](https://www.youtube.com/watch?v=oP3c1h8v2ZQ). He makes the case that stories can be distilled down to a two-dimensional plot. The y-axis is the valence of the story (what he calls the "G-B axis" for good to bad), and the x-axis is time (which he calls the "B-E" axis, for - you guessed it - beginnging to entropy). Not only does Vonnegart say that stories can be distilled to these shapes, but also that:

> There's no reason why the simple shapes of stories can't be fed into computers.

That is, we should be able to extract the shape of a story using algorithms! And so, in this post I will attempt to do just that; I will use sentiment analysis to try and empirically recreate the curves which Vonnegart attributes to several stories.

NOTE: Sorry about the maths not rendering properly in this post. I'm still figuring out this markdown format.

Let's start by importing the libraries we'll need and looking into some sentiment analysis.


```python
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.corpus import gutenberg as gt
import matplotlib.pyplot as plt
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup
import requests
import re
import seaborn as sns
import pandas as pd
```


```python
plt.style.use('seaborn')
```

## Sentiment Analysis

There are many approaches to sentiment analysis. The easiest is a simple bag-of-words model, in which we count up the number of positive words (“happy”, “good”, “amazeballs”) in the text, then count the number of negative words (“hangry”, “bleh”, “stanky”), and the valence of the text is the number of positive words minus the number of negative words. This approach will probably do for our purposes.

After a little Googling, I came across [this](https://positivewordsresearch.com/sentiment-analysis-resources/) page of sentiment analysis resources. I decided to go with [SentiWordNet](https://github.com/aesuli/sentiwordnet) because it's built on WordNet which I'm already familiar with. SentiWordNet gives a positive sentiment and negative sentiment score to every synset (group of synonymous words) in WordNet's lexicon. Because a given word could have several associated synsets, each corresponding to a different meaning of the word, there are several possible sentiment values associated with each word. I decided to simply use the sentiment associated with the *first* synset of each word.


```python
def get_sentiment(word):
    try:
        synset = wn.synsets(word)[0].name()
        sent = swn.senti_synset(synset)
        # Overall sentiment score is positive score minus negative score.
        return sent.pos_score() - sent.neg_score()
    except:
        # If the word isn't found in the synset dictionary, assume neutral sentiment.
        return 0
```

Let's make sure that this sentiment function returns sensible results:


```python
for word in ['good', 'bad', 'happy', 'sad', 'terrible',
             'death', 'hate', 'poverty', 'misery', 'party']:
    print(word, get_sentiment(word))
```

    good 0.5
    bad -0.875
    happy 0.875
    sad -0.625
    terrible -0.625
    death 0.0
    hate -0.25
    poverty -0.625
    misery -0.125
    party 0.0


Looks ok. Ideally terrible would be worse than bad, and misery would be worse than poverty.

## Text Extraction

We'll use Beautiful Soup to extract the text of the story from the web.


```python
def get_text_from_url(url):
    try:
        page = requests.get(url)
    except:
        page = requests.get(url, headers={'User-Agent': ''})
    finally:
        soup = BeautifulSoup(page.content)
        return soup.get_text()
```

Let's see if we can extract the text for Cinderella.


```python
cind_url_la = 'https://www.pitt.edu/~dash/grimm021.html'
texterella = get_text_from_url(cind_url_la)

print(texterella[:500] + '\n\n[...]\n\n' + texterella[-500:])
```


```python
def extract_story(text, story_start, story_end):
    # Given some text, extract the part between (and including) story_start and story_end
    start_i = text.find(story_start)
    end_i = text.find(story_end) + len(story_end)
#     text = re.sub(f'(.*)({story_start})', r'\2', text, flags=re.DOTALL)
#     text = re.sub(f'({story_end})(.*)', r'\1', text, flags=re.DOTALL)
    return text[start_i: end_i]
```


```python
story_start = 'A rich man\'s wife'
story_end = 'they were punished \nwith blindness as long as they lived.'
texterella = extract_story(texterella, story_start, story_end)

print(texterella[:500] + '\n\n[...]\n\n' + texterella[-500:])
```

    A rich man's wife became sick, and when she felt that her end was
    drawing near, she called her only daughter to her bedside and said, "Dear
    child, remain pious and good, and then our dear God will always protect
    you, and I will look down on you from heaven and be near you." With this
    she closed her eyes and died.
    The girl went out to her mother's grave every day and wept, and she
    remained pious and good. When winter came the snow spread a white cloth
    over the grave, and when the spring su

    [...]

    o share her good fortune.
    When the bridal couple walked into the church, the older sister walked on
    their right side and the younger on their left side, and the pigeons
    pecked out one eye from each of them. Afterwards, as they came out of the
    church, the older one was on the left side, and the younger one on the
    right side, and then the pigeons pecked out the other eye from each of
    them. And thus, for their wickedness and falsehood, they were punished
    with blindness as long as they lived.


Hmmm... this is looking a bit Grimm, and doesn't really match the narrative arc that Vonnegart described. Here's another version I found that looks better suited to this project:


```python
cind_url_la = 'https://www.shortkidstories.com/story/cinderella-2/'
texterella = get_text_from_url(cind_url_la)
story_start = 'Cinderella’s mother died'
story_end = 'glad that he had found the glass slipper.'
texterella = extract_story(texterella, story_start, story_end)

print(texterella[:500] + '\n\n[...]\n\n' + texterella[-500:])
```

    Cinderella’s mother died while she was a very little child, leaving her to the care of her father and her step-sisters, who were very much older than herself; for Cinderella’s father had been twice married, and her mother was his second wife. Now, Cinderella’s sisters did not love her, and were very unkind to her. As she grew older they made her work as a servant, and even sift the cinders; on which account they used to call her in mockery “Cinderella.” It was not her real name, but she became a

    [...]

    ed, the Fairy godmother suddenly entered the room, and placing her godchild’s hand in the Prince’s, said:“Take this young girl for your wife, Prince; she is good and patient, and as she has known how to submit to injustice meekly, she will know how to reign justly.”So Cinderella was married to the Prince in great state, and they lived together very happily. She forgave her sisters, and treated them always very kindly, and the Prince had great cause to be glad that he had found the glass slipper.


# Data Inspection

We'll split this text up into 100 chunks of words, and calculate the mean sentiment for each chunk.


```python
def text_to_df(text):
    words = word_tokenize(text)
    window_size = len(words) // 100
    df = pd.DataFrame()
    df['Percent'] = np.arange(100)
    df['Words'] = [ words[window_size*i: window_size*(i+1)]
                   for i in range(100) ]
    df['Sentiment'] = df['Words'].apply(lambda window:
                                        np.mean([ get_sentiment(word) for word in window ] ) )
    return df
```


```python
cinderella_df = text_to_df(texterella)
cinderella_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Percent</th>
      <th>Words</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[Cinderella, ’, s, mother, died, while, she, was, a, very, little, child, ,, leaving, her, to]</td>
      <td>-0.015625</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>[the, care, of, her, father, and, her, step-sisters, ,, who, were, very, much, older, than, herself]</td>
      <td>0.070312</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>[;, for, Cinderella, ’, s, father, had, been, twice, married, ,, and, her, mother, was, his]</td>
      <td>-0.007812</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>[second, wife, ., Now, ,, Cinderella, ’, s, sisters, did, not, love, her, ,, and, were]</td>
      <td>-0.023438</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>[very, unkind, to, her, ., As, she, grew, older, they, made, her, work, as, a, servant]</td>
      <td>0.070312</td>
    </tr>
  </tbody>
</table>
</div>



Let's look at the points where the highest and lowest sentiments occur:


```python
pd.options.display.max_colwidth = 200
cinderella_df['Text'] = cinderella_df['Words'].apply(lambda x: ' '.join(x))
```


```python
cinderella_df.sort_values('Sentiment').tail()[['Percent', 'Sentiment', 'Text']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Percent</th>
      <th>Sentiment</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.070312</td>
      <td>the care of her father and her step-sisters , who were very much older than herself</td>
    </tr>
    <tr>
      <th>57</th>
      <td>57</td>
      <td>0.078125</td>
      <td>envious eyes , and knew that they wished they were as beautiful , and as well-dressed</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.085938</td>
      <td>well known by it that her proper one has been forgotten.She was a very sweet-tempered ,</td>
    </tr>
    <tr>
      <th>65</th>
      <td>65</td>
      <td>0.093750</td>
      <td>have new dresses , for she is so splendid . She makes every one look shabby.</td>
    </tr>
    <tr>
      <th>45</th>
      <td>45</td>
      <td>0.117188</td>
      <td>amused to hear them admire her grace and beauty , and say that they were sure</td>
    </tr>
  </tbody>
</table>
</div>




```python
cinderella_df.sort_values('Sentiment').head()[['Percent', 'Sentiment', 'Text']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Percent</th>
      <th>Sentiment</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>70</th>
      <td>70</td>
      <td>-0.093750</td>
      <td>out of his sight , and Cinderella , who was getting a little spoiled by all</td>
    </tr>
    <tr>
      <th>89</th>
      <td>89</td>
      <td>-0.078125</td>
      <td>be a Princess tried to put it on , but in vain . Cinderella ’ s</td>
    </tr>
    <tr>
      <th>86</th>
      <td>86</td>
      <td>-0.054688</td>
      <td>and as he felt sure that no one else could wear such a tiny shoe as</td>
    </tr>
    <tr>
      <th>55</th>
      <td>55</td>
      <td>-0.054688</td>
      <td>he asked her to dance , and would have no other partner , and as he</td>
    </tr>
    <tr>
      <th>90</th>
      <td>90</td>
      <td>-0.046875</td>
      <td>sisters tried , but could not get it on , and then Cinderella asked if she</td>
    </tr>
  </tbody>
</table>
</div>



Some of these make sense, others not so much. On the whole, I think this will be acceptable, but it will certainly be worth trying other sentiment analysis tools in the future.

# The Sentiment Plot

We now need to decide how to plot the sentiment over the course of the story. For Cinderella, the plot Vonnegart drew looked something like this:


```python
def cinderella_f(x):
    if x < 20:
        return -1
    elif 20 <= x < 40:
        return (x-20)//4/4 - 1
    elif 40 <= x < 60:
        return 1-((x-60)/20)**2
    elif 60 <= x < 80:
        return -0.5
    elif 80 <= x < 100:
        return 1 / (99.5-x) - 0.5 - 1/(101-80)

vonnegart_df = cinderella_df[['Percent']]
vonnegart_df.loc[:, 'Sentiment'] = vonnegart_df['Percent'].apply(cinderella_f)
vonnegart_df.loc[:, 'Curve'] = 'Vonnegart Curve'
sns.relplot(x='Percent', y='Sentiment', kind='line', data=vonnegart_df)
plt.title('Vonnegart');
```


![png](/assets/images/2020-01-31-shape-of-stories/output_25_0.png)


The progress goes something like this:
 * 0-20%: Cinderella's mother has died and is forced to do nasty chores.
 * 20-40%: The fairy godmother gives Ciderella lots of nice clothes, makeup, and dresses so she can go to the ball.
 * 40-60%: Cinderella dances with the Prince and has a wonderful time.
 * 60-80%: After the midnight bell rings, she goes back down to a low valence. But not *as* low as she was originally because now she's got a wonderful memory.
 * 80-100%: Cinderella marries hte prince and lives happilly ever after.

Let's now compare this to the empirical sentiment over time.


```python
sns.relplot(x='Percent', y='Sentiment', kind='line', data=cinderella_df);
```


![png](/assets/images/2020-01-31-shape-of-stories/output_28_0.png)


# Smoothing the Sentiment Plot

The empirical sentiment plot is so jagged it's hard to discern any overall pattern. Let's smooth out the curve so that we can get a better sense of some high-level trends.

### Sliding Window

First, let's try running a sliding window across the sentiments. The width of the window is a hyperparameter we need to tune, so I plotted a few reasonable sounding values to see which looks best.


```python
def sliding_window(x, window_size):
    ret = []
    for i in range(1, 101):
        pad_factor = int( i/100 * window_size )
        lower = max(0, i-window_size//2)
        upper = min(100, i+window_size//2)
#         lower = max(0, i-window_size+pad_factor)
#         upper = min(101, i+pad_factor)
        ret.append(np.mean(x[lower:upper]))
    return ret

windows_df = pd.DataFrame(columns=['Percent', 'Sentiment', 'Window Size', 'Curve'])

for window_size in [2, 5, 10, 25]:
    window_df = cinderella_df[['Percent']]
    window_df.loc[:, 'Sentiment'] = sliding_window(cinderella_df['Sentiment'], window_size)
    window_df.loc[:, 'Window Size'] = f'{window_size}%'
    window_df.loc[:, 'Curve'] = 'Sliding Window'

    windows_df = windows_df.append(window_df)

sns.relplot(x='Percent', y='Sentiment', col='Window Size', kind='line', data=windows_df);
```

<img src="/assets/images/2020-01-31-shape-of-stories/output_30_0.png" alt="All the stories!" width="100%"/>


Window sizes of 5%, 10%, and 25% all look reasonable. I decided to go with the middle one of these: 10%.

### EWMA

Another approach to smoothing out the graph would be to use an exponentially weighted moving average (EWMA). This has the nice property that the contribution of words to the current sentiment value decays exponentially as you move through time. If the sentiment in the current window is given by $s_i$, then the EWMA sentiment value is given by

$$ S_i = \alpha \cdot S_{i-1} + (1-\alpha)\cdot s_i. $$

Again, we have a hyperparameter: $\alpha$, the decay constant. And again, I plotted a few reasonable sounding values to see what looks best.


```python
def ewma(x, alpha):
    S = x[0]
    ret = [S]
    for x_i in x[1:]:
        S = alpha * S + (1-alpha) * x_i
        ret.append(S)
    return ret

ewmas_df = pd.DataFrame(columns=['Percent', 'Sentiment', 'Alpha', 'Curve'])

for alpha in [0.25, 0.5, 0.75, 0.9]:
    ewma_df = cinderella_df[['Percent']]
    ewma_df.loc[:, 'Sentiment'] = ewma(cinderella_df['Sentiment'], alpha)
    ewma_df.loc[:, 'Alpha'] = alpha
    ewma_df.loc[:, 'Curve'] = 'EWMA'

    ewmas_df = ewmas_df.append(ewma_df)

sns.relplot(x='Percent', y='Sentiment', col='Alpha', kind='line', data=ewmas_df);
```

<img src="/assets/images/2020-01-31-shape-of-stories/output_33_0.png" alt="All the stories!" width="100%"/>


I think $\alpha=0.75$ is probably the best of these. But not as good as the sliding window with window size = 10%, so I've used that one from now on. Let's now put that on the same axes as Vonnegart's plot, and see how well it matches up. Note that the range of the sliding window sentiments is very small, so we have to normalise it.


```python
window_df = windows_df[ windows_df['Window Size'] == '10%' ]
min_sent = window_df['Sentiment'].min()
max_sent = window_df['Sentiment'].max()
window_df.loc[:, 'Sentiment'] = window_df['Sentiment'].apply(lambda x: (x-min_sent) / (max_sent - min_sent) * 2 - 1)
cinderella_df_2 = window_df.append(vonnegart_df)

sns.relplot(x='Percent', y='Sentiment', hue='Curve', style='Curve', kind='line', data=cinderella_df_2);
```


![png](/assets/images/2020-01-31-shape-of-stories/output_35_0.png)


Not too bad. The main problem I see with this is that the empirical sentiment starts at a high point, whereas the Vonnegart curve starts at a low point. Looking back at the first paragraph, the opening is a little ambiguous:
> A rich man's wife became sick, and when she felt that her end was
drawing near, she called her only daughter to her bedside and said, "Dear
child, remain pious and good, and then our dear God will always protect
you, and I will look down on you from heaven and be near you." With this
she closed her eyes and died.

# All The Stories!

Now that we've got our sentiment plotting procedure down, we can plot all the kinds of stories mentioned  in Vonnegart's talk. In addition to Cinderella, we've got
 * Man in hole: the protagonist starts in a comfortable environment, gets into trouble, then gets out again richer for the experience. I used The Hobbit as a typical example of this type of story, and approximated Vonnegart's curve as a cosine with linearly increasing amplitude.
 * Boy meets girl: a boy meets a girl, is over the moon, everything goes to custard, but then he gets her back again and everything is wonderful. I had a hard time thinking of an example of this genre. I eventually settled on Jane Eyre, which doesn't quite fit the mould but seems to be close enough. I modelled Vonnegart's curve as a sine with linearly increasing amplitude.
 * Kafka: not very pleasant man turns into a bug and everything is terrible forever. Easy enough to model as negative parabola with y-intercept of -0.5.
 * Hamlet: a bunch of stuff happens, but it's never clear if it's good or bad. So we stay at -0.5 throughout the whole story.

Let's see how the Vonnegart plots compare to the empirical sentiment for each of these stories.


```python
def hamlet_f(x):
    return -0.5

hamlet_url = 'http://shakespeare.mit.edu/hamlet/full.html'
hamlet_text = get_text_from_url(hamlet_url)
words_words_words = extract_story(hamlet_text, 'ACT I', 'Go, bid the soldiers shoot.')

def kafka_f(x):
    if x < 20:
        return -0.5 - (x/20)**2
    else:
        return None

kafka_url = 'https://www.gutenberg.org/cache/epub/5200/pg5200.txt'
kafka_text = get_text_from_url(kafka_url)
kafka_text = extract_story(kafka_text, 'One morning', 'stretch out her young body.')

def man_in_hole_f(x):
    return np.cos(x*2*np.pi/100) * (0.5 + 0.5*x/100)

hobbit_url = 'https://archive.org/stream/TheHobbitByJRRTolkienEBOOK/The%20Hobbit%20byJ%20%20RR%20Tolkien%20EBOOK_djvu.txt'
hobbit_text = get_text_from_url(hobbit_url)
hobbit_text = extract_story(hobbit_text, 'Chapter I', 'handed him the tobacco-jar.')

def boy_girl_f(x):
    return np.sin(x*2.5*np.pi/100) * (0.5 + 0.5*x/100)

jane_eyre_url = "http://gutenberg.org/files/1260/1260-h/1260-h.htm"
jane_eyre_text = get_text_from_url(jane_eyre_url)
jane_eyre_text = extract_story(jane_eyre_text, 'CHAPTER I', 'Amen; even so come, Lord\r\nJesus!’”')

story_fs = [cinderella_f, hamlet_f, kafka_f, man_in_hole_f, boy_girl_f]
story_texts = [texterella, words_words_words, kafka_text, hobbit_text, jane_eyre_text]
story_names = ['Cinderella', 'Hamlet', 'Kafka', 'Man in Hole', 'Boy meets Girl']
```


```python
storys_df = pd.DataFrame(columns=['Progress', 'Sentiment', 'Curve', 'Story'])
for f, story, story_name in zip(story_fs, story_texts, story_names):

    # Empirical plot
    story_df = text_to_df(story)
    story_df.loc[:, 'Sentiment'] = sliding_window(story_df['Sentiment'], 10)
    story_df.loc[:, 'Curve'] = 'Empirical'
    sent_min = story_df['Sentiment'].min()
    sent_max = story_df['Sentiment'].max()
    story_df.loc[:, 'Sentiment'] = story_df['Sentiment'].apply(lambda x: (x-sent_min) * 2 / (sent_max - sent_min) - 1)


    # Vonnegart plot
    vonnegart_df = story_df.copy()
    vonnegart_df.loc[:, 'Sentiment'] = vonnegart_df['Percent'].apply(f)
    vonnegart_df.loc[:, 'Curve'] = 'Vonnegart Curve'

    story_df = story_df.append(vonnegart_df)
    story_df.loc[:, 'Story'] = story_name

    storys_df = storys_df.append(story_df)

sns.relplot(x='Percent', y='Sentiment', col='Story', hue='Curve', style='Curve', kind='line', data=storys_df);
```


<img src="/assets/images/2020-01-31-shape-of-stories/output_39_0.png" alt="All the stories!" width="100%"/>



Observations:
 * I can't spell "stories".
 * The Man in Hole plot and Boy Meets Girl plots are actually pretty good. With a little tinkering with the phase of the Vonnegart curves, they could probably match the empirical curves pretty well.
 * I've just noticed that the Cinderella curve is actually an instance of the Boy Meets Girl curve.
 * The Kafka curve is way off, but that's not really surprising as that one seemed more like a joke.
 * The Hamlet curve isn't too bad, although a negative Man in Hole curve would probably suit it better.
 * For all of these curves, it looks like there's an oscillation which goes right through the story. This oscillation can change amplitude or gain an additive constant, but keeps a constant period. Which makes me what to do a Fourier analysis on these curves. Perhaps they can be modelled by the sum of two sines?

I was curious what the tall peak corresponded to in the Hamlet plot, so I wrote a function to look at the text for the top $n$ peaks or troughs of a given curve.


```python
def get_extremes(story_name, n_peaks=5, mode='peaks'):

    # Sort the story segments by sentiment
    if mode=='peaks':
        ascending = False
    elif mode=='troughs':
        ascending = True
    else:
        raise ValueError('mode should be "peaks" or "troughs"')
    story_df = storys_df[(storys_df['Story']==story_name) & (storys_df['Curve']=='Empirical')]
    story_df.sort_values('Sentiment', inplace=True, ascending=ascending)

    # This loop makes sure that each discrete peak is only represented once
    top_lines = pd.DataFrame(columns=story_df.columns)
    for i in range(n_peaks):
        story_df.reset_index(drop=True, inplace=True)
        top_row = story_df.loc[0, :]
        top_lines.loc[i, :] = top_row
        percentage = top_row['Percent']

        # exclude all other rows within 10 percentage points of this peak
        story_df = story_df[ (story_df['Percent'] > percentage + 10) | (story_df['Percent'] < percentage - 10) ]

    # Make the dataframe look pretty
    def words_to_text(words):
        text = ' '.join(words)
        text_len = 160
        if len(text) > text_len:
            half_len = text_len//2
            text = text[:half_len] + ' ... ' + text[-half_len:]
        return text
    top_lines['Text'] = top_lines['Words'].apply(words_to_text)
    top_lines = top_lines[['Percent', 'Sentiment', 'Text']]

    return top_lines
```


```python
get_extremes('Hamlet', mode='peaks')
```

    /anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:11: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      # This is added back by InteractiveShellApp.init_path()





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Percent</th>
      <th>Sentiment</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26</td>
      <td>1</td>
      <td>GERTRUDE Thanks , Guildenstern and gentle Rosencrantz : And I beseech you instan ... rom Norway , and in fine Makes vow before his uncle never more To give the assay</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>0.230961</td>
      <td>do not that way tend ; Nor what he spake , though it lack 'd form a little , Was ...  a robustious periwig-pated fellow tear a passion to tatters , to very rags , to</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.180527</td>
      <td>ACT I SCENE I. Elsinore . A platform before the castle . FRANCISCO at his post . ...  of heaven Where now it burns , Marcellus and myself , The bell then beating one</td>
    </tr>
    <tr>
      <th>3</th>
      <td>78</td>
      <td>0.127854</td>
      <td>how otherwise ? -- Will you be ruled by me ? LAERTES Ay , my lord ; So you will  ... ING CLAUDIUS He made confession of you , And gave you such a masterly report For</td>
    </tr>
    <tr>
      <th>4</th>
      <td>66</td>
      <td>0.0867992</td>
      <td>you me for a sponge , my lord ? HAMLET Ay , sir , that soaks up the king 's coun ...  ? HAMLET At supper . KING CLAUDIUS At supper ! where ? HAMLET Not where he eats</td>
    </tr>
  </tbody>
</table>
</div>



Looks like it's the bit where Guidenstern and Rosencratz arrive at Elsinore. Is that a high point of the story? I don't remember it standing out, but I don't know Hamlet super well.

# Conclusion

So what have we learned here? We've learned that using a sliding window with a window size of 10% of the text does smooth out the sentiment curve pretty well so that you can make out the arc of the story. We've learned that we can sometimes make out the kinds of curves that Vonnegart talks about in stories - as in the cases of Cinderella, The Hobbit, and Jane Eyre - but not always - as in the cases of Hamlet and Kafka.

I've got several ideas about how to extend this work. As mentioned earlier, it's probably worth exploring some other sentiment analysis technologies, to see if I can get sentiment scores that line up with intuition better. I also mentioned earlier that I'd be interested in doing Fourier analysis of these story plots, to see what kinds of cycles there are, and whether stories can be modelled by a pair of sinusoids.

I'm also interested in creating "shape of story" plots for a wider range of stories. One that I think would be really interesting is the web serial Worm by J.C. McCrae. [Worm](https://parahumans.wordpress.com/) is a whopper of a book, at 6,000 pages if it was physically printed. It’s also really grim, but it somehow manages to keep getting grimmer as the book progresses. Does this show up in the plot?
