---
title: "Visualising the History of an Organisation"
published: True
---
### Or, Data Analysis of the Varsity Toastmasters Club

A couple of years back I joined Varsity Toastmasters - a club at my university, which provides training in public speaking. Last year we celebrated our club's 20th anniversary, so I decided to pull data on the club from the Toastmasters International website (Toastmasters International is the parent organisation of Varsity Toastmasters), and give a presentation on some statistics and visualisations of the club's history through its data.

I recently decided to tidy up my visualisations and make them into a blog post. There are a lot of aspects of an organisation which you might want to get insights on, and it can often be difficult to visually convey these aspects. In this post we will explore visualisations of
 * People entering or leaving the organisation over time
 * The length of people's stay within the organisaiton
 * Current and cumulative total membership of the organisation
 * The succession of roles people take on within the organisation
It turns out that there are a lot of nice ways to plot these various aspects of an organisation. Each plot has its own strengths and weaknesses, and no single plot can tell you everything.

To start off, let's load up the libraries and data we'll need, and look at the dtypes of the data columns. I won't show any of the actual data here for privacy reasons.


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
import matplotlib
import datetime as dt
import seaborn as sns
from IPython.display import display
from collections import defaultdict

plt.style.use('ggplot')
```

The first table gives all the members who have been a part of the club, along with when they joined and when they quit. Well, not _all_ the members. The online system seems to have been phased in after 2000, so the early data is a bit patchy.


```python
members = pd.read_excel('Past-and-Present-Members-Jun.-9-2019.xlsx', header=1, parse_dates=['Begin', 'End'])
members.dtypes
```




    Member ID                                                                          int64
    Name                                                                              object
    Address                                                                           object
    Member has opted-out of Toastmasters WHQ mail marketing communication             object
    Phone                                                                             object
    Member has opted-out of Toastmasters WHQ phone marketing communication            object
    Email Address                                                                     object
    Member has opted-out of Toastmasters WHQ email marketing communication            object
    Begin                                                                     datetime64[ns]
    End                                                                       datetime64[ns]
    dtype: object



The second table gives all the club's "officers" and when they were in office. Officers have such grandiose titles as "president", "sergeant at arms" (preparer of snacks), and "vice president public relations" (facebook page updater).


```python
officers = pd.read_excel('Past and Present Club Officers - Aug. 4, 2019.xlsx', header=1, parse_dates=['Start', 'End'])
officers.dtypes
```




    Member ID                                                                          int64
    Name                                                                              object
    Address                                                                           object
    Member has opted-out of Toastmasters WHQ mail marketing communication             object
    Phone                                                                             object
    Member has opted-out of Toastmasters WHQ phone marketing communication            object
    Email Address                                                                     object
    Member has opted-out of Toastmasters WHQ email marketing communication            object
    Position                                                                          object
    Start                                                                     datetime64[ns]
    End                                                                       datetime64[ns]
    dtype: object



At Toastmasters you also get "education awards" as you move through the curriculum. The last table gives details about these awards.


```python
awards = pd.read_excel('Past and Present Education Awards - Aug. 4, 2019.xlsx', header=1, parse_dates=['Award Date'])
awards.dtypes
```




    Member ID               int64
    Member Name            object
    Award Name             object
    Award Date     datetime64[ns]
    dtype: object



Now that we've loaded the data, let's get stuck into some visualisation.

## Membership Begin and End Dates

The first thing I want to look at is the number of members entering or leaving the club each year. Were there any mass exoduses? Were there any years where the club was unusually good at recruitment? Seaborn's `jointplot` seems like the ideal tool for this.


```python
# Have ticks on alternate years
min_year = min(members.Begin.dt.year)
max_year = max(members.End.dt.year)+2
ticks = np.arange(min_year, max_year, 2)

def dts_to_float(dts):
    # Convert a datetime series to a float series.
    # Seaborn is not good at doing this by itself.
    return dts.dt.year + dts.dt.month/12 + dts.dt.day/365

# Point coordinates
begin_years = dts_to_float(members.Begin)
end_years = dts_to_float(members.End)

# Have one bin per year
bins = np.arange(min_year, max_year, 1)

# Create plot
g = sns.jointplot(x=begin_years, y=end_years, alpha=0.2, marginal_kws=dict(bins=bins))
plt.subplots_adjust(top=0.93)
g.ax_joint.set_xticks(ticks)
g.ax_joint.set_xlabel('Membership Begin Date')
g.ax_joint.set_ylabel('Membership End Date')
g.ax_joint.set_yticks(ticks);
```


<img src="/assets/images/2020-02-10-org-visualisation/output_11_0.png"/>


Notice how the $x=y$ line is clearly visible. Points can't go below this line because you can't begin your membership before you end it. The greater the verticle height from the $x=y$ line, the longer that person was a member. It looks like the longest membership lasted from 2005 to 2013! The biggest exodus occurred in 2010, which probably prompted a big recruitment drive in 2011 where we see the largest uptake of new members.

## Tenure

How long do members stick with the club? Here's a histogram of member "tenure":


```python
members['Tenure'] = members.End - members.Begin

sns.distplot(members['Tenure'].dt.days / 365, bins=np.arange(0, 9, 0.5), kde=False)
plt.ylabel('Count')
plt.xlabel('Years')
plt.title('Member Tenure');
```


<img src="/assets/images/2020-02-10-org-visualisation/output_14_0.png"/>


Each bin is one semester long. So most members drop out after only one semester, and after 3 semester's almost everyone is done, although there have been a few hardcore members who've stuck around for many years. The longest membership was 7 years; no doube the 2005 to 2012 member we noted from the previous plot.

## Education Awards

Let's look at what education awards were obtained in each year. This data isn't all that interesting, so we won't do anything with the educational awards after this.


```python
award_names = awards['Award Name'].unique()
bottom = np.zeros(len(award_names))
cmap = matplotlib.cm.get_cmap('hsv')
for i, award_name in enumerate(award_names):
    award_data = awards[ awards['Award Name']==award_name ]
    award_counts = award_data['Award Date'].dt.year.value_counts().reindex(range(2000, 2020))
    shape_color = cmap( i/(len(award_names)) )
    shape_bar = award_counts.plot(bottom=bottom, kind='bar', color=shape_color, label=award_name)
    bottom += award_counts.fillna(0)

plt.legend(bbox_to_anchor=(1.05, 0.9), ncol=2, title='Award Name')
plt.ylabel('Count')
plt.title('Education Awards by Year')

# Rescale axes
ax = plt.gca()
ax.relim()
ax.autoscale_view()

plt.show()
```


<img src="/assets/images/2020-02-10-org-visualisation/output_17_0.png" width="100%"/>


The only thing to note here is that the curriculum changed by 2019, so a different set of education awards were available.

## Total Membership

How many members has Varsity Toastmasters had over its lifetime? How many presidents? How many officers? Let's take a look at a cumulative histogram.


```python
member_begins = pd.DataFrame({'date': dts_to_float(members['Begin'])})
member_begins['Type'] = 'Member'

member_ends = pd.DataFrame({'date': dts_to_float(members['End'])})
member_ends['Type'] = 'ex-Member'

officer_begins = pd.DataFrame({'date': dts_to_float(officers['Start'])})
officer_begins['Type'] = officers.Position

president_begins = officer_begins[officer_begins['Type']=='President']
president_begins['Type'] = 'President'

officer_begins['Type'] = 'Officer'

all_begins = pd.concat([member_begins, member_ends, officer_begins, president_begins])

kwargs = {'cumulative': True, 'alpha': 1}

# Plot each type of member
for type_ in all_begins.Type.unique():
    ax = sns.distplot(all_begins[all_begins['Type']==type_].date, bins=100, hist_kws=kwargs, kde=False, label=type_)



# Prettify the plot
ax.legend()
ax.set_xticks(list(range(2000, 2022, 2)))
plt.xlabel('Year')
plt.ylabel('Total')
plt.title('Cumulative Membership');
```

    /anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:11: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      # This is added back by InteractiveShellApp.init_path()



<img src="/assets/images/2020-02-10-org-visualisation/output_20_1.png"/>


Several things to note:
 * The gap between the the `Member` and `ex-Member` cumulative distributions (i.e., the height of the visible red section) gives the total _current members_ at a given time.
 * We can see the spike in recruitment and terminated memberships around 2011. But it looks like the recruitment spike precedes the termination spike, which is the opposite of what it looked like in the previous plot?
 * Over the course of 20 years, we've gone through about 350 members, 200 officers, and 40 presidents. That's about 20 members, 10 officers, and 2 president per year, which sounds about right: Toastmasters clubs tend to have 10-20 people, there are seven officer roles, and officers and members turnover on a roughly twice-a-year basis.

## Officer Transitions

Let's now look at how members tend to move through the ranks of officer positions. We'll do this by creating calculating a transition matrix for officer roles.


```python
members = officers['Member ID'].unique()
roles = ['Non-Officer', 'Sergeant at Arms', 'Secretary', 'Treasurer',
     'Vice President Public Relations', 'Vice President Membership',
     'Vice President Education', 'President']
counts = pd.DataFrame(0, columns=roles, index=roles)#np.zeros((len(roles), len(roles)))
for member in members:
    member_roles =  officers[officers['Member ID']==member].sort_values('Start')['Position']
    member_roles = ['Non-Officer'] + list(member_roles) + ['Non-Officer']
    for i in range(len(member_roles)-1):
        r1 = member_roles[i]
        r2 = member_roles[i+1]
        counts.loc[r1, r2] += 1

# Normalise rows to make transition matrix
transition_matrix = counts.div(counts.sum(axis=1), axis=0)
transition_matrix
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
      <th>Non-Officer</th>
      <th>Sergeant at Arms</th>
      <th>Secretary</th>
      <th>Treasurer</th>
      <th>Vice President Public Relations</th>
      <th>Vice President Membership</th>
      <th>Vice President Education</th>
      <th>President</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Non-Officer</th>
      <td>0.000000</td>
      <td>0.193878</td>
      <td>0.153061</td>
      <td>0.183673</td>
      <td>0.204082</td>
      <td>0.132653</td>
      <td>0.102041</td>
      <td>0.030612</td>
    </tr>
    <tr>
      <th>Sergeant at Arms</th>
      <td>0.437500</td>
      <td>0.062500</td>
      <td>0.062500</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.187500</td>
      <td>0.093750</td>
    </tr>
    <tr>
      <th>Secretary</th>
      <td>0.343750</td>
      <td>0.093750</td>
      <td>0.125000</td>
      <td>0.093750</td>
      <td>0.062500</td>
      <td>0.062500</td>
      <td>0.093750</td>
      <td>0.125000</td>
    </tr>
    <tr>
      <th>Treasurer</th>
      <td>0.451613</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.225806</td>
      <td>0.064516</td>
      <td>0.064516</td>
      <td>0.096774</td>
      <td>0.096774</td>
    </tr>
    <tr>
      <th>Vice President Public Relations</th>
      <td>0.516129</td>
      <td>0.032258</td>
      <td>0.064516</td>
      <td>0.000000</td>
      <td>0.064516</td>
      <td>0.193548</td>
      <td>0.032258</td>
      <td>0.096774</td>
    </tr>
    <tr>
      <th>Vice President Membership</th>
      <td>0.363636</td>
      <td>0.060606</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.121212</td>
      <td>0.181818</td>
    </tr>
    <tr>
      <th>Vice President Education</th>
      <td>0.424242</td>
      <td>0.060606</td>
      <td>0.121212</td>
      <td>0.030303</td>
      <td>0.000000</td>
      <td>0.030303</td>
      <td>0.181818</td>
      <td>0.151515</td>
    </tr>
    <tr>
      <th>President</th>
      <td>0.500000</td>
      <td>0.088235</td>
      <td>0.058824</td>
      <td>0.029412</td>
      <td>0.058824</td>
      <td>0.058824</td>
      <td>0.000000</td>
      <td>0.205882</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots()
plt.imshow(transition_matrix)
plt.colorbar()
ax.xaxis.tick_top()
ax.set_xticks(range(len(roles)))
ax.set_yticks(range(len(roles)))
ax.set_yticklabels(roles)
ax.set_xticklabels(roles)
plt.xticks(rotation=90)
plt.show()
```


<img src="/assets/images/2020-02-10-org-visualisation/output_24_0.png"/>


Observations:
 * For a member in any given officer role, the most likely next state of the member is non-officer. The officers who are most likely to retire are presidents - which makes sense because at this stage you've reached the top of the hierarchy - and VP public relations - perhaps because this is a cushy role which non-serious newbies tend to get shuffled into.
 * The most popular starting officer role is VP public relations. From there, a member is most likely to transition to VP Membership (assuming they continue to be an officer), and from there to presidency.

## Individual Member Plots

Now we get to the interesting bit. I wanted to make a plot where we can see information about each member individually. This turned out to be surprisingly hard: there are so many members that to visually summarise what's going on with all of them normally results in the graph looking chaotic and ugly. But after several hours of trial-and-error, I eventually hit upon a visualisation that I absolutely love.


```python
members2 = members.sort_values(by=['End', 'Begin'], ascending=[False, True])

dates = sorted(np.append(members2.Begin.unique(), members2.End.unique()))

#############################################
# Remove dates which are too close together
# to prevent any gradients from being too
# steep.
#############################################

min_sep = np.timedelta64(2, 'M').astype('timedelta64[ns]') # 2 months
date_dic = {}
i = 0

while i < len(dates):

    date_i = dates[i]

    members2.loc[ (date_i < members2['Begin']) & (members2['Begin'] <= date_i + min_sep), 'Begin' ] = date_i
    members2.loc[ (date_i < members2['End']) & (members2['End'] <= date_i + min_sep), 'End' ] = date_i

    while i < len(dates) and dates[i] < date_i + min_sep:
        i += 1

dates = sorted(np.append(members2.Begin.unique(), members2.End.unique()), reverse=True)

#############################################
# Move the begin dates forward a bit so that
# if Alice finishes when Bob starts then
# we don't need to stack them on top of
# each other.
#############################################

members2['Begin'] += np.timedelta64(1, 'M')

tl_df = pd.DataFrame(columns=['Index', 'Pos', 'Date', 'Member ID', 'Tenure']) # NOTE: if you use 'Member ID' rath than index, then disconintuities in members2hip get lines which is really messy

dates = sorted(np.append(members2.Begin.unique(), members2.End.unique()), reverse=True)
for i, dt_i in enumerate(dates[:-1]):

    dt_j = dates[i+1]

    current_members2 = members2[ (~(members2['Begin'] > dt_i) & ~(members2['End'] < dt_i)) ]

    dldf_i = pd.DataFrame({'Index': list(current_members2.index), 'Pos': np.arange(len(current_members2)),
                          'Member ID': current_members2['Member ID'],
                          'Tenure': current_members2['Tenure'].dt.days / 365,
                          'Begin': current_members2['Begin'].dt.year + current_members2['Begin'].dt.month/12 + current_members2['Begin'].dt.day/365})
    dldf_i['Date'] = dt_i

    tl_df = tl_df.append(dldf_i)

#     dt_i = dt_j
```


```python
fig, ax = plt.subplots(figsize=(80,10))
sns.axes_style("darkgrid")
g = sns.lineplot(x='Date', y='Pos', units='Index', hue='Begin', palette="hsv", estimator=None, data=tl_df);
plt.gca().set_ylabel('Member Count');
```


<img src="/assets/images/2020-02-10-org-visualisation/output_28_0.png" width="100%"/>


Here's what's going on in this plot. Each line represents a member, spanning from when the member joins to when they leave. The lines are all stacked on top of each other, so that you can tell how many members there are by looking at how high the stack is at a given time. Peak membership occurred in 2010, at which point it briefly exceeded 40 people. Membership has only been as low as it currently is twice before: in 2015, and when the club was founded in 2001.

The members are sorted vertically so that the member who will leave the soonest is at the top. This is why you can see several members snaking their way from the bottom of the pile up to the top, at which point they can evaporate away. I didn't sort them this way because it was particularly informative, but because of all the different ways of sorting I tried out, this one looked hands down the best. Members tend to come and go in waves, and with this sorting the membership plot actually looks like a bank of waves!

What about the colours? I wanted to be able to quickly discern which members had been around the longest at any given time. At first I just tried making the longer lines thicker, but then you couldn't tell if a line was thick because it has a long future or a long past. Ideally a member's line would become more distinctive the longer the member had been around. Making each line grow in thickness from left to right would have achieved this, but this would be a pain to do with Seaborn.

Eventually it occurred to me to set a member's colour according to when they joined the club. That member then becomes increasingly distinctive as the newer members gradually change colour around them. And as an added bonus, you can tell at a glance _when_ a member joined, _and_ which members are from the same "generation".

As I said before, it took many hours of iterating to arrive at the above plot. Some of these iterations were too good to throw out, so before moving on, let's look at... _the fire plot_:


```python
fig, ax = plt.subplots(figsize=(80,10))
sns.set(rc={'axes.facecolor':'gray', 'figure.facecolor':'gray'})
g = sns.lineplot(x='Date', y='Pos', units='Index', hue='Begin', palette="Reds_r", estimator=None, data=tl_df);
plt.gca().set_ylabel('Member Count');
```


<img src="/assets/images/2020-02-10-org-visualisation/output_30_0.png" width="100%"/>


and _this_ plot, which somehow reminds me both of [Escher](https://www.wikiart.org/en/m-c-escher/metamorphosis-iii-excerpt-3-1968) and of that famous Japanese [print of a wave](https://en.wikipedia.org/wiki/The_Great_Wave_off_Kanagawa)


```python
fig, ax = plt.subplots(figsize=(80,10))
sns.axes_style("darkgrid")
g = sns.lineplot(x='Date', y='Pos', units='Index', color='navy', size='Begin', sizes=(5, 0.1), estimator=None, data=tl_df);
# g.set(axis_bgcolor='k')
plt.gca().set_ylabel('Member Count');
```


<img src="/assets/images/2020-02-10-org-visualisation/output_32_0.png" width="100%"/>


This last plot is interesting because it's looks so three dimensional.


```python
fig, ax = plt.subplots(figsize=(80,10))
sns.axes_style("darkgrid")
g = sns.lineplot(x='Date', y='Pos', units='Index', color='navy', size='Tenure', sizes=(0.1, 4), estimator=None, data=tl_df);
# g.set(axis_bgcolor='k')
plt.gca().set_ylabel('Member Count');
```


<img src="/assets/images/2020-02-10-org-visualisation/output_34_0.png" width="100%"/>


## Individual Officer Timelines

I wanted to do a similar thing to above, but include information about who was holding officer roles. Again, this was really hard to do without making it ugly, but after hours of trial and error I finally came across something that looks nice.


```python
members2 = members.copy()
officers2 = officers.copy()

dates = sorted(np.concatenate([members2.Begin.unique(), members2.End.unique(), officers2.Start.unique(), officers2.End.unique()]))
# temp = members2[members2['Begin'].dt.year < 2005]
# dates = sorted(np.concatenate(temp.Begin.unique(), temp.End.unique(), officers2.Start.unique(), officers2.End.unique()))

#############################################
# Remove dates which are too close together
# to prevent any gradients from being too
# steep.
#############################################

min_sep = np.timedelta64(6, 'M').astype('timedelta64[ns]') # 2 months
i = 0

dates2 = []

while i < len(dates):

    date_i = dates[i]
    dates2.append(date_i)

    members2.loc[ (date_i < members2['Begin']) & (members2['Begin'] <= date_i + min_sep), 'Begin' ] = date_i
    members2.loc[ (date_i < members2['End']) & (members2['End'] <= date_i + min_sep), 'End' ] = date_i

    officers2.loc[ (date_i < officers2['Start']) & (officers2['Start'] <= date_i + min_sep), 'Start' ] = date_i
    officers2.loc[ (date_i < officers2['End']) & (officers2['End'] <= date_i + min_sep), 'End' ] = date_i

    while i < len(dates) and dates[i] < date_i + min_sep:
        i += 1

dates = dates2

#############################################
# Move the begin dates forward a bit so that
# if Alice finishes when Bob starts then
# we don't need to stack them on top of
# each other.
#############################################

members2['Begin'] += np.timedelta64(3, 'M')
officers2['Start'] += np.timedelta64(3, 'M')

members2.sort_values(by=['Begin', 'End'], ascending=[True, False], inplace=True)

tl_df = pd.DataFrame(columns=['Index', 'Pos', 'Date', 'Member ID', 'Tenure']) # NOTE: if you use 'Member ID' rath than index, then disconintuities in members2hip get lines which is really messy

# officer timeline dataframe
of_tldf = pd.DataFrame(columns=['Index', 'Date', 'Member ID', 'Position'])

dates = sorted(np.concatenate([members2.Begin.unique(), members2.End.unique(), officers2.Start.unique(), officers2.End.unique()]))

class counterdict(defaultdict):
    def __missing__(self, key):
        self[key] = len(self)
        return self[key]

pos_dic = defaultdict(counterdict)

for i, date in enumerate(dates[:-1]):

    # Current members

    current_members = members2[~(members2.End<dates[i]) & ~(dates[i+1]<members2.Begin)]

    for k in [i, i+1]:

        date_ = dates[k]

        tl_df_i = pd.DataFrame({
            'Index': current_members.index,
            'Pos': [ pos_dic[k][j] for j in current_members['Member ID']],
            'Date': [date_ for i in range(len(current_members))],
            'Tenure': current_members['Tenure'],
            'Member ID': current_members['Member ID']})

        tl_df = tl_df.append(tl_df_i)

    # Current officers2

    current_ofs = officers2[~(officers2.End<dates[i]) & ~(dates[i+1]<officers2.Start)]

    for date_ in dates[i: i+2]:

        of_tldf_i = pd.DataFrame({
            'Position': current_ofs.Position,
            'Index': current_ofs.index,
            'Date': [date_ for i in range(len(current_ofs))],
            'Member ID': current_ofs['Member ID']})

        of_tldf = of_tldf.append(of_tldf_i)



of_tldf['Date'] = of_tldf['Date'].astype('datetime64[ns]')
tl_df['Date'] = tl_df['Date'].astype('datetime64[ns]')
```


```python
# Membership timeline dataframe
mem_tldf = tl_df.copy()
mem_tldf['Position'] = 'Member'

# We want the officer lines to inherit vertical positions ('Pos') from
# membership lines (the pairing between the two is given by 'Member ID').
of_tldf = of_tldf.merge(tl_df[['Member ID', 'Pos', 'Date']], on=['Member ID', 'Date'], how='inner')

# The officer lines should be more prominent than the membership lines,
#   so I'm setting line thickness according to an 'Officer' column.
of_tldf['Officer'] = 'Yes'
mem_tldf['Officer'] = 'No'

# Combine the data for membership lines and officer lines into a single table.
tl_df2 = mem_tldf.append(of_tldf)

# We can have duplicates of 'Member ID' and 'Date'
#   if a member leaves but then rejoins the club within
#   one discrete time window.
tl_df3 = tl_df2.drop_duplicates(['Member ID', 'Date', 'Position'])

# Set the order of the officer roles so that the colour is more informative
tl_df3['Position'] = tl_df3['Position'].astype('category')
tl_df3['Position'].cat.reorder_categories(
    ['Member', 'Sergeant at Arms', 'Secretary', 'Treasurer',
     'Vice President Public Relations', 'Vice President Membership',
     'Vice President Education', 'President'], inplace=True)
```

    /anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:23: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
fig, ax = plt.subplots(figsize=(40,10))
sns.lineplot(x='Date', y='Pos', units='Index',
             hue='Position', palette=['k', 'm', 'b', 'c', 'g', 'y', 'orange', 'r'],
             size='Officer', sizes=[0.5, 2],
             estimator=None, data=tl_df3);
ax.set_ylabel('Member Count');
ax.set_title('Varsity Toastmasters Membership and Officers Timeline');
```


<img src="/assets/images/2020-02-10-org-visualisation/output_38_0.png" width="100%"/>


This plot has members sorted so that the oldest member is at the bottom. We can see that presidents tend to be among the oldest members, whereas not-so important roles like secretary tend to be medium-seniority members. The times have been discretized to prevent too many ugly steep gradients. An unfortunate artefact of this is that there can often be several people with the same officer role at the same time.

Another attempt I had at doing this visualisation looked like this:


```python
# Sort members by 'Tenure' so that the longest (and thus most important) members
#   will have nice flat trajectories along the bottom of the timeline.
# Secondarily sort by 'Begin' so that new members get dropped on top of the plot.
members.sort_values(by=['Tenure', 'Begin'], inplace=True, ascending=[False, True]) # , 'Begin'
# officer_ids = officers['Member ID'].unique()
# members = members[members['Member ID'].isin(officer_ids)]
```


```python
# Timeline start and end points
start_dt = dt.datetime(2000, 1, 1)
end_dt = dt.datetime(2020, 1, 1)

# We test membership within quarter-year windows.
#   If the window was smaller then this then the plot would look
#   jagged as members tend to come and leave in chunks.
delta_dt = (dt.datetime(2001, 1, 1) - start_dt) / 4

# Dataframe of all the points which will form the data frame.
# Points with the same 'Index' form a contiguous line.
# We use 'Index' rather than 'Member ID' because if a member leaves and then
#   returns later, this should form a new line (otherwise we have lines crossing
#   over each other which looks ugly).
tl_df = pd.DataFrame(columns=['Index' ,'Pos', 'Date', 'Member ID'])

# Officer timeline dataframe
otl_df = pd.DataFrame(columns=['Index', 'Position', 'Date', 'Member ID'])

# Lower bound on each time window
dt_i = start_dt

while dt_i < end_dt:

    # Upper bound on each time window
    dt_j = dt_i + delta_dt

    # Current Members within time window

    current_members = members[ (~(members['Begin'] > dt_j) & ~(members['End'] < dt_i)) ]

    dldf_i = pd.DataFrame({'Index': list(current_members.index), 'Pos': np.arange(len(current_members))+1,
                          'Member ID': current_members['Member ID']})
    dldf_i['Date'] = dt_i

    tl_df = tl_df.append(dldf_i)

    # Current Officers within time window

    current_officers = officers[ (~(officers['Start'] > dt_j) & ~(officers['End'] < dt_i)) ]

    otldf_i = current_officers[['Position', 'Member ID']]
    otldf_i['Date'] = dt_i
    otldf_i.loc[:, 'Index'] = list(current_officers.index)

    otl_df = otl_df.append(otldf_i)

    # Increment time window

    dt_i = dt_j
```

    /anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:43: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
# Dataframe of membership lines
mem_tldf = tl_df.copy()
mem_tldf['Position'] = 'Member'

# Dataframe of officer lines (to be plotted on top of membership lines)
# We want the officer lines to inherit vertical positions ('Pos') from
#   membership lines (the pairing between the two is given by 'Member ID').
otl_df = otl_df.merge(tl_df[['Member ID', 'Pos', 'Date']], on=['Member ID', 'Date'], how='left')
# otl_df['Index'] = otl_df['Index'].apply(lambda x: x + 1000) # I don't think we need this line

# The officer lines should be more prominent than the membership lines,
#   so I'm setting line thickness according to an 'Officer' column.
otl_df['Officer'] = 'Yes'
mem_tldf['Officer'] = 'No'

# Combine the data for membership lines and officer lines into a single table.
tl_df = mem_tldf.append(otl_df)
tl_df.head()
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
      <th>Date</th>
      <th>Index</th>
      <th>Member ID</th>
      <th>Officer</th>
      <th>Pos</th>
      <th>Position</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>2000-01-01 00:00:00</td>
      <td>3</td>
      <td>362348</td>
      <td>No</td>
      <td>1</td>
      <td>Member</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000-04-01 12:00:00</td>
      <td>3</td>
      <td>362348</td>
      <td>No</td>
      <td>1</td>
      <td>Member</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000-07-02 00:00:00</td>
      <td>3</td>
      <td>362348</td>
      <td>No</td>
      <td>1</td>
      <td>Member</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000-07-02 00:00:00</td>
      <td>4</td>
      <td>376387</td>
      <td>No</td>
      <td>2</td>
      <td>Member</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2000-07-02 00:00:00</td>
      <td>5</td>
      <td>376388</td>
      <td>No</td>
      <td>3</td>
      <td>Member</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(40,10))
sns.lineplot(x='Date', y='Pos', units='Index',
             hue='Position', palette=['k', 'r', 'orange', 'y', 'g', 'c', 'b', 'm'],
             size='Officer', sizes=[0.5, 2],
             estimator=None, data=tl_df);
ax.set_ylabel('Member Count');
ax.set_title('Varsity Toastmasters Membership and Officers Timeline');
```


<img src="/assets/images/2020-02-10-org-visualisation/output_43_0.png" width="100%"/>


This plot has the line start and end x-coordinates rounded to the nearest quarter-year, which gives it this really neat pattern. It looks like a piece of [Mondrian](https://www.wikiart.org/en/piet-mondrian/broadway-boogie-woogie-1943)-style abstract art.

## Another Individual Member Plot

Here's another way I thought of visualising each individual member.


```python
# For any two members which start at the same time, push one of the times down by ten days
#   so that the two can be visually distinguished.

members2 = members.sort_values('Begin').reset_index()
min_sep = np.timedelta64(10, 'D').astype('timedelta64[ns]')

for i in range(len(members2[:-1])):
    min_next = members2.loc[i, 'Begin'] + min_sep
    if members2.loc[i+1, 'Begin'] < min_next:
        delta = min_next - members2.loc[i+1, 'Begin']
        members2.loc[i+1, 'Begin'] += delta
        members2.loc[i+1, 'End'] += delta

member_start_dates = {member_id: members2[members2['Member ID']==member_id]['Begin'].min()
                      for member_id in members2['Member ID'].unique()}

# Dataframe of points to display on the timeline
tl_df = pd.DataFrame(columns=['x', 'y', 'id', 'Position', 'Officer'])

# Add membership lines to tl_df
j = 1
for i in range(len(members)):
    for thing in ['Begin', 'End']:
        member_id = members.loc[i, 'Member ID']
        x = members.loc[i, thing]
        y = x - member_start_dates[member_id]
        tl_df.loc[j, :] = [x, y, i, 'Member', 0]

        j += 1

# Add officer lines to tl_df
for i in range(len(officers)):
    for thing in ['Start', 'End']:
        member_id = officers.loc[i, 'Member ID']
        role = officers.loc[i, 'Position']
        x = officers.loc[i, thing]
        y = x - member_start_dates[member_id]
        tl_df.loc[j, :] = [x, y, i, role, 1]

        j += 1

# The units of the y-axis is years
tl_df['y'] = tl_df['y'].astype('timedelta64[ns]') / np.timedelta64(1, 'Y')

# Make sure the officer roles are plotted in the right order
tl_df['Position'] = tl_df['Position'].astype('category')
tl_df['Position'].cat.reorder_categories(
    ['Member', 'Sergeant at Arms', 'Secretary', 'Treasurer',
     'Vice President Public Relations', 'Vice President Membership',
     'Vice President Education', 'President'], inplace=True)

# Create plot
fig, ax = plt.subplots(figsize=(40,10))
sns.lineplot(x='x', y='y', units='id', estimator=None,
             hue='Position', palette=['k', 'm', 'b', 'c', 'g', 'y', 'orange', 'r'],
             size='Officer', sizes=[0.5, 2],
             data=tl_df)
ax.set_ylabel('Tenure (Years)');
ax.set_xlabel('Joining Date')
ax.set_title('Individual Membership and Officer Role Plot');
```


<img src="/assets/images/2020-02-10-org-visualisation/output_46_0.png" width="100%"/>


Here each member is a vertical line, with colours representing officer roles. This plot has a number of nice properties:
 * For any given time $t$, we can see which members were active at that time because they will intersect the $x=t$ line (and similarly for telling who were the officers at that time). We can thus estimate the number of members at that time.
 * At any given time, we can also see how old each member is at that time by looking at the y-coordinate of the intersection of the membership line and the $x=t$ line.
 * We can tell when a member leaves and then later rejoins the club, because they will have a floating line segment which lines up with their original membership line.
 * By looking at how many lines reach a given height, we can see how many members reached each level of tenure. By looking for the longest continuous line, we can again see that member who lasted from 2007 to 2012 - a 7-8 year tenure.

## Conclusion

Back in the introduction I said that we'd be trying to plot the following aspects of Varsity Toastmaster's club history:
 1. People entering or leaving the organisation over time
 2. The length of people's stay within the organisaiton
 3. Current membership of the organisation
 4. Cumulative membership of the organisation
 5. The succession of roles people take on within the organisation

We ended up with many plots, each of which conveyed a different mix of these aspects. This table summarises which information is conveyed by each plot, and whether it is conveyed in aggregate (i.e., aggregated across all members) or individually (i.e., you can read off information about individual members), as well as whether the information is explicit (you can read the value directly off the plot) or implicit (you can infer the value from other values you can read off the plot).

| Plot | Entering and leaving | Member stay length | Current Membership | Cumulative Membership | Role Succession |
|---|---|---|---|---|---|
| Begin and end membership jointplot | Aggregate | - | - | - | - |
| Member tenure displot | - | Aggregate | - | - | - |
| Cumulative membership plot| Implicit | - | Implicit | Yes | - |
| Transition matrix | - | - | - | - | Aggregate |
| Individual member plots | Individual | Individual | Yes | - | - |
| Individual officer timelines | Individual | Individual | Yes | - | Individual and Aggregate |
| Another individual member plot | Individual and Aggregate | Individual and Aggregate | Yes | - | Individual |

Based on this, I'd say that the last plot is most informative, and if you combine it with the cumulative membership plot then you can convey most of the data available fairly well. But if you're going for visual appeal, the first individual member plot is definitely the way to go.

If I ever return to this project, I've got an idea about how to extend these visualiations. Because we're dealing with temporal data, an animation or interactive slider would make sense (like they have on [Our World in Data](https://ourworldindata.org/grapher/share-of-the-population-living-in-extreme-poverty?year=1990)). For example, imagine a circle representing each role within the club - member, president, treasurer, etc - and dots representing each member. When a new member joins, they enter from the left of the screen, then move between the various circles as they take on different roles (perhaps growing larger the longer they stay), and then exit to the right of the screen. And if each dot leaves a trail you can get a sense of the transition rates between different roles.  
