# STAT1013-final-project
hw2
---
jupyter:
  colab: null
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

<div class="cell markdown" id="ZjKq-LpcJZoE">

# CUHK-STAT1013: Practical Assignment Part1: Sharing your idea and data

</div>

<div class="cell markdown" id="fVc9A6N1KHiX">

## MV playback baclground

**Description**:

Dataset describing the views between kpop and cpop mv.

**Github**:
<https://github.com/hongyanyi0v0/STAT1013.proj/blob/main/mv_playback.csv>

**Sample size**: 200

**Featrue documentation**:

| Feature      | class      | shape | Dtype  |
|:-------------|:-----------|:------|:-------|
| Title        | Tensor     |       | string |
| Group/Artist | Tensor     |       | string |
| Views        | Tensor     |       | string |
| upload date  | Tensor     |       | string |
| type of song | ClassLabel |       | int64  |

</div>

<div class="cell markdown" id="vS0qp-soMqlt">

## Hypothesis

-   Tell us what your idea is and why you have chosen to pursue this
    idea.

-   I am interesting in 'Which type of songs do poeple like the most?'

-   What two groups you are comparing:

-   **G1**: views of kpop ; **G2**: views of cpop

-   What you will be measuring (i.e., what your response variable will
    be)

-   `views`

-   Is your response variable quantitative rather than categorical?

-   `views` is a numerical value which can regrade as quantitive data.

-   Make a prediction about what kind of difference you expect to see
    between your samples and WHY.

-   I expected **G1** \> **G2** since [more poeple perfer listening to
    kpop
    music](https://www.quora.com/Why-is-Kpop-so-much-more-popular-than-Cpop-or-Jpop)

-   Talk about how you will gather your data

-   From GitHub
    <https://github.com/hongyanyi0v0/STAT1013.proj/blob/main/mv_playback.csv>

-   If you had unlimited resources (time, money, staff, etc.) how would
    you collect your data?

-   To collect more data from different platform

</div>

<div class="cell markdown" id="Bosb8NB2NrRz">

## Prepare your dataset

</div>

<div class="cell code" execution_count="63"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:204}"
id="RqZv-CmFN4D-" outputId="ecee7dd8-6cc6-44a0-dd2c-3c749921147b">

``` python
## load dataset from github

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/hongyanyi0v0/STAT1013.proj/main/mv_playback.csv?token=GHSAT0AAAAAAB7DMM2ZY2DNIJYBLWVBMHXKY76FSWA')
df.head(5)
```

<div class="output execute_result" execution_count="63">

       case no.           title  group/artist  views(million) upload date  \
    0         1   Gangnam Style           PSY         4680.00   15/7/2012   
    1         2   DDU-DU DDU-DU     Blackpink         2013.00   15/6/2018   
    2         3  Kill This Love     Blackpink         1735.00    5/4/2019   
    3         4    Boy With Luv           BTS         1634.00   12/4/2019   
    4         5       Gentlemen           PSY           15.39   13/4/2013   

      type of song  
    0         kpop  
    1         kpop  
    2         kpop  
    3         kpop  
    4         kpop  

</div>

</div>

<div class="cell markdown" id="89muzb08OExw">

-   Tell us what groups you want to compare in the dataset

</div>

<div class="cell markdown" id="wKiVxFgTOPoj">

-   Print first 5 records of each group, respectively

</div>

<div class="cell code" execution_count="64"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="BpiHvRZ0OD-L" outputId="7f1fab94-7326-4945-9b3b-bf212fa706f3">

``` python
## First 5 records of G1 (kpop)
(df[df['type of song'] == 'kpop']['views(million)']).head(5)
```

<div class="output execute_result" execution_count="64">

    0    4680.00
    1    2013.00
    2    1735.00
    3    1634.00
    4      15.39
    Name: views(million), dtype: float64

</div>

</div>

<div class="cell code" execution_count="65"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="In3-MQllOj1S" outputId="cc0d9dc4-86af-4f93-f3be-8af77a93a53e">

``` python
## First 5 records of G2 (F)
(df[df['type of song'] == 'cpop']['views(million)']).head(5)
```

<div class="output execute_result" execution_count="65">

    81    114.8
    82     95.9
    83     34.0
    84     30.7
    85     28.7
    Name: views(million), dtype: float64

</div>

</div>

<div class="cell code" execution_count="66"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="US8BCQSzOras" outputId="ce52309e-ca61-4f2e-cd5f-715d14c6f91a">

``` python
## Other data description and visualization you want to add
df.info()
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 120 entries, 0 to 119
    Data columns (total 6 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   case no.        120 non-null    int64  
     1   title           120 non-null    object 
     2    group/artist   120 non-null    object 
     3   views(million)  120 non-null    float64
     4   upload date     120 non-null    object 
     5   type of song    120 non-null    object 
    dtypes: float64(1), int64(1), object(4)
    memory usage: 5.8+ KB

</div>

</div>

<div class="cell code" execution_count="67"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Nye2vlJjxMUQ" outputId="bad539d7-8e1d-4e3f-f135-a92b33421df5">

``` python
## How many views larger than 123
len(df[df['views(million)'] > 123])
```

<div class="output execute_result" execution_count="67">

    94

</div>

</div>

<div class="cell code" execution_count="68"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Sr5QlCemx1wr" outputId="6862c056-a982-4a19-d2f4-64edd36d79cc">

``` python
(df.groupby('type of song')['views(million)'].mean())
```

<div class="output execute_result" execution_count="68">

    type of song
    cpop       94.938889
    cpop\t     93.333333
    kpop      613.310741
    Name: views(million), dtype: float64

</div>

</div>

<div class="cell code" execution_count="69"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="xSJSr1Iz0URY" outputId="e0e37be8-1128-4eaf-90e2-9c0a61d6c695">

``` python
(df.groupby('type of song')['views(million)'].std())
```

<div class="output execute_result" execution_count="69">

    type of song
    cpop       78.854341
    cpop\t      2.081666
    kpop      601.662332
    Name: views(million), dtype: float64

</div>

</div>

<div class="cell code" execution_count="70"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:661}"
id="XVg3EP-r0dOA" outputId="ad426f37-1447-4bc2-c36a-380e7fb050a8">

``` python
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [10, 5]

sns.set()

sns.histplot(data=df, x='views(million)' ,bins='auto')
plt.show()

sns.violinplot(data=df, x='views(million)')
plt.show()
```

<div class="output display_data">

![](bb367f1a942209c2c2f97844101518dc34f90393.png)

</div>

<div class="output display_data">

![](10f318072cace79203b16d1b90da2e1f3fdaac33.png)

</div>

</div>

<div class="cell code" execution_count="71"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:235}"
id="aFC6E2L82EfU" outputId="61b591e7-0a12-46d1-87d6-1f22bffb92a9">

``` python
df.head(10).T
```

<div class="output execute_result" execution_count="71">

                                0              1               2             3  \
    case no.                    1              2               3             4   
    title           Gangnam Style  DDU-DU DDU-DU  Kill This Love  Boy With Luv   
     group/artist             PSY      Blackpink       Blackpink           BTS   
    views(million)         4680.0         2013.0          1735.0        1634.0   
    upload date         15/7/2012      15/6/2018        5/4/2019     12/4/2019   
    type of song             kpop           kpop            kpop          kpop   

                            4          5          6           7  \
    case no.                5          6          7           8   
    title           Gentlemen  BoomBaYah        DNA    Mic Drop   
     group/artist         PSY  Blackpink        BTS         BTS   
    views(million)      15.39     1532.0     1511.0      1301.0   
    upload date     13/4/2013   8/8/2016  18/9/2017  24/11/2017   
    type of song         kpop       kpop       kpop        kpop   

                                       8          9  
    case no.                           9         10  
    title           As if it's ypur last       IDOl  
     group/artist              Blackpink        BTS  
    views(million)                1246.0     1198.0  
    upload date                22/6/2017  24/8/2018  
    type of song                    kpop       kpop  

</div>

</div>

<div class="cell code" id="Dp08cWlJ2ixW">

``` python
```

</div>
