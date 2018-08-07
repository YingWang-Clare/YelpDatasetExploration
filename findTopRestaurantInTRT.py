# -*- coding: utf-8 -*-
import collections
import csv
import simplejson as json
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from wordcloud import WordCloud


def process_json(source_path, des_path):
    """Create a new file contains only business in Toronto."""
    with open(des_path, 'w') as dest_file:
        with open(source_path, 'r') as source_file:
            for line in source_file:
                line_content = json.loads(line.strip())
                if line_content['city'] == 'Toronto':
                    dest_file.write(json.dumps(line_content) + '\n')


def get_superset_of_column_names_from_file(json_file_path):
    """Read in the json dataset file and return the superset of column names."""
    column_names = set()
    with open(json_file_path) as fin:
        for line in fin:
            line_contents = json.loads(line)
            column_names.update(
                set(get_column_names(line_contents).keys())
            )
    return column_names


def get_column_names(line_contents, parent_key=''):
    """Return a list of flattened key names given a dict.
    Example:
        line_contents = {
            'a': {
                'b': 2,
                'c': 3,
                },
        }
        will return: ['a.b', 'a.c']
    These will be the column names for the eventual csv file.
    """
    column_names = []
    for k, v in line_contents.items():
        column_name = "{0}.{1}".format(parent_key, k) if parent_key else k
        if isinstance(v, collections.MutableMapping):
            column_names.extend(
                get_column_names(v, column_name).items()
            )
        else:
            column_names.append((column_name, v))
    return dict(column_names)


def read_and_write_file(json_file_path, csv_file_path, column_names):
    """Read in the json dataset file and write it out to a csv file, given the column names."""
    with open(csv_file_path, 'w') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(list(column_names))
        with open(json_file_path, 'r') as fin:
            for line in fin:
                line_contents = json.loads(line)
                csv_file.writerow(get_row(line_contents, column_names))


def get_row(line_contents, column_names):
    """Return a csv compatible row given column names and a dict."""
    row = []
    for column_name in column_names:
        line_value = get_nested_value(
            line_contents,
            column_name,
        )
        if isinstance(line_value, str):
            row.append('{0}'.format(line_value))
        elif line_value is not None:
            row.append('{0}'.format(line_value))
        else:
            row.append('')
    return row


def get_nested_value(d, key):
    """Return a dictionary item given a dictionary `d` and a flattened key from `get_column_names`.

    Example:
        d = {
            'a': {
                'b': 2,
                'c': 3,
                },
        }
        key = 'a.b'
        will return: 2

    """
    if '.' not in key:
        if key not in d:
            return None
        if d[key] == '' or d[key] is None:
            return None
        return d[key]
    base_key, sub_key = key.split('.', 1)
    if base_key not in d:
        return None
    sub_dict = d[base_key]
    if sub_dict is None:
        return None
    return get_nested_value(sub_dict, sub_key)


CLEANED_ITEMS = ('business_id', 'review_count', 'categories', 'stars', 'address', 'neighborhood', 'name')
RESTAURANT_KEYWORDS = 'Restaurants Food Bars Coffee Tea Sandwiches Breakfast Brunch ' \
                      'Cafes Bakeries Pizza Desserts Burgers Pubs Sushi Ice Cream ' \
                      'Yogurt Seafood Smoothies Chicken Wings Salad Barbeque Vegetarian ' \
                      'Steakhouses Noodles Bagels Donuts Buffets'


def clean_dataframe(df):
    """Preserve specific columns and specific categories of business"""
    for column in df:
        if column not in CLEANED_ITEMS:
            df.drop(column, axis=1, inplace=True)
    for index, df_row in df.iterrows():
        if isinstance(df_row['categories'], float):
            df.drop(index, axis=0, inplace=True)
            continue
        if any(word in RESTAURANT_KEYWORDS for word in df_row['categories'].split(', ')):
            continue
        else:
            df.drop(index, axis=0, inplace=True)
    df.drop_duplicates(subset='name', keep='first', inplace=True)
    df.reset_index()


def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)


def visualization(freq_data, hist_title):
    """Visualize the data in WordCloud and in Histogram."""
    fig = plt.figure(1, figsize=(18, 13))
    ax1 = fig.add_subplot(2, 1, 1)

    words = dict()
    trunc_occurences_word_cloud = freq_data[0:2000]
    trunc_occurences_histogram = freq_data[0:50]
    for s in trunc_occurences_word_cloud:
        words[s[0]] = s[1]
    tone = 55.0
    wordcloud = WordCloud(width=1000, height=300, background_color='white',
                          max_words=1200, relative_scaling=0.4,
                          normalize_plurals=False)
    wordcloud.generate_from_frequencies(words)
    ax1.imshow(wordcloud, interpolation="bilinear")
    ax1.axis('off')
    ax2 = fig.add_subplot(2, 1, 2)
    y_axis = [i[1] for i in trunc_occurences_histogram]
    x_axis = [k for k, i in enumerate(trunc_occurences_histogram)]
    x_label = [i[0] for i in trunc_occurences_histogram]
    plt.xticks(rotation=85, fontsize=8)
    plt.yticks(fontsize=12)
    plt.xticks(x_axis, x_label)
    plt.ylabel(hist_title, fontsize=15, labelpad=10)
    ax2.bar(x_axis, y_axis, align='center', color='g')
    plt.title(hist_title, bbox={'facecolor': 'k', 'pad': 5}, color='w', fontsize=20)
    plt.show()


def count_word(df_file, ref_col, list):
    """Count the word frequency regarding one column of the dataframe."""
    keyword_count = dict()
    for word in list:
        keyword_count[word] = 0
    for list_keywords in df_file[ref_col]:
        if isinstance(list_keywords, float):
            continue
        for key_word in str(list_keywords).split(', '):
            if isinstance(key_word, float):
                continue
            keyword_count[key_word] += 1
    keyword_occurences = []
    for k, v in keyword_count.items():
        keyword_occurences.append([k, v])
    keyword_occurences.sort(key=lambda x: x[1], reverse=True)
    return keyword_occurences, keyword_count


def category_key_words(df_file):
    """Collect all keywords in the categories without repetitions."""
    categories = set()
    for category in df_file['categories']:
        if isinstance(category, float):
            continue
        for ctg in str(category).split(', '):
            if isinstance(ctg, float):
                continue
            categories.add(ctg)
    return categories


def count_single_factor(df_file, ref_col):
    """
    Return a dictionary with key-value pair indicating the business's name and the
    value of the reference column.
    """
    dictionary = dict()
    for index, df_row in df_file.iterrows():
        dictionary[df_row['name']] = df_row[ref_col]
    sorted_dictionary = []
    for n, v in dictionary.items():
        sorted_dictionary.append([n, v])
        sorted_dictionary.sort(key=lambda x: x[1], reverse=True)
    return sorted_dictionary, dictionary


def popularity_measurement(df_file):
    """
    Define a metric for Restaurant Popularity Measurement.
    Specifically, the degree of popularity is based on product of the normalized number of reviews of the restaurant
    and the normalized level of star of the restaurant.
    The final popularity is sorted by the normalized popularity score.
    """
    df_file['review_count_norm'] = (df_file['review_count'] - df_file['review_count'].mean()) \
                                   / df_file['review_count'].std()
    df_file['stars_norm'] = (df_file['stars'] - df_file['stars'].mean()) / df_file['stars'].std()
    df_file['popularity'] = df_file['review_count_norm'] * df_file['stars_norm']
    df_file['popularity_norm'] = (df_file['popularity'] - df_file['popularity'].min()) \
                                 / (df_file['popularity'].max() - df_file['popularity'].min())
    popularity = dict()
    for index, df_row in df_file.iterrows():
        popularity[df_row['name']] = df_row['popularity_norm']
    sorted_popularity = []
    for n, v in popularity.items():
        sorted_popularity.append([n, v])
    unsorted_popularity = sorted_popularity
    sorted_popularity.sort(key=lambda x: x[1], reverse=True)
    return sorted_popularity, unsorted_popularity


if __name__ == '__main__':
    """
    INTRODUCTION:
    This task is divided into three parts.
    Section 1 preprocesses the original data, such as data format convert, for the convenience of further usage.
    Section 2 actually has little relation with the main task, it is only the by-product of the task, such as the 
    exploration of some features and visualization of the dataset.
    Section 3 includes the key steps of the tasks.
    """

    """
    ---------------------------------------------------------------------------------------------
            Section 1: DATA PREPROCESSING
    """
    """The path to the original json file and the location where I put the processed data."""
    source = '/Users/wangying/Desktop/yelp_dataset/yelp_academic_dataset_business.json'
    dest = 'data/yelp_academic_dataset_trt_business.json'
    process_json(source_path=source, des_path=dest)

    """Convert the json file to csv file and then read the csv as pandas dataframe, then clean the data."""
    csv_file = '{0}.csv'.format(dest.split('.json')[0])
    cleaned_csv_file = '{0}_cleaned.csv'.format(dest.split('.json')[0])
    column_names = get_superset_of_column_names_from_file(dest)
    read_and_write_file(dest, csv_file, column_names)
    df_file = pd.read_csv(csv_file)
    clean_dataframe(df_file)

    """Store the cleaned data to another csv file for future usage."""
    df_file.to_csv(open(cleaned_csv_file, 'w'), sep='\t', encoding='utf-8', index=False)

    """
    ---------------------------------------------------------------------------------------------
            Section 2: DATASET EXPLORATION & VISUALIZATION
    """
    df_file = pd.read_csv(cleaned_csv_file, sep='\t')

    """Set some parameters for the visualiztion."""
    plt.rcParams["patch.force_edgecolor"] = True
    plt.style.use('fivethirtyeight')
    mpl.rc('patch', edgecolor='dimgray', linewidth=1)
    pd.options.display.max_columns = 50
    warnings.filterwarnings('ignore')

    """Since for each business, descriptions in categories reflects its field. I collect all keywords
    of all foodservice industry in Toronto."""
    categories = category_key_words(df_file)
    keyword_occurences, key_orig = count_word(df_file, 'categories', categories)
    # Please uncomment this line to see the visualization result.
    # visualization(keyword_occurences, "keyword occurences")

    """Use the same logic as the keywords, I sort all restaurants according to the number of reviews
    they received and the level of stars."""
    review, rvw_orig = count_single_factor(df_file, 'review_count')
    # Please uncomment this line to see the visualization result.
    # visualization(review, "review")
    stars, str_orig = count_single_factor(df_file, 'stars')
    # Please uncomment this line to see the visualization result.
    # visualization(stars, "stars")

    """
    ---------------------------------------------------------------------------------------------
            Section 3: Popularity Measurement
    """
    """This section does the main work of finding the top 10 most popular restaurants."""
    popularity, ppl_orig = popularity_measurement(df_file)
    print('The Top 10 most popular restaurants in Toronto are: {}'.format(popularity[0:10]))
