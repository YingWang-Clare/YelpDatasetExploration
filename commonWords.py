import simplejson as json
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def find_target_business_ID(business, target_name):
    """Find the business ID of the given business's name."""
    with open(business, 'r') as file:
        for line in file:
            line_content = json.loads(line.strip())
            if line_content['name'] == target_name:
                target_id = line_content['business_id']
                break
    return target_id

def putAllReviewAsOne(review, dest, txt, target_BID):
    """Find all reviews of the given business and put all reviews into a txt file."""
    cnt = 0
    with open(txt, 'w') as txtfile:
        with open(dest, 'w') as jsonfile:
            with open(review, 'r') as fin:
                for line in fin:
                    line_content = json.loads(line.strip())
                    if line_content['business_id'] == target_BID:
                        cnt += 1
                        jsonfile.write(json.dumps(line_content) + '\n')
                        txtfile.write(line_content['text'] + ' ')
    txtfile.close()
    print("There are totally {} valid reviews.".format(cnt))

def findMostCommonWords(review_txt_path):
    """Sort words (except stopwords) in the given txt file by their frequecy."""
    with open(review_txt_path, 'r') as file:
        review_data = file.read().replace('\n', '')
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    allWords = tokenizer.tokenize(review_data)
    stop_words = stopwords.words('english')
    filtered_sentence = [w.lower() for w in allWords if not w.lower() in stop_words]
    lemmatized_sentence = [lemmatizer.lemmatize(w) for w in filtered_sentence]
    allWordExceptStopDist = nltk.FreqDist(lemmatized_sentence)
    mostCommonWords = allWordExceptStopDist.most_common()
    return mostCommonWords

if __name__ == '__main__':
    business_path = '/Users/wangying/Desktop/yelp_dataset/yelp_academic_dataset_business.json'
    review_path = '/Users/wangying/Desktop/yelp_dataset/yelp_academic_dataset_review.json'
    dest_path = 'data/yelp_academic_dataset_review_ChipotleMexicanGrill.json'
    review_txt_path = 'data/review_ChipotleMexicanGrill.txt'

    targetBssID = find_target_business_ID(business_path, "Chipotle Mexican Grill")
    putAllReviewAsOne(review_path, dest_path, review_txt_path, targetBssID)

    mostCommonWords = findMostCommonWords(review_txt_path)

    print('The most 10 words (with their frequencies) appearing in all reviews of {} are as follows: \n{}'
          .format("Chipotle Mexican Grill", mostCommonWords[:10]))
