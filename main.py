from typing import List
import wandb
import json
from src import NMF

# Sentiment analysis module 
from afinn import Afinn
afinn = Afinn(language="en")


data_folders = {
    "news": { 
        "full": "all_news_results/doc_to_topics.json",
        "topics": "all_news_results/topic_to_words.json"
    },
    "trip": {
        "full": "tripadvisor_hotel_reviews_results/doc_to_topics.json",
        "topics": "tripadvisor_hotel_reviews_results/topic_to_words.json"
    }
}

def most_relevant_data(topic_idx: int, full: dict) -> List[str]:
    ful_items = full.items()
    sorted_items = sorted(ful_items, key=lambda t: t[topic_idx], reverse=True)
    return sorted_items[:10]

def all_news_eda():
    sub_topics = {}
    topics = {}

    with open(data_folders["news"]["topics"], "r") as f:
        topics = json.load(f)
        for topic in topics:
            print(topic, topics[topic])
            sub_topics[topic] = 0
            

    with open(data_folders["news"]["full"], "r") as f:
        full = json.load(f)
        for article in full:
            associated_topics = full[article]
            items = associated_topics.items()

            sorted_items = sorted(items, key=lambda e: e[1], reverse=True)
            if sorted_items[0][1] == 0:
                continue
            else: 
                # The only the ones that are conclusive for our search
                sub_topics[sorted_items[0][0]] += 1

    # keys = sub_topics.keys()
    # values = [sub_topics[k] for k in keys]
    # data = [[key, val] for [key, val] in zip(keys, values)]
    # table = wandb.Table(data=data, columns = ["key", "val"])
    # wandb.log({"my_bar_chart_id" : wandb.plot.bar(table, "key", "val", title="Custom Bar Chart")})

    # Most relevant articles for  current topic modeling session
    print(most_relevant_data(topic_idx=0, full=full))




def tripadvisor_reviews_eda():
    # Associate topics with positive, neutral or negative feelings indetified in reviews
    topics = None
    feelings = {
        "positive": {},
        "negative": {},
        "neutral": {}
    }
   
        

    with open(data_folders["trip"]["topics"], "r") as f:
        topics = json.load(f)
        for topic in topics:
            print(topic, topics[topic])
    
     # Populate feeling with subtopics
    for feeling in feelings:
        for topic in topics:
            feelings[feeling][topic] = 0


    with open(data_folders["trip"]["full"], "r") as f:
        full = json.load(f)
        for review in full:
            associated_topics = full[review]
            items = associated_topics.items()
            sorted_items = sorted(items, key=lambda e: e[1], reverse=True)
            feeling_score = afinn.score(review)
            feeling = None

            if feeling_score > 0:
                feeling = "positive"
            elif feeling_score == 0:
                feeling = "neutral"
            else:
                feeling = "negative"

            if sorted_items[0][1] == 0:
                continue
            else:
                # The only the ones that are conclusive for our search
                feelings[feeling][sorted_items[0][0]] += 1
    
    for feeling in feelings:
        keys = feelings[feeling].keys()
        values = [feelings[feeling][k] for k in keys]
        data = [[key, val] for [key, val] in zip(keys, values)]
        table = wandb.Table(data=data, columns = ["key", "val"])
        wandb.log({"{}".format(feeling) : wandb.plot.bar(table, "key", "val", title="{}".format(feeling))})


if __name__ == '__main__':
    
    # Init wadb logging system
    # wandb.init(project="neural-matrix-factorization", entity="popescuaaa")
    
    # All news 
    # nmf = NMF(csv_path="./data/all_news.csv", outdir="./all_new_results")
    all_news_eda()

    """
        We can observe the highly political bias for all the news. 
    """

    # Tripadvisor
    # tripadvisor_reviews_eda()
    
    

    
    
    
