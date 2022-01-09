import matplotlib.pyplot as plt
import wandb
import os
import json

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

if __name__ == '__main__':
    # wandb.init(project="neural-matrix-factorization", entity="popescuaaa")

    # Tripadvisor
    with open(data_folders["news"]["topics"], "r") as f:
        pass
