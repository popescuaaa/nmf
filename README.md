# Non-negative Matrix Factorization

---

Todo:

## Setup

---
- [x] Add a nmf class adapted form sklearn
- [x] Add datasets 
  - [x] Tripadvisor reviews (kaggle)
  - [x] All news dataset (kaggle)

## Experimental

---
- [x] Integrate wandb for logging
- [x] Adapt for new ds
- [ ] Log relations in results and create report !!!
  - [ ] Tripadvisor
    - [x] Try to associate subtopics - with positive or negative feelings
    - [?] The coherence of a model (or how to find the optimal number of topics using a coherence score)
    

- [ ] Define the task of subdomain topic modeling (the whole project)
- [ ] Print topic probabilities
- [ ] Report results

## Docs

---
- [ ] Start word document and write introduction 10 pages
  - Content: 

      • Importance and practical applications of the algorithm 2

      • The algorithm general presentation 2

      • Known results and issues 2

      • Datasets used (names and samples) 2

      • Results (including samples) and evaluation 2

      • References:

        - https://arxiv.org/pdf/2105.13440.pdf

        - https://methods.sagepub.com/base/download/DatasetStudentGuide/non-negative-matrix-factorization-in-news-2016

        - https://sci-hub.hkvisa.net/10.3233/jifs-191690

        - Evaluation: https://highdemandskills.com/topic-model-evaluation/

        - http://derekgreene.com/papers/greene14topics.pdf

        - https://github.com/derekgreene/topic-model-tutorial/blob/master/topic-modelling-with-scikitlearn.pdf

Evalaution:

Splitting the data into training and testing sets is a common step in evaluating the performance of a learning algorithm. It's more clear-cut for supervised learning, wherein you train the model on the training set, then see how well its classifications on the test set match the true class labels. For unsupervised learning, such evaluation is a little trickier. In the case of topic modeling, a common measure of performance is perplexity. You train the model (like LDA) on the training set, and then you see how "perplexed" the model is on the testing set. More specifically, you measure how well the word counts of the test documents are represented by the word distributions represented by the topics.

Perplexity is good for relative comparisons between models or parameter settings, but it's numeric value doesn't really mean much. I prefer to evaluate topic models using the following, somewhat manual, evaluation process:

- Inspect the topics: Look at the highest-likelihood words in each topic. Do they sound like they form a cohesive "topic" or just some random group of words?
Inspect the topic assignments: 
- Hold out a few random documents from training and see what topics LDA assigns to them. 
- Manually inspect the documents and the top words in the assigned topics. 

Does it look like the topics really describe what the documents are actually talking about?

I realize that this process isn't as nice and quantitative as one might like, but to be honest, the applications of topic models are rarely quantitative either. I suggest evaluating your topic model according to the problem you're applying it to.

Stack user: https://stackoverflow.com/users/1481245/gregamis
