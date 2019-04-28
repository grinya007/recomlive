# Recom.live — the real-time recommendation system
This is the core of the recommendation system, wrapped with UDP server daemon, having very simple RPC protocol.

## To set up Recom.live server directly onto your physical machine use:
```
git clone https://github.com/grinya007/recomlive.git
cd recomlive
./server.py start
```
Although, I would encourage you to use [Recom.live docker image](https://github.com/grinya007/recomlive-docker). 

After the server is set up, you can use [Recom.live client](https://github.com/grinya007/recomlive-client) library to interact with the server.



## What it's all about?
Recom.live is the real-time shallow-learning unsupervised item-based collaborative filtering recommendation system. It takes advantage of ARC algorithm to keep up the actual state of visitors interest, TFIDF-alike statistic to align visitors and documents importance and Cosine similarity measure to come up with recommendations.

## What problem does it solve?
Let's assume you have a news website, where recommendations block below articles is driven by the smart batch-model recommendation system. Whenever another training iteration is finished, fresh recommendations bring you a superior CTR. But how long does it take to collect another batch, sufficient for training? What recommendations would you show underneath a breaking news article, which is facing views spike if it happens to be published in 10 minutes before the next training iteration is finished? Such a situation, where good recommendations for breaking news arrive too late, is quite common. Just imagine how many page views and engaged readers your website loses. To smooth it out usual practice is to populate recommendations block with links to the most popular articles while proper recommendations are on their way. This solution is better than nothing. But if the described situation is to some extent relevant to what you have—you must give a try to Recom.live! Its intention is to fill that gap before a smarter but unhurried recommendation system kicks in.
