# Recom.live — the real-time recommendation system
This is the core of the recommendation system, wrapped with UDP server daemon, having very simple RPC protocol.

To set up Recom.live server directly onto your physical machine use:
```
git clone https://github.com/grinya007/recomlive.git
cd recomlive
./server.py start
```
Although, I would encourage you to use [Recom.live docker image](https://github.com/grinya007/recomlive-docker), there you'll find a more detailed installation guide and usage examples. 

After the server is set up, you can use [Recom.live client](https://github.com/grinya007/recomlive-client) library to interact with the server.



## What it's all about?
Recom.live is the real-time shallow-learning unsupervised item-based collaborative filtering recommendation system. It takes advantage of [ARC](https://en.wikipedia.org/wiki/Adaptive_replacement_cache) algorithm to keep up the actual state of visitors interest, [TFIDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)-alike statistic to align visitors and documents importance and [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) measure to come up with recommendations.

## What problem does it solve?
Let's assume you have a news website, where recommendations block below articles is driven by the smart batch-model recommendation system. Whenever another training iteration is finished, fresh recommendations bring you a superior CTR. But how long does it take to collect another batch, sufficient for training? What recommendations would you show underneath a breaking news article, which is facing views spike if it happens to be published in 10 minutes before the next training iteration is finished? Such a situation, where good recommendations for breaking news arrive too late, is quite common. Just imagine how many page views and engaged readers your website loses. To smooth it out usual practice is to populate recommendations block with links to the most popular articles while proper recommendations are on their way. This solution is better than nothing. But if the described situation is to some extent relevant to what you have—you must give a try to Recom.live! Its intention is to fill that gap before a smarter but unhurried recommendation system kicks in. Moreover, if you use a third-party recommendation system by embedding its widgets into pages of your website, I can bet that Recom.live will perform better even having default settings. The absolute majority of recommendation systems on the market use their "recommendation service" as a stalking-horse just to get access to the traffic of your website. They don't really care about the quality of recommendations while putting much effort into the monetization of the traffic. It is likely that you have different priorities and the bounce rate worries you more than a few bucks shared by a third-party recommendation system. Whereas, Recom.live enables you to get full control over the recommendation system and without the need to hire a "data scientists".

## How does it work?
Every document_id and person_id are kept in two separate instances of ARC cache class. In addition to the usual functionality of the cache the particular implementation used in Recom.live assignes a unique ID to each cached item. These IDs are taken from the range of 0 to N - 1 where N is the maximum number of cached items so that IDs are being reused with time. In the Recommender class, documents and persons are kept as a matrix where these cache IDs are used as respective indexes. So, when a new document or person comes in and the cache algorithm replaces some outdated item the ID of the latter is reused and the corresponding row or column in the matrix is reinitialized with zeros. The value for the pair of document/person is calculated as follows: 
