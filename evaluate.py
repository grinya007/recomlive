#!/usr/bin/env python3

from recommender import Recommender
import sys

visits = 0
tries = 0
guesses = 0

persons_10_20 = set()
with open('/home/ag/hlam/10_15_persons') as f:
    for line in f:
        persons_10_20.add(line.rstrip())

r = Recommender(2500, 2500, 1)

# Pour document_id,person_id into this script
for row in sys.stdin:
    row = row.rstrip().split(',')

    # if not row[1] in persons_10_20:
        # continue

    visits += 1
    # let's see if a person has already been here
    prs_hist = r.person_history(row[1])
    if len(prs_hist) > 0:

        # if so, we can try to guess current document_id
        # by looking into recommendations for the previous document_id
        prev_did = prs_hist[-1]
        rec = r.recommend(prev_did, row[1])
        tries += 1

        # If we have something to recommend at all
        # and current document_id isn't the same as previous
        if len(rec) > 0 and prev_did != row[0]:

            if row[0] in rec:

                # Hooray!
                guesses += 1

    # finally, record current visit to keep recommender up to date
    r.record(row[0], row[1])
    # if visits % 1000 == 0:
        # print('Total visits:    {}'.format(visits))
        # print('Tries to guess:  {}'.format(tries))
        # print('Guesses:         {}'.format(guesses))
        # print('CTR:             {:.2f}%'.format(guesses*100/tries))


print('Total visits:    {}'.format(visits))
print('Tries to guess:  {}'.format(tries))
print('Guesses:         {}'.format(guesses))
print('CTR:             {:.2f}%'.format(guesses*100/tries))

