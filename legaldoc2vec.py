from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument, TaggedDocument, LabeledSentence
from gensim.models.phrases import Phraser, Phrases
import sentence_maker
from bs4 import BeautifulSoup as soup
import pickle
import re
import numpy as np
from random import shuffle, sample
import asyncio
import aiohttp
import async_timeout
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb


def make_links():
    """make some links to Court of Appeal cases on bailii.org"""
    years = [y for y in range(1990, 2018)]
    nums = list(range(60))
    combos = [(y, n) for y in years for n in nums]
    base_addr = "http://www.bailii.org/ew/cases/EWCA/Civ/"
    return [''.join([base_addr, str(y), "/", str(n), ".html"]) for (y, n) in combos][:3]


async def get_text(session, url):
    """download case from url, return BeautifulSoup text"""
    with async_timeout.timeout(1200):
        try:
            async with session.get(url) as response:
                t = await response.text()
                return (url, soup(t, "lxml").text)
        except Exception as e:
            print(e)


async def coordinate_downloads(urls):
    """assemble the download tasks"""
    tasks = []
    async with aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0'}) as session:
        print("retrieving {} cases".format(len(urls)))
        for url in urls:
            tasks.append(asyncio.ensure_future(get_text(session, url)))
        return await asyncio.gather(*tasks)


def build_model():
    """build doc2vec model from cases"""
    
    """get urls for cases"""
    urls = make_links()
    shuffle(urls)

    """async downloads"""
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(coordinate_downloads(urls))
    cases = [c for c in loop.run_until_complete(future) if len(c[1]) > 25]
    print("retrieved {} usable cases".format(len(cases)))
    
    lls = []
    for label, case in cases:
        lls.append(LabeledSentence(words=case.split(), tags=label))
    
    model = Doc2Vec(size=300, window=10, min_count=5, workers=6, alpha=0.025, min_alpha=0.025)
    model.build_vocab(lls)
    
    for epoch in range(10):
        model.train(lls)

    print("trained")
    for dv in model.docvecs:
        print(dv)
    
    input()
    print(model.most_similar("court"))
    
    """make sentences"""
    print("preprocessing text...")
    sentences = []
    for c in cases:
        s = sentence_maker.split_into_sentences(c[1], lower=True)
        sentences.extend(sentence_maker.split_into_sentences(c[1], lower=True))
    
    print("found {} sentences".format(len(sentences)))
    
    """phrase pre-processing"""
    print("building phrases...")
    phrases = Phrases(sentences, min_count=5, threshold=100)
    bigramphraser = Phraser(phrases)
    """produce a representation of the text including 2 and 3 word phrases"""
    trg_phrases = Phrases(bigramphraser[sentences], min_count=5, threshold=100)
    trigram_phraser = Phraser(trg_phrases)
    phrased_sentences = list(trigram_phraser[list(bigramphraser[sentences])])
    print("building Word2Vec model...")
    return Word2Vec(phrased_sentences, min_count=10, workers=6)


def main():
    """download cases, build vocabulary model"""
    
    w2v = build_model()
    
    


if __name__ == "__main__":
    main()
