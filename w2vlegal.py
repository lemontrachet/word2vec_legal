from gensim.models import Word2Vec
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
from word_vectoriser import Word_Vectoriser
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import requests
import get_links
from concurrent.futures import ProcessPoolExecutor


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
        print("retrieving {} cases...\n".format(len(urls)))
        for url in urls:
            tasks.append(asyncio.ensure_future(get_text(session, url)))
        return await asyncio.gather(*tasks)


def download():
    """get urls for cases"""
    urls = get_links.enqueue_links()
    print("retrieved {} urls".format(len(urls)))
    shuffle(urls)

    """async downloads"""
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(coordinate_downloads(urls))
    cases = [c for c in loop.run_until_complete(future) if (c and len(c[1]) > 25)]
    return cases


def find_relevant(case):
    pattern = re.compile(".*\[\d\d\d\d\].*")
    return re.findall(pattern, str(case))


def text_to_list(text=None):
    """tokenise text and find phrases"""
    if not text:
        cases = download()
    else:
        cases = [(1, text)]
    cases = [c for c in cases if len(c[1]) > 500]
    """make sentences"""
    print("preprocessing text...\n")
    sentences = []
    
    with ProcessPoolExecutor(max_workers=8) as ex:
        for s in list(ex.map(sentence_maker.split_into_sentences, [c[1] for c in cases])):
            sentences.extend(s)

    print("\nfound {} sentences\n".format(len(sentences)))

    """phrase pre-processing"""
    print("building phrases...\n")
    phrases = Phrases(sentences, min_count=5, threshold=100)
    bigramphraser = Phraser(phrases)
    """produce a representation of the text including 2 and 3 word phrases"""
    trg_phrases = Phrases(bigramphraser[sentences], min_count=5, threshold=100)
    trigramphraser = Phraser(trg_phrases)
    phrased_sentences = list(trigramphraser[list(bigramphraser[sentences])])
    return phrased_sentences, cases, bigramphraser, trigramphraser


def encode_extracts(cases, vectoriser, bigramphraser, trigramphraser):
    """encode relevant extracts from cases"""
    extracts = [(c[0], find_relevant(c[1])) for c in cases]
    
    with ProcessPoolExecutor(max_workers=8) as ex:
        split_extracts = ex.map(sentence_maker.split_into_sentences, [str(e[1]) for e in extracts])
    """produce a representation of the text including 2 and 3 word phrases"""
    phrased_extracts = list(trigramphraser[list(bigramphraser[split_extracts])])
    vecs = np.array([x for x in
                    [vectoriser.transform(s) for s in phrased_extracts]
                    if x.size > 0])
    
    print("transformed {} cases into {} vectors of shape {}\n"
            .format(len(cases),
                vecs.shape[0],
                vecs.shape))
    return vecs


def build_model(phrased_sentences):
    """build word2vec model from cases"""
    
    print("building Word2Vec model...")
    return Word2Vec(phrased_sentences, min_count=10, workers=8)


def pca(casevecs):
    transformer = PCA(n_components=15)
    compressed = transformer.fit_transform(casevecs)
    return transformer, compressed


def clustering(X):
    print(len(X))
    print("clustering cases: 5 clusters \n")
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    print(kmeans.labels_)


def compare_cases(stored, new):
    print(stored.shape)
    print(new.shape)
    return np.argmax([np.mean(cosine_similarity(s.reshape(-1, 1), new.reshape(-1, 1))) for s in stored])



def main():
    """download cases, build vocabulary model"""
    
    """get case data and preprocess"""
    try:
        with open("stored/ph_sentences.pkl", "rb") as f:
            phrased_sentences, cases, bp, tp = pickle.load(f)
        print("loaded {} sentences".format(len(phrased_sentences)))
    except Exception as e:
        print(e)
        phrased_sentences, cases, bp, tp = text_to_list()
        print("found {} sentences in {} cases\n".format(len(phrased_sentences), len(cases)))
        with open("stored/ph_sentences.pkl", "wb") as f:
            pickle.dump((phrased_sentences, cases, bp, tp), f)

    """load model if available"""
    try:
        w2v = Word2Vec.load("stored/apr17")
        print("loaded model\n")
        print(w2v, "\n")
    except FileNotFoundError as e:
        print(e)
        w2v = build_model(phrased_sentences)
    
    """save model"""
    w2v.save('stored/apr17')
    #print(dir(w2v.wv))
    
    """build vectoriser"""
    vectoriser = Word_Vectoriser(w2v.wv).fit(phrased_sentences)      
    
    """encoding"""
    print("encoding...\n")
    encoded_extracts = encode_extracts(cases, vectoriser, bp, tp)
    
    """PCA"""
    print("principal component analysis...\n")
    pca_transformer, compressed_cases = pca(encoded_extracts)

    print("reduced {} training cases into {} vectors of shape {}\n"
            .format(len(cases),
                len(compressed_cases), compressed_cases[0].shape))

    """clustering"""
    clustering(compressed_cases)
    
    """find similar cases"""    
    new_case = soup(requests.get("http://www.bailii.org/ew/cases/EWCA/Civ/2017/2.html",
        headers={'User-Agent': 'Mozilla/5.0'}).text, "lxml").text
    
    try:
        new_case_encoded = encode_extracts([('', new_case)], vectoriser, bp, tp)
        assert new_case_encoded.size > 0
        new_compressed = pca_transformer.transform(new_case_encoded)
        most_similar = compare_cases(compressed_cases, new_compressed)
    except Exception as e:
        print(e)
        print("no relevant extracts found")

    

if __name__ == "__main__":
    main()
