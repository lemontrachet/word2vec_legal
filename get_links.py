from bs4 import BeautifulSoup as bs
import requests
import re


def enqueue_links():
    queue = []
    for year in range(2000, 2018):
        soup = get_soup(year)

        # collect links
        links = soup.find_all("a")

        # extract links to all cases in index
        pattern = re.compile('EWHC/QB/{}/.*\.html'.format(year))
                
        # place links in queue
        queue.extend(queue_links(links, pattern))
        
    return queue
    
    
def queue_links(links, p):
    i = 0
    queue = []
    for case in links:
        m = re.search(p, str(case))
        if m != None:
            queue.append("".join(["http://www.bailii.org/ew/cases/", m.group()]))
        i += 1
    return queue


def get_soup(year):
    # bailii database to search
    url = ("http://www.bailii.org/ew/cases/EWHC/QB/{}/".format(year))
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    html_content = r.text
    return bs(html_content, "lxml")




if __name__ == "__main__":
    queue = enqueue_links()
    for q in queue:
        print(q)
    print("found {} links".format(len(queue)))

