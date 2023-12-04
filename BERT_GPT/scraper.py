import re
from googlesearch import search
import requests
from bs4 import BeautifulSoup

# Function to perform a Google search and extract snippets
def google_search_and_extract_text(query):
    results = []
    for result in search(query, num_results=3):
        try:
            # Send a GET request to the search result URL
            response = requests.get(result)
            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Parse the HTML content of the page
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract and store text content (excluding HTML tags)
                text_content = soup.get_text()
                results.append(text_content)
                # break
            else:
                print(f"Failed to retrieve content from {result}")
        except Exception as e:
            print(f"Error while fetching content from {result}: {e}")
    t = []
    for text_content in results:
        t.append(re.sub(r'\s+', ' ', text_content))
    # print(len(t))
    return t

# Example usage:
# query = input('Question: ')
# num_results = 5  # Number of search results to retrieve
# search_results = google_search_and_extract_text(query, num_results)



