import requests
from bs4 import BeautifulSoup
import os
from PIL import Image
from io import BytesIO

def fetch_images(query, num_images, site):
    if site == "pixabay":
        url = f'https://pixabay.com/images/search/{query}/'
    elif site == "unsplash":
        url = f'https://unsplash.com/s/photos/{query}'
    else:
        return []

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    if site == "pixabay":
        images = soup.find_all('img', class_='responsive-image')  # Update with actual class name
    elif site == "unsplash":
        images = soup.find_all('img', class_='_2zEKz')  # Update with actual class name

    urls = [img['src'] for img in images if 'src' in img.attrs][:num_images]
    print(f"Found {len(urls)} URLs for {query} on {site}")
    return urls

def download_image(url, folder, actor_name, count):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            file_name = os.path.join(folder, f"{actor_name.replace(' ', '_')}_{count}.jpg")
            image.save(file_name)
            print(f"Downloaded: {file_name}")
        else:
            print(f"Failed to download {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def download_images(actor_name, num_images):
    folder = f"{actor_name.replace(' ', '_')}_images"
    os.makedirs(folder, exist_ok=True)

    for site in ["pixabay", "unsplash"]:
        urls = fetch_images(actor_name, num_images, site)
        for count, url in enumerate(urls):
            download_image(url, folder, actor_name, count + 1)

if __name__ == "__main__":
    actors = [
        "Denzel Washington",
        "Leonardo DiCaprio",
        "Meryl Streep",
        "Robert Downey Jr",
        "Tom Hanks"
    ]
    
    num_images = 100  # Adjust the number of images you want to download
    for actor in actors:
        print(f"Downloading images for {actor}...")
        download_images(actor, num_images)
