import os
import sys
import time
import requests
import json

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def fetch_and_save_wikipedia_articles():
    """
    Fetches full-text Wikipedia articles in Ukrainian, saves them as raw .txt files,
    and displays a summary report of characters saved and download status.
    """
    # Using requests instead of wikipedia library
    
    articles = [
        'Київ', 
        'Україна', 
        'Штучний інтелект', 
        'Машинне навчання', 
        'Тарас Шевченко', 
        'Верховна Рада', 
        'Львів', 
        'Харків', 
        'Чорнобиль', 
        'Голодомор'
    ]
    
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    total_saved = 0
    total_chars = 0
    
    print("Starting download of Ukrainian Wikipedia articles...\n")
    
    for article in articles:
        filename_title = article.replace(" ", "_")
        file_path = os.path.join(output_dir, f"{filename_title}.txt")
        
        max_retries = 3
        for attempt in range(max_retries):
            time.sleep(1.0)
            try:
                headers = {'User-Agent': 'UkrainianRAGBot/1.0 (mykhailoisupov@github)'}
                url = "https://uk.wikipedia.org/w/api.php"
                params = {
                    'action': 'query',
                    'prop': 'extracts',
                    'explaintext': '1',
                    'titles': article,
                    'format': 'json',
                    'redirects': '1'
                }
                response = requests.get(url, headers=headers, params=params, timeout=15)
                data = response.json()
                
                pages = data.get('query', {}).get('pages', {})
                if not pages or '-1' in pages:
                    print(f"Warning: PageError (not found) for '{article}'")
                    results.append({"article": article, "chars": 0, "status": "skipped"})
                    break
                    
                page_data = list(pages.values())[0]
                content = page_data.get('extract', '')
                
                if not content:
                    print(f"Warning: No text content returned for '{article}'")
                    results.append({"article": article, "chars": 0, "status": "skipped"})
                    break
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                    
                char_count = len(content)
                results.append({
                    "article": article,
                    "chars": char_count,
                    "status": "success"
                })
                total_saved += 1
                total_chars += char_count
                print(f"Successfully downloaded: {article}")
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Retry {attempt+1}/{max_retries} for '{article}'...")
                    time.sleep(2.0 * (attempt + 1))
                else:
                    print(f"Warning: Unexpected exception for '{article}': {e}")
                    results.append({"article": article, "chars": 0, "status": "skipped"})
            
            
    print("\n" + "=" * 65)
    print(f"{'Article Name':<25} | {'Chars Saved':<15} | {'Status':<15}")
    print("=" * 65)
    for res in results:
        print(f"{res['article']:<25} | {res['chars']:<15} | {res['status']:<15}")
    print("=" * 65)
    print(f"\nTotal articles saved: {total_saved}")
    print(f"Total characters: {total_chars}")

if __name__ == '__main__':
    fetch_and_save_wikipedia_articles()
