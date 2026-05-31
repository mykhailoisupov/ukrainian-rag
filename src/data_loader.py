import os
import sys
import time
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError

# Reconfigure stdout/stderr to use UTF-8, especially on Windows to prevent UnicodeEncodeError in terminal
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
    wikipedia.set_lang('uk')
    
    # Set a custom user agent to avoid being rate-limited / blocked by Wikimedia API
    wikipedia.set_user_agent("UkrainianRAGBot/1.0 (contact@example.com)")
    
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
        
        # Introduce a small delay to avoid hitting Wikimedia's rate limit
        time.sleep(1.0)
        
        try:
            # Fetch the article page content
            page = wikipedia.page(article)
            content = page.content
            
            # Save the article as a UTF-8 encoded .txt file
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
            
        except DisambiguationError as e:
            print(f"Warning: DisambiguationError for '{article}': {e}")
            results.append({
                "article": article,
                "chars": 0,
                "status": "skipped"
            })
        except PageError as e:
            print(f"Warning: PageError (not found) for '{article}': {e}")
            results.append({
                "article": article,
                "chars": 0,
                "status": "skipped"
            })
        except Exception as e:
            print(f"Warning: Unexpected exception for '{article}': {e}")
            results.append({
                "article": article,
                "chars": 0,
                "status": "skipped"
            })
            
            
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

