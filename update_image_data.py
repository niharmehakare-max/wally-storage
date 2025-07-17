import os
import json
import time
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Define directories
destination_dir = r"D:\storage\cache"
main_dir = r"D:\storage\main"
output_file = r"D:\storage\index.json"

# Get Gemini API Key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Rate limiting variables
BATCH_SIZE = 15
BATCH_WAIT_TIME = 60  # seconds
MAX_RETRIES = 2
RETRY_DELAY = 10  # seconds

def generate_image_data(image_path, retry_count=0):
    """Generate detailed image analysis data using Gemini with retry logic."""
    try:
        # Read the image file
        with Image.open(image_path) as img:
            # Resize image if too large to save bandwidth and processing time
            max_size = 800
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert RGBA to RGB if necessary (for JPEG compatibility)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create a white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to bytes
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
        
        # Set up the Gemini model
        model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
        
        # Create the prompt with better structure
        prompt = """
        Analyze this image and return a JSON object with the following structure:

        {
            "character_names": ["list of recognizable character names or empty array"],
            "series": "anime/game/series name or unknown",
            "art_style": "description of art style",
            "color_scheme": "dominant colors description",
            "mood": "emotional tone/mood",
            "technique": "art technique used",
            "scene_description": "what's happening in the image",
            "type": "category like anime, game art, fan art, original art, abstract, etc.",
            "tags": ["10-15 relevant tags covering characters, series, style, colors, mood, themes"]
        }

        Focus on:
        - Character identification (anime/game characters)
        - Art style and technique
        - Color scheme and mood
        - Scene content
        - Relevant searchable tags

        Return ONLY the JSON object, no other text.
        """
        
        # Call the Gemini API
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
        
        # Check if response is blocked
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            if hasattr(response.prompt_feedback, 'block_reason'):
                print(f"  → Prompt blocked for {os.path.basename(image_path)}: {response.prompt_feedback.block_reason}")
                return generate_fallback_data(image_path)
        
        # Check if response is empty
        if not response.text or not response.text.strip():
            print(f"  → Empty response for {os.path.basename(image_path)}")
            if retry_count < MAX_RETRIES:
                print(f"  → Retrying in {RETRY_DELAY} seconds... (attempt {retry_count + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
                return generate_image_data(image_path, retry_count + 1)
            return generate_fallback_data(image_path)
        
        # Parse JSON response
        try:
            response_text = response.text.strip()
            
            # Clean up response text
            response_text = clean_json_response(response_text)
            
            data = json.loads(response_text)
            
            # Validate and clean the data
            cleaned_data = {
                "character_names": data.get("character_names", []) if isinstance(data.get("character_names"), list) else [],
                "series": str(data.get("series", "unknown")),
                "art_style": str(data.get("art_style", "unknown")),
                "color_scheme": str(data.get("color_scheme", "unknown")),
                "mood": str(data.get("mood", "unknown")),
                "technique": str(data.get("technique", "unknown")),
                "scene_description": str(data.get("scene_description", "unknown")),
                "type": str(data.get("type", "unknown")),
                "tags": data.get("tags", []) if isinstance(data.get("tags"), list) else []
            }
            
            # Ensure tags is a list of strings
            cleaned_data["tags"] = [str(tag) for tag in cleaned_data["tags"] if tag]
            
            return cleaned_data
            
        except json.JSONDecodeError as e:
            print(f"  → JSON parsing error for {os.path.basename(image_path)}: {str(e)}")
            if retry_count < MAX_RETRIES:
                print(f"  → Retrying in {RETRY_DELAY} seconds... (attempt {retry_count + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
                return generate_image_data(image_path, retry_count + 1)
            return generate_fallback_data(image_path)

    except Exception as e:
        error_msg = str(e)
        print(f"  → Error generating image data for {os.path.basename(image_path)}: {error_msg}")
        
        # Check for specific error types that might benefit from retry
        if "blocked prompt" in error_msg.lower() or "empty" in error_msg.lower():
            if retry_count < MAX_RETRIES:
                print(f"  → Retrying in {RETRY_DELAY} seconds... (attempt {retry_count + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
                return generate_image_data(image_path, retry_count + 1)
        
        return generate_fallback_data(image_path)

def clean_json_response(response_text):
    """Clean up JSON response text to improve parsing."""
    # Remove markdown formatting
    if response_text.startswith('```json'):
        response_text = response_text[7:]
    if response_text.startswith('```'):
        response_text = response_text[3:]
    if response_text.endswith('```'):
        response_text = response_text[:-3]
    
    # Remove any text before the first {
    json_start = response_text.find('{')
    if json_start > 0:
        response_text = response_text[json_start:]
    
    # Remove any text after the last }
    json_end = response_text.rfind('}')
    if json_end > 0:
        response_text = response_text[:json_end + 1]
    
    return response_text.strip()

def generate_fallback_data(image_path):
    """Generate fallback data based on filename analysis."""
    filename = os.path.basename(image_path).lower()
    base_name = os.path.splitext(filename)[0]
    
    # Basic categorization based on filename
    fallback_type = "unknown"
    fallback_tags = []
    
    # Simple keyword matching
    if any(word in base_name for word in ['anime', 'manga', 'waifu', 'girl', 'boy']):
        fallback_type = "anime"
        fallback_tags.extend(["anime", "manga"])
    elif any(word in base_name for word in ['abstract', 'geometric', 'pattern']):
        fallback_type = "abstract"
        fallback_tags.extend(["abstract", "geometric"])
    elif any(word in base_name for word in ['nature', 'landscape', 'forest', 'mountain']):
        fallback_type = "nature"
        fallback_tags.extend(["nature", "landscape"])
    elif any(word in base_name for word in ['car', 'vehicle', 'bike', 'motorcycle']):
        fallback_type = "automotive"
        fallback_tags.extend(["car", "vehicle"])
    elif any(word in base_name for word in ['tech', 'cyber', 'digital', 'neon']):
        fallback_type = "tech"
        fallback_tags.extend(["tech", "digital"])
    
    # Color detection from filename
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'black', 'white', 'grey', 'gray']
    for color in colors:
        if color in base_name:
            fallback_tags.append(color)
    
    # Add filename-based tags
    words = re.findall(r'[a-zA-Z]+', base_name)
    fallback_tags.extend([word for word in words if len(word) > 2 and word not in ['png', 'jpg', 'jpeg', 'webp']])
    
    # Remove duplicates and limit tags
    fallback_tags = list(set(fallback_tags))[:10]
    
    print(f"  → Using fallback data with {len(fallback_tags)} tags")
    
    return {
        "character_names": [],
        "series": "unknown",
        "art_style": "unknown",
        "color_scheme": "unknown",
        "mood": "unknown",
        "technique": "unknown",
        "scene_description": "unknown",
        "type": fallback_type,
        "tags": fallback_tags
    }

def get_file_without_extension(filename):
    return os.path.splitext(filename)[0]

def update_image_data():
    """Update image data for existing images in main/cache directories."""
    print("=" * 60)
    print("IMAGE DATA UPDATE SCRIPT")
    print("=" * 60)
    
    # Load existing index.json if it exists
    existing_data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"Loaded {len(existing_data)} existing entries from index.json")
        except Exception as e:
            print(f"Error loading existing index file: {str(e)}")
            return
    else:
        print("No existing index.json found. Please run the main script first.")
        return
    
    # Ask user what type of update they want
    print("\nUpdate options:")
    print("1. Update entries missing data field")
    print("2. Update all entries (regenerate all data)")
    print("3. Update specific entries by pattern")
    print("4. Update entries with empty/fallback data")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        entries_to_update = []
        for entry in existing_data:
            if "data" not in entry:
                entries_to_update.append(entry)
        update_type = "missing data"
    elif choice == "2":
        entries_to_update = existing_data.copy()
        update_type = "all entries"
    elif choice == "3":
        pattern = input("Enter pattern to match in file names (e.g., 'anime', 'nature'): ").strip().lower()
        entries_to_update = []
        for entry in existing_data:
            if pattern in entry.get("file_name", "").lower():
                entries_to_update.append(entry)
        update_type = f"entries matching '{pattern}'"
    elif choice == "4":
        entries_to_update = []
        for entry in existing_data:
            if "data" in entry:
                data = entry["data"]
                # Check if data is empty or fallback
                if (not data.get("tags") or 
                    len(data.get("tags", [])) == 0 or 
                    data.get("series") == "unknown" and data.get("art_style") == "unknown"):
                    entries_to_update.append(entry)
        update_type = "entries with empty/fallback data"
    else:
        print("Invalid choice. Exiting.")
        return
    
    if not entries_to_update:
        print(f"No entries found for {update_type}.")
        return
    
    print(f"\nFound {len(entries_to_update)} entries for {update_type}.")
    confirm = input("Do you want to proceed? (y/n): ").lower().strip()
    
    if confirm not in ['y', 'yes']:
        print("Update cancelled.")
        return
    
    # Process entries in batches
    updated_count = 0
    errors_count = 0
    
    for i in range(0, len(entries_to_update), BATCH_SIZE):
        batch = entries_to_update[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        print(f"\nProcessing batch {batch_num} ({len(batch)} entries)...")
        
        for j, entry in enumerate(batch):
            try:
                # Find the image file
                image_path = None
                main_file = entry.get("file_main_name", "")
                cache_file = entry.get("file_cache_name", "")
                
                # Prefer main file over cache file
                if main_file:
                    potential_path = os.path.join(main_dir, main_file)
                    if os.path.isfile(potential_path):
                        image_path = potential_path
                
                if not image_path and cache_file:
                    potential_path = os.path.join(destination_dir, cache_file)
                    if os.path.isfile(potential_path):
                        image_path = potential_path
                
                if not image_path:
                    print(f"Warning: No image file found for {entry.get('file_name', 'unknown')}")
                    errors_count += 1
                    continue
                
                # Generate new data
                print(f"Processing ({i+j+1}/{len(entries_to_update)}): {entry.get('file_name', 'unknown')}...")
                data = generate_image_data(image_path)
                entry["data"] = data
                
                updated_count += 1
                print(f"  → Updated: {len(data.get('tags', []))} tags, type: {data.get('type', 'unknown')}")
                
                # Update the index file after each entry (for continuous saving)
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, indent=4)
                except Exception as e:
                    print(f"  → Error updating index file: {e}")
                
            except Exception as e:
                print(f"Error processing {entry.get('file_name', 'unknown')}: {e}")
                errors_count += 1
        
        # Wait if not the last batch
        if i + BATCH_SIZE < len(entries_to_update):
            print(f"Batch {batch_num} complete. Waiting {BATCH_WAIT_TIME} seconds...")
            time.sleep(BATCH_WAIT_TIME)
    
    # Final save
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)
        print(f"\nUpdate completed successfully!")
        print(f"Updated: {updated_count} entries")
        print(f"Errors: {errors_count} entries")
        print(f"Total processed: {len(entries_to_update)} entries")
    except Exception as e:
        print(f"Error saving final index file: {e}")

if __name__ == "__main__":
    update_image_data()
