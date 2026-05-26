import os
import shutil
from PIL import Image
from PIL import ImageFile
from concurrent.futures import ThreadPoolExecutor
import warnings
import math
import json
import time
import re
from pathlib import Path
from datetime import datetime, timezone
import google.generativeai as genai
import base64
import requests
from io import BytesIO
from dotenv import load_dotenv
import sqlite3

# Load environment variables from .env file
load_dotenv()

# Suppress DecompressionBombWarning and increase image size limit
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

# Define directories
from pathlib import Path

source_dir = Path("~/new").expanduser()
destination_dir = Path("~/storage/cache").expanduser()
main_dir = Path("~/storage/main").expanduser()
output_file = Path("~/storage/index.json").expanduser()
db_file = Path("~/storage/processed_files.db").expanduser()
# Ensure destination directories exist
os.makedirs(destination_dir, exist_ok=True)
os.makedirs(main_dir, exist_ok=True)

# Get Gemini API Keys from environment variables
GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")
if not GEMINI_API_KEYS or not GEMINI_API_KEYS[0]:
    # Fallback to single key if plural not found
    single_key = os.getenv("GEMINI_API_KEY")
    if single_key:
        GEMINI_API_KEYS = [single_key]
    else:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

current_key_index = 0
images_processed_with_current_key = 0

def rotate_api_key():
    """Rotate to the next API key after 15 images."""
    global current_key_index, images_processed_with_current_key
    images_processed_with_current_key += 1
    
    if images_processed_with_current_key >= 15:
        if len(GEMINI_API_KEYS) > 1:
            current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
            images_processed_with_current_key = 0
            new_key = GEMINI_API_KEYS[current_key_index].strip()
            genai.configure(api_key=new_key)
            print(f"\n--- Switched to API Key {current_key_index + 1}/{len(GEMINI_API_KEYS)} ---")
        else:
            # Only one key, just reset counter to keep track but no switch possible
            images_processed_with_current_key = 0

# Initial configuration
genai.configure(api_key=GEMINI_API_KEYS[0].strip())

# Rate limiting variables
BATCH_SIZE = 9
BATCH_WAIT_TIME = 60  # seconds

# Predefined categories
CATEGORIES = [
    "#nature",
    "#anime",
    "#art",
    "#abstract",
    "#cars",
    "#architecture",
    "#minimal",
    "#tech",
    "#amoled"
]

def init_database():
    """Initialize SQLite database to track processed files."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_name TEXT UNIQUE,
            new_name TEXT,
            processed_date TEXT,
            process_type TEXT,
            analysis_data TEXT
        )
    ''')
    # Check if analysis_data column exists, if not add it
    cursor.execute("PRAGMA table_info(processed_files)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'analysis_data' not in columns:
        cursor.execute('ALTER TABLE processed_files ADD COLUMN analysis_data TEXT')
    conn.commit()
    conn.close()

def is_file_processed(original_name, process_type):
    """Check if a file has already been processed."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        'SELECT new_name, analysis_data FROM processed_files WHERE original_name = ? AND process_type = ?',
        (original_name, process_type)
    )
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0], result[1]
    return None, None

def mark_file_processed(original_name, new_name, process_type, analysis_data=None):
    """Mark a file as processed in the database."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO processed_files 
        (original_name, new_name, processed_date, process_type, analysis_data) 
        VALUES (?, ?, ?, ?, ?)
    ''', (original_name, new_name, datetime.now().isoformat(), process_type, analysis_data))
    conn.commit()
    conn.close()

def prepare_image_for_gemini(image_path):
    """Prepare image for Gemini API by resizing and converting to RGB."""
    try:
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
            return buffer.getvalue()
    except Exception as e:
        print(f"Error preparing image {image_path}: {e}")
        return None

def analyze_image_combined(image_path):
    """Perform full image analysis and filename generation in a single Gemini call."""
    # Rotate API key if needed
    rotate_api_key()
    
    image_bytes = prepare_image_for_gemini(image_path)
    if not image_bytes:
        return None

    try:
        # Set up the Gemini model
        model = genai.GenerativeModel('gemini-3.1-flash-lite')
        
        # Create the combined prompt
        prompt = f"""
        ANALYZE THIS IMAGE WITH EXTREME PRECISION FOR ALGORITHMIC CLASSIFICATION AND FILENAMING.
        
        1. CATEGORIZATION:
        Categorize the image into exactly one of these categories:
        {', '.join(CATEGORIES)}
        Note: #amoled is for images with pure black backgrounds and vibrant colors.
        
        2. FILENAME GENERATION:
        Generate a short, descriptive filename (without extension).
        - Use lowercase letters, numbers, and hyphens.
        - Be descriptive but concise (max 20 characters).
        
        3. DETAILED ANALYSIS:
        Return ONLY this JSON structure with SPECIFIC STANDARDIZED TERMS:
        
        {{
          "suggested_filename": "the-generated-filename",
          "category": "#category-name",
          "character_names": ["exact character names from anime/manga/games"],
          "series": "exact franchise name or unknown",
          "art_style": "SELECT ONE: anime | realistic | cartoon | pixel-art | digital-painting | oil-painting | watercolor | sketch | vector-art | 3d-render | photography | cel-shading | line-art | minimalist | abstract | grunge | vintage",
          "primary_colors": ["SELECT 3-5 FROM: red | crimson | scarlet | maroon | pink | rose | magenta | fuchsia | orange | coral | peach | yellow | gold | amber | green | emerald | lime | forest | cyan | teal | blue | azure | navy | purple | violet | lavender | brown | tan | beige | black | charcoal | gray | silver | white | cream"],
          "secondary_colors": ["SELECT 2-4 FROM ABOVE COLOR LIST"],
          "color_palette": "SELECT ONE: monochromatic | complementary | analogous | triadic | warm-tones | cool-tones | pastel | vibrant | muted | high-contrast | low-contrast | neon | earth-tones | jewel-tones",
          "mood": "SELECT ONE: cheerful | melancholic | energetic | calm | mysterious | romantic | dramatic | peaceful | intense | playful | serious | nostalgic | dreamy | dark | bright",
          "technique": "SELECT ONE: digital-painting | traditional-painting | photography | 3d-rendering | vector-graphics | pixel-art | mixed-media | pencil-drawing | ink-drawing | watercolor | oil-painting | acrylic-painting",
          "scene_description": "detailed description of what is shown",
          "character_details": {{
            "hair_color": "SELECT FROM COLOR LIST or unknown",
            "eye_color": "SELECT FROM COLOR LIST or unknown",
            "clothing_style": "SELECT ONE: casual | formal | fantasy | modern | traditional | futuristic | gothic | cute | elegant | sporty | military | school-uniform or unknown",
            "pose": "SELECT ONE: standing | sitting | lying | running | walking | dancing | fighting | flying | crouching | kneeling | portrait-pose or unknown",
            "facial_expression": "SELECT ONE: happy | sad | angry | surprised | neutral | smiling | serious | cute | determined | shy | confident or unknown"
          }},
          "environment": "SELECT ONE: indoor | outdoor | fantasy-world | urban | natural | space | underwater | sky | abstract-background | studio | bedroom | kitchen | park | forest | beach | mountain | city | desert | winter | summer | spring | autumn",
          "lighting": "SELECT ONE: natural-daylight | artificial-light | sunset | sunrise | golden-hour | blue-hour | night | neon | backlit | front-lit | side-lit | dramatic | soft | harsh | ambient | spotlight | candlelight",
          "composition": "SELECT ONE: portrait | landscape | close-up | wide-shot | medium-shot | centered | rule-of-thirds | symmetrical | asymmetrical | diagonal | vertical | horizontal",
          "art_quality": "SELECT ONE: professional | high-detail | medium-detail | sketch-quality | rough | polished | photorealistic | stylized | simple | complex",
          "style_influences": ["SELECT FROM: anime | manga | western-animation | disney | pixar | ghibli | cyberpunk | steampunk | gothic | kawaii | chibi | realistic | impressionism | surrealism | pop-art | art-nouveau | minimalism"],
          "objects": ["list prominent objects/items visible"],
          "textures": ["SELECT FROM: smooth | rough | metallic | fabric | glass | wood | stone | plastic | fur | skin | water | fire | clouds | grass | concrete | brick | leather"],
          "type": "SELECT ONE: character-portrait | full-body-character | multiple-characters | landscape | still-life | abstract-art | vehicle | animal | food | architecture | nature | technology | weapon | clothing | accessories",
          "tags": ["15-20 SPECIFIC searchable tags covering: character-name, series-name, colors, art-style, mood, objects, environment, lighting, pose, clothing, hair-color, eye-color, technique, quality, composition"]
        }}
        
        CRITICAL REQUIREMENTS:
        - Use EXACT terms from the SELECT lists above.
        - Return ONLY the JSON object, no explanations.
        """
        
        # Call the Gemini API
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
        
        # Parse JSON response
        response_text = response.text.strip()
        # Remove markdown formatting if present
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
        
        data = json.loads(response_text)
        
        # Clean up filename
        filename = data.get("suggested_filename", "image").lower()
        filename = re.sub(r'[^a-z0-9\-]', '', filename)
        if not filename:
            filename = "image"
        data["suggested_filename"] = filename
        
        # Validate category
        category = data.get("category", "#art")
        if category not in CATEGORIES:
            for valid_cat in CATEGORIES:
                if valid_cat.lower() in category.lower():
                    category = valid_cat
                    break
            else:
                category = "#art"
        data["category"] = category
        
        return data

    except Exception as e:
        print(f"Error in combined analysis for {image_path}: {str(e)}")
        return None

def compress_image(filename, rename_files=True):
    """Compress and convert images to WebP format while copying originals to main directory."""
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        source_path = os.path.join(source_dir, filename)
        
        # Check if file was already processed for renaming
        if rename_files:
            existing_new_name, existing_analysis = is_file_processed(filename, "rename")
            if existing_new_name:
                print(f"File {filename} already renamed to {existing_new_name}, skipping...")
                return existing_new_name
        
        # Generate new filename and analysis if renaming is enabled
        analysis_data = None
        if rename_files:
            try:
                analysis = analyze_image_combined(source_path)
                if analysis:
                    new_base_name = analysis.get("suggested_filename")
                    analysis_data = json.dumps(analysis)
                else:
                    new_base_name = os.path.splitext(filename)[0]
                
                file_extension = os.path.splitext(filename)[1]
                new_filename = f"{new_base_name}{file_extension}"
                
                # Ensure unique filename
                counter = 1
                while os.path.exists(os.path.join(main_dir, new_filename)):
                    new_filename = f"{new_base_name}-{counter}{file_extension}"
                    counter += 1
                
                # Check if files with old names exist and rename them
                old_main_path = os.path.join(main_dir, filename)
                old_cache_path = os.path.join(destination_dir, os.path.splitext(filename)[0] + ".webp")
                
                # Rename existing main file if it exists
                if os.path.exists(old_main_path):
                    new_main_path = os.path.join(main_dir, new_filename)
                    try:
                        os.rename(old_main_path, new_main_path)
                        print(f"Renamed existing main file: {filename} -> {new_filename}")
                    except Exception as e:
                        print(f"Error renaming main file {filename}: {e}")
                
                # Rename existing cache file if it exists
                if os.path.exists(old_cache_path):
                    new_cache_path = os.path.join(destination_dir, os.path.splitext(new_filename)[0] + ".webp")
                    try:
                        os.rename(old_cache_path, new_cache_path)
                        print(f"Renamed existing cache file: {os.path.splitext(filename)[0]}.webp -> {os.path.splitext(new_filename)[0]}.webp")
                    except Exception as e:
                        print(f"Error renaming cache file for {filename}: {e}")
                
                mark_file_processed(filename, new_filename, "rename", analysis_data)
                print(f"Generated new name and analysis: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Error generating filename for {filename}, using original: {e}")
                new_filename = filename
        else:
            new_filename = filename
        
        # Set up paths with new filename
        main_path = os.path.join(main_dir, new_filename)
        destination_path = os.path.join(destination_dir, os.path.splitext(new_filename)[0] + ".webp")
        
        # Copy original file to main directory with new name (only if it doesn't exist)
        if not os.path.exists(main_path):
            try:
                shutil.copy2(source_path, main_path)
                print(f"Copied original: {filename} -> {new_filename} to {main_dir}")
            except Exception as e:
                print(f"Error copying original file {filename}: {e}")
        
        # Get original file size in KB
        original_size = os.path.getsize(source_path) / 1024
        
        # Only create compressed version if it doesn't exist
        if not os.path.exists(destination_path):
            try:
                # Open the image
                with Image.open(source_path) as img:
                    # Get original dimensions
                    width, height = img.size
                    
                    # Resize if image is too large (max width or height of 1920px)
                    max_dimension = 1920
                    if width > max_dimension or height > max_dimension:
                        # Calculate new dimensions maintaining aspect ratio
                        if width > height:
                            new_width = max_dimension
                            new_height = math.floor(height * (max_dimension / width))
                        else:
                            new_height = max_dimension
                            new_width = math.floor(width * (max_dimension / height))
                        
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Save with different settings based on file size
                    if original_size > 1000:  # For large files (>1MB)
                        img.save(destination_path, "WEBP", quality=10, method=6)
                    elif original_size > 500:  # For medium files (>500KB)
                        img.save(destination_path, "WEBP", quality=20, method=6)
                    else:  # For smaller files
                        img.save(destination_path, "WEBP", quality=30, method=6)
                    
                    # Calculate compression ratio
                    new_size = os.path.getsize(destination_path) / 1024
                    compression_ratio = (1 - (new_size / original_size)) * 100
                    
                    print(f"{new_filename}: {original_size:.2f}KB → {new_size:.2f}KB ({compression_ratio:.2f}% reduction)")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Compressed version already exists: {os.path.splitext(new_filename)[0]}.webp")
        
        return new_filename if rename_files else filename

def run_optimization(rename_files=True):
    """Run the image optimization process with optional renaming."""
    print("Starting image optimization...")
    
    # Initialize database
    init_database()
    
    # Get list of files to process
    files_to_process = [f for f in os.listdir(source_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if rename_files:
        print(f"Processing {len(files_to_process)} files with renaming and rate limiting...")
        
        # Process files in batches with rate limiting
        for i in range(0, len(files_to_process), BATCH_SIZE):
            batch = files_to_process[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            
            print(f"\nProcessing batch {batch_num} ({len(batch)} files)...")
            
            # Process current batch
            for filename in batch:
                compress_image(filename, rename_files=True)
            
            # Wait if not the last batch
            if i + BATCH_SIZE < len(files_to_process):
                print(f"Batch {batch_num} complete. Waiting {BATCH_WAIT_TIME} seconds...")
                time.sleep(BATCH_WAIT_TIME)
    else:
        # Use ThreadPoolExecutor for faster processing without renaming
        with ThreadPoolExecutor(max_workers=6) as executor:
            executor.map(lambda f: compress_image(f, rename_files=False), files_to_process)
    
    print("Image compression and conversion completed.")

# Function to get files without extension
def get_file_without_extension(filename):
    return os.path.splitext(filename)[0]

# Function to determine resolution category and orientation
def get_resolution_info(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Determine orientation (Desktop vs Mobile)
            orientation = "Mobile" if height > width else "Desktop"
            
            # Determine resolution category
            if width >= 7680 or height >= 7680:
                resolution = "8K"
            elif width >= 3840 or height >= 3840:
                resolution = "4K"
            elif width >= 2560 or height >= 2560:
                resolution = "2K"
            elif width >= 1920 or height >= 1920:
                resolution = "1080p"
            elif width >= 1280 or height >= 1280:
                resolution = "720p"
            else:
                resolution = "SD"
                
            return {
                "width": width,
                "height": height,
                "resolution": resolution,
                "orientation": orientation
            }
    except Exception as e:
        print(f"Error reading image {image_path}: {str(e)}")
        return {
            "width": 0,
            "height": 0,
            "resolution": "Unknown",
            "orientation": "Unknown"
        }

# Function to get file timestamp in ISO format
def get_file_timestamp(file_path):
    try:
        mod_time = os.path.getmtime(file_path)
        # Convert Unix timestamp to ISO format string
        dt_object = datetime.fromtimestamp(mod_time, tz=timezone.utc)
        iso_timestamp = dt_object.isoformat()
        return iso_timestamp
    except Exception as e:
        print(f"Error getting timestamp for {file_path}: {str(e)}")
        return ""

# Function to compare two ISO format timestamps
def is_newer_timestamp(timestamp1, timestamp2):
    if not timestamp1:
        return False
    if not timestamp2:
        return True
    try:
        dt1 = datetime.fromisoformat(timestamp1)
        dt2 = datetime.fromisoformat(timestamp2)
        return dt1 > dt2
    except Exception as e:
        print(f"Error comparing timestamps: {str(e)}")
        return False

def run_indexing():
    """Run the image indexing process with rate limiting and continuous file updates."""
    print("\nStarting image indexing...")
    
    # Load existing index.json if it exists
    existing_entries = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                # Create a dictionary with file_name as key for quick lookup
                for entry in existing_data:
                    if 'file_name' in entry:
                        existing_entries[entry['file_name']] = entry
            print(f"Loaded {len(existing_entries)} existing entries from {output_file}")
        except Exception as e:
            print(f"Error loading existing index file: {str(e)}")
            existing_entries = {}

    # Get all files in cache directory
    cache_files = {}
    if os.path.exists(destination_dir):
        for file in os.listdir(destination_dir):
            if os.path.isfile(os.path.join(destination_dir, file)):
                base_name = get_file_without_extension(file)
                cache_files[base_name] = file

    # Get all files in main directory
    main_files = {}
    if os.path.exists(main_dir):
        for file in os.listdir(main_dir):
            if os.path.isfile(os.path.join(main_dir, file)):
                base_name = get_file_without_extension(file)
                main_files[base_name] = file

    # Create matching pairs
    result = []
    new_entries = 0
    updated_entries = 0
    entries_to_process = []

    # First, collect all entries that need processing
    for base_name in set(cache_files.keys()).union(set(main_files.keys())):
        cache_file = cache_files.get(base_name, "")
        main_file = main_files.get(base_name, "")
        
        if cache_file or main_file:  # Only include if at least one exists
            # Check if this file already exists in the index
            if base_name in existing_entries:
                # Entry already exists - keep it as-is, don't process or update it
                result.append(existing_entries[base_name])
            else:
                # Create a new entry (only process files NOT already in the index)
                entry = {
                    "file_name": base_name,
                    "file_cache_name": cache_file,
                    "file_main_name": main_file
                }
                
                # Add resolution information if main file exists
                if main_file:
                    main_path = os.path.join(main_dir, main_file)
                    if os.path.isfile(main_path):
                        resolution_info = get_resolution_info(main_path)
                        entry.update({
                            "width": resolution_info["width"],
                            "height": resolution_info["height"],
                            "resolution": resolution_info["resolution"],
                            "orientation": resolution_info["orientation"],
                            "timestamp": get_file_timestamp(main_path)
                        })
                        entries_to_process.append((entry, main_path, base_name, "new"))
                    else:
                        result.append(entry)
                        new_entries += 1
                elif cache_file:  # If no main file but cache file exists, get timestamp from cache
                    cache_path = os.path.join(destination_dir, cache_file)
                    if os.path.isfile(cache_path):
                        entry["timestamp"] = get_file_timestamp(cache_path)
                        entries_to_process.append((entry, cache_path, base_name, "new"))
                    else:
                        result.append(entry)
                        new_entries += 1
                else:
                    result.append(entry)
                    new_entries += 1

    # Write initial index file with existing entries that don't need processing
    print(f"Writing initial index with {len(result)} existing entries...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

    # Process images in batches with rate limiting and continuous file updates
    print(f"Collected {len(entries_to_process)} entries that need image analysis")
    
    for i in range(0, len(entries_to_process), BATCH_SIZE):
        batch = entries_to_process[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        print(f"\nProcessing indexing batch {batch_num} ({len(batch)} images)...")
        
        # Process each image in the current batch
        for j, (entry, image_path, base_name, entry_type) in enumerate(batch):
            # Check if we have analysis data in the database
            analysis = None
            filename = os.path.basename(image_path)
            
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                # Check both original_name and new_name columns
                cursor.execute('SELECT analysis_data FROM processed_files WHERE original_name = ? OR new_name = ?', (filename, filename))
                db_result = cursor.fetchone()
                conn.close()
                
                if db_result and db_result[0]:
                    analysis = json.loads(db_result[0])
                    print(f"  → Using stored analysis for {base_name}")
            except Exception as e:
                print(f"  → Error checking database for {base_name}: {e}")
            
            # If not in DB, analyze now
            if not analysis:
                analysis = analyze_image_combined(image_path)
            
            if analysis:
                entry["category"] = analysis.get("category", "#art")
                entry["data"] = analysis
            else:
                if "category" not in entry: entry["category"] = "#art"
                if "data" not in entry: entry["data"] = {}
            
            result.append(entry)
            if entry_type == "new":
                new_entries += 1
                print(f"New ({i+j+1}/{len(entries_to_process)}): {base_name} - {entry.get('category', 'unknown')} - {len(entry.get('data', {}).get('tags', []))} tags")
            else:
                updated_entries += 1
                print(f"Updated ({i+j+1}/{len(entries_to_process)}): {base_name} - {entry.get('category', 'unknown')} - {len(entry.get('data', {}).get('tags', []))} tags")
        
        # Write the index file once per batch (not after every image)
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4)
            print(f"Batch {batch_num} saved - Index file updated ({len(result)} total entries)")
        except Exception as e:
            print(f"  → Error updating index file: {e}")
        
        # No waiting required during indexing as per user request
    
    # Final write to ensure everything is saved
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

    print(f"\nIndex created successfully: {len(result)} total entries ({new_entries} new, {updated_entries} updated)")

def rename_existing_files():
    """Rename existing files in main_dir and destination_dir using AI."""
    print("\nScanning for existing files to rename...")
    
    # Get existing files in main directory
    main_files = []
    if os.path.exists(main_dir):
        for file in os.listdir(main_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(main_dir, file)):
                # Check if already processed
                if not is_file_processed(file, "rename")[0]:
                    main_files.append(file)
    
    # Get existing files in cache directory  
    cache_files = []
    if os.path.exists(destination_dir):
        for file in os.listdir(destination_dir):
            if file.lower().endswith('.webp') and os.path.isfile(os.path.join(destination_dir, file)):
                # Find corresponding original name
                base_name = get_file_without_extension(file)
                # Look for original file extensions
                for ext in ['.png', '.jpg', '.jpeg']:
                    original_name = base_name + ext
                    if not is_file_processed(original_name, "rename")[0]:
                        cache_files.append((file, original_name))
                        break
    
    total_files = len(main_files) + len(cache_files)
    
    if total_files == 0:
        print("No existing files found that need renaming.")
        return
    
    print(f"Found {len(main_files)} files in main directory and {len(cache_files)} cache files that can be renamed.")
    rename_existing = input("Do you want to rename existing files using AI? (y/n): ").lower().strip()
    
    if rename_existing not in ['y', 'yes']:
        print("Skipping existing file renaming.")
        return
    
    print(f"Renaming {total_files} existing files with rate limiting...")
    
    # Combine all files to process
    files_to_rename = []
    
    # Add main files
    for file in main_files:
        files_to_rename.append(('main', file, file))
    
    # Add cache files
    for cache_file, original_name in cache_files:
        files_to_rename.append(('cache', cache_file, original_name))
    
    # Process in batches
    for i in range(0, len(files_to_rename), BATCH_SIZE):
        batch = files_to_rename[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        print(f"\nProcessing rename batch {batch_num} ({len(batch)} files)...")
        
        for file_type, current_name, original_name in batch:
            try:
                analysis = None
                analysis_data = None
                
                if file_type == 'main':
                    main_path = os.path.join(main_dir, current_name)
                    analysis = analyze_image_combined(main_path)
                    if analysis:
                        new_base_name = analysis.get("suggested_filename")
                        analysis_data = json.dumps(analysis)
                    else:
                        new_base_name = os.path.splitext(current_name)[0]
                    
                    file_extension = os.path.splitext(current_name)[1]
                    new_filename = f"{new_base_name}{file_extension}"
                    
                    # Ensure unique filename
                    counter = 1
                    while os.path.exists(os.path.join(main_dir, new_filename)) and new_filename != current_name:
                        new_filename = f"{new_base_name}-{counter}{file_extension}"
                        counter += 1
                    
                    if new_filename != current_name:
                        # Rename main file
                        new_main_path = os.path.join(main_dir, new_filename)
                        os.rename(main_path, new_main_path)
                        print(f"Renamed main file: {current_name} -> {new_filename}")
                        
                        # Check if corresponding cache file exists and rename it too
                        old_cache_name = os.path.splitext(current_name)[0] + ".webp"
                        old_cache_path = os.path.join(destination_dir, old_cache_name)
                        if os.path.exists(old_cache_path):
                            new_cache_name = os.path.splitext(new_filename)[0] + ".webp"
                            new_cache_path = os.path.join(destination_dir, new_cache_name)
                            os.rename(old_cache_path, new_cache_path)
                            print(f"Renamed cache file: {old_cache_name} -> {new_cache_name}")
                        
                        mark_file_processed(original_name, new_filename, "rename", analysis_data)
                    else:
                        print(f"No rename needed for: {current_name}")
                        mark_file_processed(original_name, current_name, "rename", analysis_data)
                
                elif file_type == 'cache':
                    # Check if we already renamed the main file in this batch
                    if not is_file_processed(original_name, "rename")[0]:
                        # Try to find main file to generate name from
                        main_candidates = []
                        base_name = get_file_without_extension(current_name)
                        for ext in ['.png', '.jpg', '.jpeg']:
                            candidate = base_name + ext
                            candidate_path = os.path.join(main_dir, candidate)
                            if os.path.exists(candidate_path):
                                main_candidates.append((candidate, candidate_path))
                        
                        if main_candidates:
                            # Use the first found main file
                            main_file, main_path = main_candidates[0]
                            analysis = analyze_image_combined(main_path)
                        else:
                            # Use cache file itself
                            cache_path = os.path.join(destination_dir, current_name)
                            analysis = analyze_image_combined(cache_path)
                        
                        if analysis:
                            new_base_name = analysis.get("suggested_filename")
                            analysis_data = json.dumps(analysis)
                        else:
                            new_base_name = os.path.splitext(current_name)[0]
                        
                        new_cache_name = f"{new_base_name}.webp"
                        
                        # Ensure unique filename
                        counter = 1
                        while os.path.exists(os.path.join(destination_dir, new_cache_name)) and new_cache_name != current_name:
                            new_cache_name = f"{new_base_name}-{counter}.webp"
                            counter += 1
                        
                        if new_cache_name != current_name:
                            # Rename cache file
                            cache_path = os.path.join(destination_dir, current_name)
                            new_cache_path = os.path.join(destination_dir, new_cache_name)
                            os.rename(cache_path, new_cache_path)
                            print(f"Renamed cache file: {current_name} -> {new_cache_name}")
                        
                        mark_file_processed(original_name, new_cache_name if new_cache_name != current_name else current_name, "rename", analysis_data)
            
            except Exception as e:
                print(f"Error renaming {current_name}: {e}")
        
        # Wait if not the last batch
        if i + BATCH_SIZE < len(files_to_rename):
            print(f"Rename batch {batch_num} complete. Waiting {BATCH_WAIT_TIME} seconds...")
            time.sleep(BATCH_WAIT_TIME)
    
    print("Existing file renaming completed.")

def main():
    """Main execution function that runs optimization first, then indexing."""
    print("=" * 60)
    print("STARTING CONTINUOUS IMAGE PROCESSING WORKFLOW")
    print("=" * 60)
    
    # Initialize database
    init_database()
    
    # Ask user if they want to rename existing files first
    existing_rename_choice = input("Do you want to rename existing files in main/cache directories? (y/n): ").lower().strip()
    if existing_rename_choice in ['y', 'yes']:
        rename_existing_files()
    
    # Ask user if they want to rename new files during optimization
    rename_choice = input("Do you want to rename new files using AI during optimization? (y/n): ").lower().strip()
    rename_files = rename_choice in ['y', 'yes']
    
    # Step 1: Run optimization
    run_optimization(rename_files=rename_files)
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE - STARTING INDEXING")
    print("=" * 60)
    
    # Step 2: Run indexing
    run_indexing()
    
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
