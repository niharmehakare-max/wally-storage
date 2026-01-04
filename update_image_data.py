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
BATCH_SIZE = 15
BATCH_WAIT_TIME = 60  # seconds
MAX_RETRIES = 2
RETRY_DELAY = 10  # seconds

# Available Gemini models (in order of preference)
GEMINI_MODELS = [
    'gemini-2.5-flash'
]

# Current model index
current_model_index = 0

def get_current_model():
    """Get the current Gemini model to use."""
    global current_model_index
    if current_model_index < len(GEMINI_MODELS):
        return GEMINI_MODELS[current_model_index]
    else:
        # If we've exhausted all models, reset to first one
        current_model_index = 0
        return GEMINI_MODELS[0]

def switch_to_next_model():
    """Switch to the next available Gemini model."""
    global current_model_index
    current_model_index += 1
    if current_model_index < len(GEMINI_MODELS):
        print(f"  → Switching to model: {GEMINI_MODELS[current_model_index]}")
        return True
    else:
        print(f"  → All models exhausted, resetting to first model")
        current_model_index = 0
        return False

def generate_image_data(image_path, retry_count=0):
    """Generate detailed image analysis data using Gemini with retry logic."""
    # Rotate API key if needed
    rotate_api_key()
    
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
            img.save(buffer, format="JPEG", quality=85)
            image_bytes = buffer.getvalue()
        
        # Set up the Gemini model
        current_model = get_current_model()
        model = genai.GenerativeModel(current_model)
        print(f"  → Using model: {current_model}")
        
        # Create a highly specific prompt optimized for algorithmic processing
        prompt = """
        ANALYZE THIS IMAGE WITH EXTREME PRECISION FOR ALGORITHMIC CLASSIFICATION.
        
        Return ONLY this JSON structure with SPECIFIC STANDARDIZED TERMS:
        
        {
          "character_names": ["exact character names from anime/manga/games"],
          "series": "exact franchise name or unknown",
          "art_style": "SELECT ONE: anime | realistic | cartoon | pixel-art | digital-painting | oil-painting | watercolor | sketch | vector-art | 3d-render | photography | cel-shading | line-art | minimalist | abstract | grunge | vintage",
          "primary_colors": ["SELECT 3-5 FROM: red | crimson | scarlet | maroon | pink | rose | magenta | fuchsia | orange | coral | peach | yellow | gold | amber | green | emerald | lime | forest | cyan | teal | blue | azure | navy | purple | violet | lavender | brown | tan | beige | black | charcoal | gray | silver | white | cream"],
          "secondary_colors": ["SELECT 2-4 FROM ABOVE COLOR LIST"],
          "color_palette": "SELECT ONE: monochromatic | complementary | analogous | triadic | warm-tones | cool-tones | pastel | vibrant | muted | high-contrast | low-contrast | neon | earth-tones | jewel-tones",
          "mood": "SELECT ONE: cheerful | melancholic | energetic | calm | mysterious | romantic | dramatic | peaceful | intense | playful | serious | nostalgic | dreamy | dark | bright",
          "technique": "SELECT ONE: digital-painting | traditional-painting | photography | 3d-rendering | vector-graphics | pixel-art | mixed-media | pencil-drawing | ink-drawing | watercolor | oil-painting | acrylic-painting",
          "scene_description": "detailed description of what is shown",
          "character_details": {
            "hair_color": "SELECT FROM COLOR LIST or unknown",
            "eye_color": "SELECT FROM COLOR LIST or unknown",
            "clothing_style": "SELECT ONE: casual | formal | fantasy | modern | traditional | futuristic | gothic | cute | elegant | sporty | military | school-uniform or unknown",
            "pose": "SELECT ONE: standing | sitting | lying | running | walking | dancing | fighting | flying | crouching | kneeling | portrait-pose or unknown",
            "facial_expression": "SELECT ONE: happy | sad | angry | surprised | neutral | smiling | serious | cute | determined | shy | confident or unknown"
          },
          "environment": "SELECT ONE: indoor | outdoor | fantasy-world | urban | natural | space | underwater | sky | abstract-background | studio | bedroom | kitchen | park | forest | beach | mountain | city | desert | winter | summer | spring | autumn",
          "lighting": "SELECT ONE: natural-daylight | artificial-light | sunset | sunrise | golden-hour | blue-hour | night | neon | backlit | front-lit | side-lit | dramatic | soft | harsh | ambient | spotlight | candlelight",
          "composition": "SELECT ONE: portrait | landscape | close-up | wide-shot | medium-shot | centered | rule-of-thirds | symmetrical | asymmetrical | diagonal | vertical | horizontal",
          "art_quality": "SELECT ONE: professional | high-detail | medium-detail | sketch-quality | rough | polished | photorealistic | stylized | simple | complex",
          "style_influences": ["SELECT FROM: anime | manga | western-animation | disney | pixar | ghibli | cyberpunk | steampunk | gothic | kawaii | chibi | realistic | impressionism | surrealism | pop-art | art-nouveau | minimalism"],
          "objects": ["list prominent objects/items visible"],
          "textures": ["SELECT FROM: smooth | rough | metallic | fabric | glass | wood | stone | plastic | fur | skin | water | fire | clouds | grass | concrete | brick | leather"],
          "type": "SELECT ONE: character-portrait | full-body-character | multiple-characters | landscape | still-life | abstract-art | vehicle | animal | food | architecture | nature | technology | weapon | clothing | accessories",
          "tags": ["15-20 SPECIFIC searchable tags covering: character-name, series-name, colors, art-style, mood, objects, environment, lighting, pose, clothing, hair-color, eye-color, technique, quality, composition"]
        }
        
        CRITICAL REQUIREMENTS:
        - Use EXACT terms from the SELECT lists above
        - For colors: Use specific names like "crimson", "azure", "emerald" NOT generic terms
        - Include character names if recognizable from popular media
        - Include series/franchise names if identifiable  
        - Tags must be hyphenated (e.g., "blue-hair", "night-scene", "fantasy-art")
        - Provide 15-20 tags for maximum searchability
        - Use "unknown" only when truly not applicable
        
        Return ONLY the JSON object, no explanations."""
        
        # Configure generation parameters for better consistency
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Call the Gemini API with safety settings
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        try:
            response = model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": image_bytes}],
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        except Exception as api_error:
            error_msg = str(api_error).lower()
            print(f"  → API Error: {str(api_error)}")
            
            # Check if it's a rate limit or quota error
            if any(keyword in error_msg for keyword in ["rate limit", "quota", "429", "resource has been exhausted"]):
                print(f"  → Rate limit/quota hit, attempting to switch models...")
                if switch_to_next_model():
                    print(f"  → Retrying with new model in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                    return generate_image_data(image_path, retry_count)
                else:
                    print(f"  → All models exhausted, waiting longer...")
                    time.sleep(BATCH_WAIT_TIME)  # Wait longer before retrying with first model
            
            if retry_count < MAX_RETRIES:
                print(f"  → Retrying in {RETRY_DELAY} seconds... (attempt {retry_count + 1}/{MAX_RETRIES + 1})")
                time.sleep(RETRY_DELAY)
                return generate_image_data(image_path, retry_count + 1)
            return generate_fallback_data(image_path)
        
        # Check for blocked content
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            if hasattr(response.prompt_feedback, 'block_reason'):
                print(f"  → Content blocked: {response.prompt_feedback.block_reason}")
                return generate_fallback_data(image_path)
        
        # Check if candidates are empty
        if not hasattr(response, 'candidates') or not response.candidates:
            print(f"  → No response candidates generated")
            if retry_count < MAX_RETRIES:
                print(f"  → Retrying in {RETRY_DELAY} seconds... (attempt {retry_count + 1}/{MAX_RETRIES + 1})")
                time.sleep(RETRY_DELAY)
                return generate_image_data(image_path, retry_count + 1)
            return generate_fallback_data(image_path)
        
        # Check if response text is available
        try:
            response_text = response.text.strip()
            if not response_text:
                print(f"  → Empty response text")
                if retry_count < MAX_RETRIES:
                    print(f"  → Retrying in {RETRY_DELAY} seconds... (attempt {retry_count + 1}/{MAX_RETRIES + 1})")
                    time.sleep(RETRY_DELAY)
                    return generate_image_data(image_path, retry_count + 1)
                return generate_fallback_data(image_path)
        except (ValueError, AttributeError, Exception) as text_error:
            print(f"  → Error accessing response text: {str(text_error)[:50]}")
            if retry_count < MAX_RETRIES:
                print(f"  → Retrying in {RETRY_DELAY} seconds... (attempt {retry_count + 1}/{MAX_RETRIES + 1})")
                time.sleep(RETRY_DELAY)
                return generate_image_data(image_path, retry_count + 1)
            return generate_fallback_data(image_path)
        
        # Parse JSON response with enhanced error handling
        try:
            # Enhanced JSON cleaning
            response_text = clean_json_response(response_text)
            
            # Try to parse JSON
            data = json.loads(response_text)
            
            # Validate and clean the data with enhanced structure
            cleaned_data = validate_and_clean_data(data)
            
            return cleaned_data
            
        except json.JSONDecodeError as e:
            print(f"  → JSON parsing failed: {str(e)[:100]}...")
            # Try to extract and fix common JSON issues
            try:
                fixed_json = fix_json_response(response_text)
                if fixed_json:
                    data = json.loads(fixed_json)
                    cleaned_data = validate_and_clean_data(data)
                    return cleaned_data
            except:
                pass
            
            if retry_count < MAX_RETRIES:
                print(f"  → Retrying in {RETRY_DELAY} seconds... (attempt {retry_count + 1}/{MAX_RETRIES + 1})")
                time.sleep(RETRY_DELAY)
                return generate_image_data(image_path, retry_count + 1)
            return generate_fallback_data(image_path)

    except Exception as e:
        error_msg = str(e)
        print(f"  → Unexpected error: {error_msg[:100]}...")
        
        # Check for specific error types that might benefit from retry or model switching
        if any(keyword in error_msg.lower() for keyword in ["blocked", "empty", "timeout", "rate limit", "quota", "429"]):
            if any(keyword in error_msg.lower() for keyword in ["rate limit", "quota", "429"]):
                print(f"  → Rate limit detected, attempting to switch models...")
                if switch_to_next_model():
                    print(f"  → Retrying with new model...")
                    return generate_image_data(image_path, retry_count)
                else:
                    print(f"  → All models exhausted, waiting longer...")
                    time.sleep(BATCH_WAIT_TIME)
            
            if retry_count < MAX_RETRIES:
                print(f"  → Retrying in {RETRY_DELAY} seconds... (attempt {retry_count + 1}/{MAX_RETRIES + 1})")
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
    
    # Fix common JSON issues
    response_text = response_text.replace('\\n', ' ').replace('\\t', ' ')
    response_text = re.sub(r'\s+', ' ', response_text)
    
    return response_text.strip()

def fix_json_response(response_text):
    """Attempt to fix common JSON formatting issues."""
    try:
        # Remove markdown and extra text
        cleaned = clean_json_response(response_text)
        
        # Try to fix missing quotes around keys
        cleaned = re.sub(r'(\w+):', r'"\1":', cleaned)
        
        # Fix trailing commas
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        # Try to validate the structure
        if cleaned.startswith('{') and cleaned.endswith('}'):
            return cleaned
        
        return None
    except:
        return None

def validate_and_clean_data(data):
    """Validate and clean the parsed JSON data."""
    # Ensure all required fields exist with proper types
    cleaned_data = {
        "character_names": data.get("character_names", []) if isinstance(data.get("character_names"), list) else [],
        "series": str(data.get("series", "unknown")),
        "art_style": str(data.get("art_style", "unknown")),
        "primary_colors": data.get("primary_colors", []) if isinstance(data.get("primary_colors"), list) else [],
        "secondary_colors": data.get("secondary_colors", []) if isinstance(data.get("secondary_colors"), list) else [],
        "color_palette": str(data.get("color_palette", "unknown")),
        "mood": str(data.get("mood", "unknown")),
        "technique": str(data.get("technique", "unknown")),
        "scene_description": str(data.get("scene_description", "unknown")),
        "character_details": {
            "hair_color": str(data.get("character_details", {}).get("hair_color", "unknown")),
            "eye_color": str(data.get("character_details", {}).get("eye_color", "unknown")),
            "clothing_style": str(data.get("character_details", {}).get("clothing_style", "unknown")),
            "pose": str(data.get("character_details", {}).get("pose", "unknown")),
            "facial_expression": str(data.get("character_details", {}).get("facial_expression", "unknown"))
        },
        "environment": str(data.get("environment", "unknown")),
        "lighting": str(data.get("lighting", "unknown")),
        "composition": str(data.get("composition", "unknown")),
        "art_quality": str(data.get("art_quality", "unknown")),
        "style_influences": data.get("style_influences", []) if isinstance(data.get("style_influences"), list) else [],
        "objects": data.get("objects", []) if isinstance(data.get("objects"), list) else [],
        "textures": data.get("textures", []) if isinstance(data.get("textures"), list) else [],
        "type": str(data.get("type", "unknown")),
        "tags": data.get("tags", []) if isinstance(data.get("tags"), list) else []
    }
    
    # Ensure all array fields contain only strings and remove empty values
    for key in ["character_names", "primary_colors", "secondary_colors", "style_influences", "objects", "textures", "tags"]:
        cleaned_data[key] = [str(item).strip() for item in cleaned_data[key] if item and str(item).strip()]
    
    # Ensure we have at least some tags
    if not cleaned_data["tags"]:
        # Generate some basic tags from other fields
        basic_tags = []
        if cleaned_data["type"] != "unknown":
            basic_tags.append(cleaned_data["type"])
        if cleaned_data["primary_colors"]:
            basic_tags.extend(cleaned_data["primary_colors"][:3])
        if cleaned_data["mood"] != "unknown":
            basic_tags.append(cleaned_data["mood"])
        cleaned_data["tags"] = basic_tags
    
    return cleaned_data

def generate_fallback_data(image_path):
    """Generate enhanced fallback data based on filename analysis."""
    filename = os.path.basename(image_path).lower()
    base_name = os.path.splitext(filename)[0]
    
    # Basic categorization based on filename
    fallback_type = "unknown"
    fallback_tags = []
    primary_colors = []
    secondary_colors = []
    
    # Enhanced keyword matching
    type_keywords = {
        "anime": ["anime", "manga", "waifu", "girl", "boy", "character", "kawaii"],
        "abstract": ["abstract", "geometric", "pattern", "lines", "shapes"],
        "nature": ["nature", "landscape", "forest", "mountain", "tree", "flower", "sky"],
        "automotive": ["car", "vehicle", "bike", "motorcycle", "road", "racing"],
        "tech": ["tech", "cyber", "digital", "neon", "circuit", "robot"],
        "architecture": ["building", "house", "tower", "bridge", "city", "urban"],
        "minimal": ["minimal", "clean", "simple", "basic", "plain"],
        "dark": ["dark", "night", "shadow", "black", "gothic"],
        "fantasy": ["fantasy", "magic", "dragon", "castle", "fairy"]
    }
    
    for category, keywords in type_keywords.items():
        if any(word in base_name for word in keywords):
            fallback_type = category
            fallback_tags.extend(keywords[:2])
            break
    
    # Enhanced color detection
    color_map = {
        "red": ["red", "crimson", "scarlet", "rose"],
        "blue": ["blue", "azure", "navy", "cyan", "teal"],
        "green": ["green", "emerald", "lime", "forest"],
        "yellow": ["yellow", "gold", "amber", "lemon"],
        "purple": ["purple", "violet", "magenta", "lavender"],
        "orange": ["orange", "coral", "peach", "sunset"],
        "pink": ["pink", "rose", "blush", "cherry"],
        "black": ["black", "dark", "shadow", "night"],
        "white": ["white", "light", "bright", "snow"],
        "brown": ["brown", "tan", "beige", "wood"],
        "gray": ["gray", "grey", "silver", "steel"]
    }
    
    for main_color, variations in color_map.items():
        for variation in variations:
            if variation in base_name:
                if main_color not in primary_colors:
                    primary_colors.append(main_color)
                    fallback_tags.append(main_color)
                break
    
    # Extract descriptive words from filename
    words = re.findall(r'[a-zA-Z]+', base_name)
    meaningful_words = [word for word in words if len(word) > 2 and word not in ['png', 'jpg', 'jpeg', 'webp', 'image', 'wallpaper']]
    fallback_tags.extend(meaningful_words[:5])
    
    # Remove duplicates and limit tags
    fallback_tags = list(dict.fromkeys(fallback_tags))[:12]  # Preserve order while removing duplicates
    
    print(f"  → Enhanced fallback data: {len(fallback_tags)} tags, {len(primary_colors)} colors")
    
    return {
        "character_names": [],
        "series": "unknown",
        "art_style": f"unknown - {fallback_type} style" if fallback_type != "unknown" else "unknown",
        "primary_colors": primary_colors[:5],
        "secondary_colors": secondary_colors,
        "color_palette": f"{', '.join(primary_colors)} tones" if primary_colors else "unknown",
        "mood": "unknown",
        "technique": "unknown",
        "scene_description": f"appears to be {fallback_type} related content" if fallback_type != "unknown" else "unknown",
        "character_details": {
            "hair_color": "unknown",
            "eye_color": "unknown",
            "clothing_style": "unknown",
            "pose": "unknown",
            "facial_expression": "unknown"
        },
        "environment": "unknown",
        "lighting": "unknown",
        "composition": "unknown",
        "art_quality": "unknown",
        "style_influences": [fallback_type] if fallback_type != "unknown" else [],
        "objects": meaningful_words[:3],
        "textures": [],
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
