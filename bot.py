import os
import asyncio
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict
import json
import time
import re

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

from PIL import Image
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv
import sqlite3
import math

# Load environment variables
load_dotenv()

# Get API keys from environment
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AUTHORIZED_USER_ID = int(os.getenv("AUTHORIZED_USER_ID", "0"))

if not BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")
if AUTHORIZED_USER_ID == 0:
    raise ValueError("AUTHORIZED_USER_ID not found in .env file")

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

# Define directories
temp_download_dir = r"D:\storage\temp_downloads"
destination_dir = r"D:\storage\cache"
main_dir = r"D:\storage\main"
output_file = r"D:\storage\index.json"
db_file = r"D:\storage\processed_files.db"

# Ensure directories exist
os.makedirs(temp_download_dir, exist_ok=True)
os.makedirs(destination_dir, exist_ok=True)
os.makedirs(main_dir, exist_ok=True)

# Rate limiting variables
BATCH_SIZE = 15
BATCH_WAIT_TIME = 60
MAX_RETRIES = 50  # Infinite retries until success
RETRY_DELAY = 10

# Available Gemini models
GEMINI_MODELS = [
    'gemini-2.5-flash-lite',
    'gemini-2.0-flash-lite',
    'gemini-1.5-flash',
    'gemini-1.5-pro'
]

# Current model index
current_model_index = 0

# Initialize bot and dispatcher with custom timeout
from aiogram.client.session.aiohttp import AiohttpSession

# Create session with 5 minute timeout (300 seconds)
session = AiohttpSession()
bot = Bot(token=BOT_TOKEN, session=session)
# Override the default timeout
bot.session.timeout = 300  # 5 minutes in seconds

storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Define states
class OrderStates(StatesGroup):
    waiting_for_images = State()

# Image collection storage
user_images: Dict[int, List[str]] = {}

# ==================== DATABASE FUNCTIONS ====================

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
            process_type TEXT
        )
    ''')
    conn.commit()
    conn.close()

def mark_file_processed(original_name, new_name, process_type):
    """Mark a file as processed in the database."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO processed_files 
        (original_name, new_name, processed_date, process_type) 
        VALUES (?, ?, ?, ?)
    ''', (original_name, new_name, datetime.now().isoformat(), process_type))
    conn.commit()
    conn.close()

# ==================== GEMINI HELPER FUNCTIONS ====================

def get_current_model():
    """Get the current Gemini model to use."""
    global current_model_index
    if current_model_index < len(GEMINI_MODELS):
        return GEMINI_MODELS[current_model_index]
    else:
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

# ==================== IMAGE PROCESSING FUNCTIONS ====================

def generate_filename(image_path, retry_count=0):
    """Generate a new filename using Gemini based on image content."""
    # Rotate API key if needed
    rotate_api_key()
    
    try:
        with Image.open(image_path) as img:
            max_size = 800
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
        
        current_model = get_current_model()
        model = genai.GenerativeModel(current_model)
        
        prompt = """
       Analyze this image carefully and identify:

        1. CHARACTER IDENTIFICATION:
        - Character names (if recognizable from popular anime, games, or media)
        - Character traits and appearance details
        - Art style (official art, fan art, original character, etc.)

        2. SOURCE IDENTIFICATION:
        - Anime/game/series name if identifiable
        - Franchise or universe
        - Art style classification

        3. VISUAL ELEMENTS:
        - Color scheme and mood
        - Art technique (digital painting, cel shading, realism, etc.)
        - Image quality and resolution hints
        - Scene description

        4. CATEGORIZATION:
        - Type: anime, game art, fan art, original art, etc.
        - Genre or thematic elements
        - Artistic style tags

        Based on your analysis, generate a short, descriptive **filename** (without extension) that captures the essence of the image.

        Requirements:
        - Use lowercase letters, numbers, and hyphens
        - Be descriptive but concise
        - make sure the name is within 20 characters

        Just respond with the filename only, no explanation or formatting.
        """
        
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
        filename = response.text.strip().lower()
        filename = re.sub(r'[^a-z0-9\-]', '', filename)
        
        if not filename:
            filename = "image"
        
        return filename

    except Exception as e:
        error_msg = str(e).lower()
        print(f"Error generating filename (attempt {retry_count + 1}): {str(e)}")
        
        # Always retry on any error
        if any(keyword in error_msg for keyword in ["rate limit", "quota", "429", "resource has been exhausted"]):
            switch_to_next_model()
        
        print(f"Retrying filename generation in {RETRY_DELAY} seconds...")
        time.sleep(RETRY_DELAY)
        return generate_filename(image_path, retry_count + 1)

def identify_image(image_path, retry_count=0):
    """Identify image content and categorize it."""
    # Rotate API key if needed
    rotate_api_key()
    
    try:
        categories = [
            "#nature", "#anime", "#art", "#abstract", "#cars",
            "#architecture", "#minimal", "#tech", "#amoled"
        ]
        
        with Image.open(image_path) as img:
            max_size = 800
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
        
        current_model = get_current_model()
        model = genai.GenerativeModel(current_model)
        
        prompt = f"""
        Analyze this image and categorize it into exactly one of these categories:
        {', '.join(categories)}
        
        Note: #amoled is for images with pure black backgrounds and vibrant colors, perfect for AMOLED displays.
        
        Just respond with the category name only, including the # symbol. No explanation.
        """
        
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
        category = response.text.strip()
        
        if category not in categories:
            for valid_category in categories:
                if valid_category.lower() in category.lower():
                    category = valid_category
                    break
            else:
                category = "#art"
        
        return category

    except Exception as e:
        error_msg = str(e).lower()
        print(f"Error identifying image (attempt {retry_count + 1}): {str(e)}")
        
        # Always retry on any error
        if any(keyword in error_msg for keyword in ["rate limit", "quota", "429", "resource has been exhausted"]):
            switch_to_next_model()
        
        print(f"Retrying category identification in {RETRY_DELAY} seconds...")
        time.sleep(RETRY_DELAY)
        return identify_image(image_path, retry_count + 1)

def generate_image_data(image_path, retry_count=0):
    """Generate detailed image analysis data using Gemini with retry logic."""
    # Rotate API key if needed
    rotate_api_key()
    
    try:
        with Image.open(image_path) as img:
            max_size = 800
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            image_bytes = buffer.getvalue()
        
        current_model = get_current_model()
        model = genai.GenerativeModel(current_model)
        
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
        
        Return ONLY the JSON object, no explanations.
        """
        
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
        
        try:
            response = model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": image_bytes}],
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        except Exception as api_error:
            error_msg = str(api_error).lower()
            print(f"API Error (attempt {retry_count + 1}): {str(api_error)[:100]}")
            
            # Always retry on any error
            if any(keyword in error_msg for keyword in ["rate limit", "quota", "429", "resource has been exhausted"]):
                switch_to_next_model()
            
            print(f"Retrying data generation in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            return generate_image_data(image_path, retry_count + 1)
        
        if not hasattr(response, 'text') or not response.text or not response.text.strip():
            print(f"Empty response (attempt {retry_count + 1}), retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            return generate_image_data(image_path, retry_count + 1)
        
        try:
            response_text = response.text.strip()
            response_text = clean_json_response(response_text)
            data = json.loads(response_text)
            cleaned_data = validate_and_clean_data(data)
            return cleaned_data
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing failed (attempt {retry_count + 1}): {str(json_err)[:100]}")
            print(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            return generate_image_data(image_path, retry_count + 1)

    except Exception as e:
        error_msg = str(e).lower()
        print(f"Unexpected error (attempt {retry_count + 1}): {str(e)[:100]}")
        
        # Always retry on any error
        if any(keyword in error_msg.lower() for keyword in ["rate limit", "quota", "429"]):
            switch_to_next_model()
        
        print(f"Retrying in {RETRY_DELAY} seconds...")
        time.sleep(RETRY_DELAY)
        return generate_image_data(image_path, retry_count + 1)

def clean_json_response(response_text):
    """Clean up JSON response text."""
    if response_text.startswith('```json'):
        response_text = response_text[7:]
    if response_text.startswith('```'):
        response_text = response_text[3:]
    if response_text.endswith('```'):
        response_text = response_text[:-3]
    
    json_start = response_text.find('{')
    if json_start > 0:
        response_text = response_text[json_start:]
    
    json_end = response_text.rfind('}')
    if json_end > 0:
        response_text = response_text[:json_end + 1]
    
    return response_text.strip()

def validate_and_clean_data(data):
    """Validate and clean the parsed JSON data."""
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
    
    for key in ["character_names", "primary_colors", "secondary_colors", "style_influences", "objects", "textures", "tags"]:
        cleaned_data[key] = [str(item).strip() for item in cleaned_data[key] if item and str(item).strip()]
    
    if not cleaned_data["tags"]:
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
    """Generate fallback data when AI analysis fails."""
    filename = os.path.basename(image_path).lower()
    base_name = os.path.splitext(filename)[0]
    
    fallback_tags = []
    words = re.findall(r'[a-zA-Z]+', base_name)
    meaningful_words = [word for word in words if len(word) > 2]
    fallback_tags.extend(meaningful_words[:5])
    
    return {
        "character_names": [],
        "series": "unknown",
        "art_style": "unknown",
        "primary_colors": [],
        "secondary_colors": [],
        "color_palette": "unknown",
        "mood": "unknown",
        "technique": "unknown",
        "scene_description": "unknown",
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
        "style_influences": [],
        "objects": [],
        "textures": [],
        "type": "unknown",
        "tags": fallback_tags
    }

def get_resolution_info(image_path):
    """Determine resolution category and orientation."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            orientation = "Mobile" if height > width else "Desktop"
            
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
        return {"width": 0, "height": 0, "resolution": "Unknown", "orientation": "Unknown"}

def get_file_timestamp(file_path):
    """Get file timestamp in ISO format."""
    try:
        mod_time = os.path.getmtime(file_path)
        dt_object = datetime.fromtimestamp(mod_time, tz=timezone.utc)
        return dt_object.isoformat()
    except Exception as e:
        print(f"Error getting timestamp: {str(e)}")
        return ""

def get_file_without_extension(filename):
    """Get filename without extension."""
    return os.path.splitext(filename)[0]

def compress_and_process_image(source_path, filename):
    """Compress image and process with AI."""
    try:
        # Generate AI filename
        new_base_name = generate_filename(source_path)
        file_extension = os.path.splitext(filename)[1]
        new_filename = f"{new_base_name}{file_extension}"
        
        # Ensure unique filename
        counter = 1
        while os.path.exists(os.path.join(main_dir, new_filename)):
            new_filename = f"{new_base_name}-{counter}{file_extension}"
            counter += 1
        
        # Copy to main directory
        main_path = os.path.join(main_dir, new_filename)
        shutil.copy2(source_path, main_path)
        print(f"Copied: {filename} -> {new_filename}")
        
        # Create compressed WebP version
        destination_path = os.path.join(destination_dir, os.path.splitext(new_filename)[0] + ".webp")
        
        with Image.open(source_path) as img:
            width, height = img.size
            
            # Resize if too large
            max_dimension = 1920
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = math.floor(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = math.floor(width * (max_dimension / height))
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Determine quality based on size
            original_size = os.path.getsize(source_path) / 1024
            if original_size > 1000:
                quality = 10
            elif original_size > 500:
                quality = 20
            else:
                quality = 30
            
            img.save(destination_path, "WEBP", quality=quality, method=6)
        
        # Mark as processed
        mark_file_processed(filename, new_filename, "rename")
        
        return new_filename, main_path
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        # Fallback: use original filename
        new_filename = filename
        main_path = os.path.join(main_dir, new_filename)
        shutil.copy2(source_path, main_path)
        return new_filename, main_path

async def process_images_batch(images: List[str], message: Message):
    """Process a batch of images and update index."""
    await message.answer(f"🔄 Processing {len(images)} images...")
    
    # Load existing index
    existing_entries = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for entry in existing_data:
                    if 'file_name' in entry:
                        existing_entries[entry['file_name']] = entry
        except Exception as e:
            print(f"Error loading index: {e}")
            existing_data = []
    else:
        existing_data = []
    
    processed_count = 0
    errors_count = 0
    
    # Process images in batches
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        await message.answer(f"📦 Processing batch {batch_num}/{math.ceil(len(images)/BATCH_SIZE)}...")
        
        for j, image_path in enumerate(batch):
            try:
                filename = os.path.basename(image_path)
                await message.answer(f"⚙️ [{i+j+1}/{len(images)}] Processing: {filename[:30]}...")
                
                # Compress and copy
                new_filename, main_path = compress_and_process_image(image_path, filename)
                
                # Get resolution info
                resolution_info = get_resolution_info(main_path)
                
                # Identify category
                category = identify_image(main_path)
                
                # Generate detailed data
                data = generate_image_data(main_path)
                
                # Create index entry
                base_name = get_file_without_extension(new_filename)
                entry = {
                    "file_name": base_name,
                    "file_cache_name": f"{base_name}.webp",
                    "file_main_name": new_filename,
                    "width": resolution_info["width"],
                    "height": resolution_info["height"],
                    "resolution": resolution_info["resolution"],
                    "orientation": resolution_info["orientation"],
                    "timestamp": get_file_timestamp(main_path),
                    "category": category,
                    "data": data
                }
                
                # Update or add entry
                existing_entries[base_name] = entry
                processed_count += 1
                
                await message.answer(
                    f"✅ {new_filename[:30]}\n"
                    f"Category: {category}\n"
                    f"Tags: {len(data.get('tags', []))}\n"
                    f"Resolution: {resolution_info['resolution']}"
                )
                
            except Exception as e:
                errors_count += 1
                await message.answer(f"❌ Error processing {filename}: {str(e)[:100]}")
                print(f"Error: {e}")
        
        # Save index after each batch
        try:
            result = list(existing_entries.values())
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4)
            await message.answer(f"💾 Batch {batch_num} saved to index")
        except Exception as e:
            await message.answer(f"⚠️ Error saving index: {str(e)[:100]}")
        
        # Wait between batches
        if i + BATCH_SIZE < len(images):
            await message.answer(f"⏳ Waiting {BATCH_WAIT_TIME}s before next batch...")
            await asyncio.sleep(BATCH_WAIT_TIME)
    
    # Clean up temp files
    for image_path in images:
        try:
            os.remove(image_path)
        except:
            pass
    
    await message.answer(
        f"✨ Processing complete!\n"
        f"✅ Processed: {processed_count}\n"
        f"❌ Errors: {errors_count}\n"
        f"📊 Total in index: {len(existing_entries)}"
    )
    
    # Git operations
    await message.answer("🔄 Starting git operations...")
    
    try:
        # Git add --all
        await message.answer("📝 Running: git add --all")
        import subprocess
        result = subprocess.run(
            ["git", "add", "--all"],
            cwd=r"D:\storage",
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            await message.answer("✅ Git add successful")
        else:
            await message.answer(f"⚠️ Git add warning: {result.stderr[:200]}")
        
        # Git commit
        await message.answer('📝 Running: git commit -m "new batch"')
        result = subprocess.run(
            ["git", "commit", "-m", "new batch"],
            cwd=r"D:\storage",
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            await message.answer(f"✅ Git commit successful\n{result.stdout[:200]}")
        else:
            # Check if it's just "nothing to commit"
            if "nothing to commit" in result.stdout.lower() or "nothing to commit" in result.stderr.lower():
                await message.answer("ℹ️ Nothing to commit (working tree clean)")
            else:
                await message.answer(f"⚠️ Git commit warning: {result.stderr[:200]}")
        
        # Git push
        await message.answer("📝 Running: git push")
        result = subprocess.run(
            ["git", "push"],
            cwd=r"D:\storage",
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            await message.answer(f"✅ Git push successful!\n{result.stdout[:200]}")
        else:
            await message.answer(f"❌ Git push failed: {result.stderr[:200]}")
        
        await message.answer("🎉 All git operations complete!")
        
    except subprocess.TimeoutExpired:
        await message.answer("⏰ Git operation timed out")
    except FileNotFoundError:
        await message.answer("❌ Git not found. Make sure git is installed and in PATH")
    except Exception as e:
        await message.answer(f"❌ Git error: {str(e)[:200]}")

# ==================== BOT HANDLERS ====================

def check_authorized(message: Message) -> bool:
    """Check if user is authorized."""
    return message.from_user.id == AUTHORIZED_USER_ID

@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Handle /start command."""
    if not check_authorized(message):
        await message.answer("❌ You are not authorized to use this bot.")
        return
    
    await message.answer(
        "🤖 Welcome to Image Processing Bot!\n\n"
        "Commands:\n"
        "/begin - Start collecting images\n"
        "/end - Process collected images\n"
        "/cancel - Cancel current operation\n"
        "/status - Check bot status"
    )

@dp.message(Command("begin"))
async def cmd_begin(message: Message, state: FSMContext):
    """Start collecting images."""
    if not check_authorized(message):
        await message.answer("❌ You are not authorized to use this bot.")
        return
    
    user_id = message.from_user.id
    user_images[user_id] = []
    
    await state.set_state(OrderStates.waiting_for_images)
    await message.answer(
        "📸 Image collection started!\n\n"
        "Send me images as **files** (not compressed).\n"
        "When done, send /end to process them."
    )

@dp.message(Command("end"))
async def cmd_end(message: Message, state: FSMContext):
    """End collection and process images."""
    if not check_authorized(message):
        await message.answer("❌ You are not authorized to use this bot.")
        return
    
    user_id = message.from_user.id
    
    if user_id not in user_images or not user_images[user_id]:
        await message.answer("❌ No images collected. Use /begin first.")
        return
    
    images = user_images[user_id].copy()
    user_images[user_id] = []
    await state.clear()
    
    await message.answer(f"🎯 Processing {len(images)} images...")
    
    # Process images
    await process_images_batch(images, message)

@dp.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext):
    """Cancel current operation."""
    if not check_authorized(message):
        await message.answer("❌ You are not authorized to use this bot.")
        return
    
    user_id = message.from_user.id
    
    if user_id in user_images:
        # Clean up temp files
        for img_path in user_images[user_id]:
            try:
                os.remove(img_path)
            except:
                pass
        user_images[user_id] = []
    
    await state.clear()
    await message.answer("❌ Operation cancelled.")

@dp.message(Command("status"))
async def cmd_status(message: Message):
    """Check bot status."""
    if not check_authorized(message):
        await message.answer("❌ You are not authorized to use this bot.")
        return
    
    user_id = message.from_user.id
    collected = len(user_images.get(user_id, []))
    
    # Count files in directories
    main_count = len([f for f in os.listdir(main_dir) if os.path.isfile(os.path.join(main_dir, f))])
    cache_count = len([f for f in os.listdir(destination_dir) if os.path.isfile(os.path.join(destination_dir, f))])
    
    # Count index entries
    index_count = 0
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                index_count = len(json.load(f))
        except:
            pass
    
    await message.answer(
        f"📊 Bot Status\n\n"
        f"Images collected: {collected}\n"
        f"Main directory: {main_count} files\n"
        f"Cache directory: {cache_count} files\n"
        f"Index entries: {index_count}\n"
        f"Current model: {get_current_model()}"
    )

@dp.message(OrderStates.waiting_for_images, F.document)
async def handle_document(message: Message, state: FSMContext):
    """Handle image files sent as documents."""
    if not check_authorized(message):
        return
    
    user_id = message.from_user.id
    
    if user_id not in user_images:
        user_images[user_id] = []
    
    # Check if it's an image
    if message.document.mime_type and message.document.mime_type.startswith('image/'):
        file_id = message.document.file_id
        file_name = message.document.file_name
        file_size_mb = message.document.file_size / (1024 * 1024) if message.document.file_size else 0
        
        # Download file with retry logic
        max_download_retries = 5
        for attempt in range(max_download_retries):
            try:
                await message.answer(f"⬇️ Downloading: {file_name} ({file_size_mb:.1f}MB) - attempt {attempt + 1}")
                
                file = await bot.get_file(file_id)
                file_extension = os.path.splitext(file_name)[1] or '.jpg'
                temp_path = os.path.join(temp_download_dir, f"{file_id}{file_extension}")
                
                await bot.download_file(file.file_path, temp_path)
                
                user_images[user_id].append(temp_path)
                await message.answer(f"✅ Image {len(user_images[user_id])} received: {file_name}")
                break
                
            except asyncio.TimeoutError:
                if attempt < max_download_retries - 1:
                    wait_time = (attempt + 1) * 10  # Progressive delay: 10s, 20s, 30s, etc.
                    await message.answer(f"⏰ Download timeout. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    await message.answer(f"❌ Failed to download {file_name} after {max_download_retries} attempts. Please try sending it again.")
            except Exception as e:
                if attempt < max_download_retries - 1:
                    await message.answer(f"⚠️ Error: {str(e)[:100]}. Retrying...")
                    await asyncio.sleep(5)
                else:
                    await message.answer(f"❌ Failed to download {file_name}: {str(e)[:100]}")
    else:
        await message.answer("⚠️ Please send image files only (PNG, JPG, JPEG)")

@dp.message(OrderStates.waiting_for_images)
async def handle_other_messages(message: Message):
    """Handle other messages during collection."""
    if not check_authorized(message):
        return
    
    await message.answer(
        "📸 Send images as files or use:\n"
        "/end - Process images\n"
        "/cancel - Cancel operation"
    )

@dp.message()
async def handle_unauthorized(message: Message):
    """Handle messages from unauthorized users."""
    if not check_authorized(message):
        await message.answer("❌ You are not authorized to use this bot.")

# ==================== MAIN ====================

async def main():
    """Main function to run the bot."""
    print("🤖 Initializing bot...")
    
    # Initialize database
    init_database()
    
    print(f"✅ Authorized user ID: {AUTHORIZED_USER_ID}")
    print(f"✅ Directories ready:")
    print(f"   - Main: {main_dir}")
    print(f"   - Cache: {destination_dir}")
    print(f"   - Temp: {temp_download_dir}")
    print(f"   - Index: {output_file}")
    print(f"✅ Using model: {get_current_model()}")
    print("🚀 Bot starting...")
    
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
