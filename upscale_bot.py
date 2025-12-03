import os
import asyncio
import subprocess
from io import BytesIO
from datetime import datetime

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile, BufferedInputFile
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.session.aiohttp import AiohttpSession

from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys from environment
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AUTHORIZED_USER_ID = int(os.getenv("AUTHORIZED_USER_ID", "0"))

if not BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")
if AUTHORIZED_USER_ID == 0:
    raise ValueError("AUTHORIZED_USER_ID not found in .env file")

# Temp directory
temp_dir = r"D:\storage\temp_downloads"
os.makedirs(temp_dir, exist_ok=True)

# Initialize bot and dispatcher
session = AiohttpSession()
bot = Bot(token=BOT_TOKEN, session=session)
bot.session.timeout = 300  # 5 minutes timeout

storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# ==================== HELPER FUNCTIONS ====================

def check_authorized(message: Message) -> bool:
    """Check if user is authorized."""
    return message.from_user.id == AUTHORIZED_USER_ID

async def download_image_with_retry(message: Message, file_id: str, file_name: str, max_retries: int = 3) -> str:
    """Download image from Telegram with retry logic."""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                await message.answer(f"⏳ Download attempt {attempt + 1}/{max_retries}...")
            
            file = await bot.get_file(file_id)
            file_extension = os.path.splitext(file_name)[1] or '.jpg'
            temp_path = os.path.join(temp_dir, f"{file_id}{file_extension}")
            
            await bot.download_file(file.file_path, temp_path)
            return temp_path
            
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                await message.answer(f"⏰ Download timeout. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise Exception(f"Failed to download after {max_retries} attempts")
        except Exception as e:
            if attempt < max_retries - 1:
                await message.answer(f"⚠️ Error: {str(e)[:100]}. Retrying...")
                await asyncio.sleep(5)
            else:
                raise

async def upscale_image_with_realesrgan(image_path: str, message: Message, scale: int = 4) -> str:
    """Use Real-ESRGAN to upscale the image."""
    try:
        await message.answer(f"🔧 Upscaling image with Real-ESRGAN ({scale}x)...")
        
        # Output path
        output_path = image_path.replace(os.path.splitext(image_path)[1], f"_upscaled{scale}x.png")
        
        # Try to use realesrgan-ncnn-vulkan (faster, GPU-accelerated)
        realesrgan_cmd = [
            "realesrgan-ncnn-vulkan",
            "-i", image_path,
            "-o", output_path,
            "-s", str(scale),
            "-n", "realesrgan-x4plus"  # Model name
        ]
        
        try:
            # Try GPU version first
            result = subprocess.run(
                realesrgan_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                raise Exception("GPU version failed")
                
        except (FileNotFoundError, Exception):
            # Fallback to Python version
            await message.answer("⚙️ Using Python Real-ESRGAN (slower, CPU)...")
            
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                
                # Select model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                
                # Create upsampler
                upsampler = RealESRGANer(
                    scale=4,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False  # Use FP32 for CPU
                )
                
                # Read image
                img = Image.open(image_path).convert('RGB')
                import numpy as np
                img_array = np.array(img)
                
                # Upscale
                output, _ = upsampler.enhance(img_array, outscale=scale)
                
                # Save
                output_img = Image.fromarray(output)
                output_img.save(output_path, 'PNG')
                
            except ImportError:
                raise Exception("Real-ESRGAN not installed. Run: pip install realesrgan")
        
        if not os.path.exists(output_path):
            raise Exception("Upscaling failed - output file not created")
        
        await message.answer(f"✅ Upscaling complete! ({scale}x)")
        return output_path
        
    except subprocess.TimeoutExpired:
        raise Exception("Upscaling timeout (max 5 minutes)")
    except Exception as e:
        raise Exception(f"Real-ESRGAN error: {str(e)}")

# ==================== BOT HANDLERS ====================

@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Handle /start command."""
    if not check_authorized(message):
        await message.answer("❌ You are not authorized to use this bot.")
        return
    
    await message.answer(
        "🤖 Welcome to Image Upscale Bot!\n\n"
        "Send me an image (as photo or file) and I'll upscale it 4x using Real-ESRGAN!\n\n"
        "Features:\n"
        "✨ 4x upscaling\n"
        "🎨 Enhanced details and sharpness\n"
        "🚀 GPU acceleration (if available)\n\n"
        "Commands:\n"
        "/start - Show this message\n"
        "/help - Show help"
    )

@dp.message(Command("help"))
async def cmd_help(message: Message):
    """Handle /help command."""
    if not check_authorized(message):
        await message.answer("❌ You are not authorized to use this bot.")
        return
    
    await message.answer(
        "📖 How to use:\n\n"
        "1. Send any image (photo or file)\n"
        "2. Bot will upscale it 4x using Real-ESRGAN\n"
        "3. You'll receive the upscaled image\n\n"
        "💡 Tips:\n"
        "- Send as file (not compressed photo) for best quality\n"
        "- Processing takes 30s-3min depending on image size\n"
        "- GPU acceleration is used if available\n"
        "- Maximum file size: 20MB (Telegram limit)"
    )

@dp.message(F.photo)
async def handle_photo(message: Message):
    """Handle photos sent as compressed images."""
    if not check_authorized(message):
        await message.answer("❌ You are not authorized to use this bot.")
        return
    
    try:
        await message.answer("📥 Receiving image...")
        
        # Get the largest photo
        photo = message.photo[-1]
        file_name = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        # Download image
        temp_path = await download_image_with_retry(message, photo.file_id, file_name)
        
        await message.answer("✅ Image received!")
        
        # Process with Real-ESRGAN
        try:
            result_path = await upscale_image_with_realesrgan(temp_path, message, scale=4)
            
            # Send back the result
            await message.answer("📤 Sending upscaled image...")
            result_file = FSInputFile(result_path, filename=f"upscaled_{file_name}")
            await message.answer_document(result_file, caption="✨ Upscaled 4x with Real-ESRGAN")
            
            # Clean up result
            try:
                os.remove(result_path)
            except:
                pass
                
        except Exception as e:
            await message.answer(f"❌ Processing error: {str(e)[:200]}")
        
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass
            
    except Exception as e:
        await message.answer(f"❌ Error: {str(e)[:200]}")

@dp.message(F.document)
async def handle_document(message: Message):
    """Handle images sent as files."""
    if not check_authorized(message):
        await message.answer("❌ You are not authorized to use this bot.")
        return
    
    # Check if it's an image
    if not message.document.mime_type or not message.document.mime_type.startswith('image/'):
        await message.answer("⚠️ Please send image files only (PNG, JPG, JPEG)")
        return
    
    try:
        file_name = message.document.file_name
        file_size_mb = message.document.file_size / (1024 * 1024) if message.document.file_size else 0
        
        await message.answer(f"📥 Receiving: {file_name} ({file_size_mb:.1f}MB)...")
        
        # Download image
        temp_path = await download_image_with_retry(message, message.document.file_id, file_name)
        
        await message.answer("✅ Image received!")
        
        # Process with Real-ESRGAN
        try:
            result_path = await upscale_image_with_realesrgan(temp_path, message, scale=4)
            
            # Send back the result
            await message.answer("📤 Sending upscaled image...")
            result_file = FSInputFile(result_path, filename=f"upscaled_{file_name}")
            await message.answer_document(result_file, caption="✨ Upscaled 4x with Real-ESRGAN")
            
            # Clean up result
            try:
                os.remove(result_path)
            except:
                pass
                
        except Exception as e:
            await message.answer(f"❌ Processing error: {str(e)[:200]}")
        
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass
            
    except Exception as e:
        await message.answer(f"❌ Error: {str(e)[:200]}")

@dp.message()
async def handle_other(message: Message):
    """Handle other messages."""
    if not check_authorized(message):
        await message.answer("❌ You are not authorized to use this bot.")
        return
    
    await message.answer("📸 Please send an image (photo or file)")

# ==================== MAIN ====================

async def main():
    """Main function to run the bot."""
    print("🤖 Initializing Real-ESRGAN Upscale Bot...")
    print(f"✅ Authorized user ID: {AUTHORIZED_USER_ID}")
    print(f"✅ Temp directory: {temp_dir}")
    print("\n📦 Checking Real-ESRGAN installation...")
    print("   If not installed, run: pip install realesrgan")
    print("   For GPU support: pip install realesrgan-ncnn-vulkan")
    print("\n🚀 Bot starting...")
    
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
