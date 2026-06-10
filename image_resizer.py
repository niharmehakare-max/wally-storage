import os
import io
import concurrent.futures
from PIL import Image

def process_image(file_path, max_size_bytes):
    try:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename} ({os.path.getsize(file_path) / (1024*1024):.2f} MB)")
        
        with Image.open(file_path) as img:
            orig_format = img.format if img.format else "PNG"
            
            # Use in-memory buffer to check file size without writing to disk
            buffer = io.BytesIO()
            
            # Set format-specific optimization options
            save_kwargs = {"optimize": True}
            if orig_format.upper() == "PNG":
                save_kwargs["compress_level"] = 9
            elif orig_format.upper() in ("JPEG", "JPG"):
                save_kwargs["quality"] = 90
            
            img.save(buffer, format=orig_format, **save_kwargs)
            
            # Reduce dimensions if still too large
            if buffer.tell() > max_size_bytes:
                scale = 0.9
                while buffer.tell() > max_size_bytes and scale > 0.1:
                    new_width = int(img.width * scale)
                    new_height = int(img.height * scale)
                    
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    buffer.seek(0)
                    buffer.truncate(0)
                    resized_img.save(buffer, format=orig_format, **save_kwargs)
                    scale -= 0.1

            # Finally, write the successful buffer to the original file
            with open(file_path, 'wb') as f:
                f.write(buffer.getvalue())
                
        new_size_mb = os.path.getsize(file_path) / (1024*1024)
        print(f"Successfully optimized {filename} to {new_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")

def shrink_images(folder_path, max_size_mb=15):
    max_size_bytes = max_size_mb * 1024 * 1024
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    if not os.path.exists(folder_path):
        print(f"Error: The folder {folder_path} does not exist.")
        return

    # Collect files that need processing
    files_to_process = []
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(valid_extensions):
            continue
            
        file_path = os.path.join(folder_path, filename)
        if os.path.getsize(file_path) > max_size_bytes:
            files_to_process.append(file_path)

    if not files_to_process:
        print("No images found exceeding the size limit.")
        return

    # Process images in parallel using multiprocessing
    # Max workers sets the maximum number of processes to the number of processors on the machine
    print(f"Found {len(files_to_process)} images to process. Starting parallel compression...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_image, path, max_size_bytes): path for path in files_to_process}
        for future in concurrent.futures.as_completed(futures):
            # We just wait for completion here, exceptions are handled inside process_image
            pass
    
    print("Optimization complete!")

if __name__ == "__main__":
    target_folder = r"main"
    shrink_images(target_folder)