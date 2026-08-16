import os
import json
import time
import urllib.parse
import http.server
import socketserver
import hashlib
import sys
import webbrowser
from PIL import Image
from paths import CACHE_DIR, DUPLICATE_HASH_CACHE, INDEX_FILE, MAIN_DIR, PROJECT_DIR

# Configuration
PORT = 8000
cache_dir = CACHE_DIR
main_dir = MAIN_DIR
index_path = INDEX_FILE
hash_cache_path = DUPLICATE_HASH_CACHE

def get_image_hash(filepath):
    """Compute combined dHash and aHash of the image for visual similarity comparison."""
    try:
        with Image.open(filepath) as img:
            # Convert to grayscale
            img_gray = img.convert('L')
            
            # 1. dHash (Difference Hash) - 9x8 image
            img_dhash = img_gray.resize((9, 8), Image.Resampling.BILINEAR)
            pixels = list(img_dhash.getdata())
            diff = []
            for row in range(8):
                for col in range(8):
                    diff.append(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
            dval = sum(int(bit) << i for i, bit in enumerate(diff))
            dhash_str = f"{dval:016x}"
            
            # 2. aHash (Average Hash) - 8x8 image
            img_ahash = img_gray.resize((8, 8), Image.Resampling.BILINEAR)
            apixels = list(img_ahash.getdata())
            avg = sum(apixels) / 64
            abits = [p >= avg for p in apixels]
            aval = sum(int(bit) << i for i, bit in enumerate(abits))
            ahash_str = f"{aval:016x}"
            
            return f"{dhash_str}-{ahash_str}"
    except Exception as e:
        # Fallback to MD5 if PIL fails (e.g. format issues)
        try:
            with open(filepath, 'rb') as f:
                return f"md5-{hashlib.md5(f.read()).hexdigest()}"
        except:
            return None

def scan_for_duplicates():
    """Scan all images listed in index.json, compute hashes, and group duplicates."""
    if not os.path.exists(index_path):
        return {"success": False, "message": "index.json not found.", "groups": []}
        
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
    except Exception as e:
        return {"success": False, "message": f"Failed to load index.json: {e}", "groups": []}

    # Load hash cache
    hash_cache = {}
    if os.path.exists(hash_cache_path):
        try:
            with open(hash_cache_path, 'r', encoding='utf-8') as f:
                hash_cache = json.load(f)
        except:
            pass

    hashes_map = {}
    scanned_count = 0
    cache_updated = False
    
    for entry in index_data:
        file_name = entry.get("file_name")
        cache_file = entry.get("file_cache_name")
        main_file = entry.get("file_main_name")
        
        # Determine image file to hash (prefer cache because webp is smaller/faster)
        target_path = None
        if cache_file:
            path = os.path.join(cache_dir, cache_file)
            if os.path.exists(path):
                target_path = path
        if not target_path and main_file:
            path = os.path.join(main_dir, main_file)
            if os.path.exists(path):
                target_path = path
                
        if not target_path:
            continue
            
        mtime = os.path.getmtime(target_path)
        
        # Check cache
        cache_entry = hash_cache.get(file_name)
        img_hash = None
        if cache_entry and cache_entry.get("mtime") == mtime:
            img_hash = cache_entry.get("hash")
        else:
            img_hash = get_image_hash(target_path)
            if img_hash:
                hash_cache[file_name] = {
                    "hash": img_hash,
                    "mtime": mtime
                }
                cache_updated = True
                
        if img_hash:
            scanned_count += 1
            if img_hash not in hashes_map:
                hashes_map[img_hash] = []
            hashes_map[img_hash].append(entry)

    # Save cache if updated
    if cache_updated:
        try:
            with open(hash_cache_path, 'w', encoding='utf-8') as f:
                json.dump(hash_cache, f, indent=4)
        except:
            pass

    duplicate_groups = []
    total_duplicate_count = 0
    reclaimable_space = 0
    
    for img_hash, group in hashes_map.items():
        if len(group) > 1:
            # Sort group by timestamp (newest first, based on ISO timestamp string)
            # Example timestamp: "2025-04-06T13:41:23.674408+00:00"
            group.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
            
            enriched_items = []
            for idx, entry in enumerate(group):
                f_name = entry.get("file_name")
                c_name = entry.get("file_cache_name")
                m_name = entry.get("file_main_name")
                
                # Fetch sizes
                main_size = 0
                cache_size = 0
                if m_name:
                    m_path = os.path.join(main_dir, m_name)
                    if os.path.exists(m_path):
                        main_size = os.path.getsize(m_path)
                if c_name:
                    c_path = os.path.join(cache_dir, c_name)
                    if os.path.exists(c_path):
                        cache_size = os.path.getsize(c_path)
                
                total_size = main_size + cache_size
                
                item = {
                    "file_name": f_name,
                    "file_cache_name": c_name,
                    "file_main_name": m_name,
                    "width": entry.get("width"),
                    "height": entry.get("height"),
                    "resolution": entry.get("resolution"),
                    "timestamp": entry.get("timestamp"),
                    "category": entry.get("category"),
                    "main_size_bytes": main_size,
                    "cache_size_bytes": cache_size,
                    "total_size_bytes": total_size,
                    "is_keep": idx == 0  # Keep the newest one
                }
                
                if idx > 0:
                    reclaimable_space += total_size
                    total_duplicate_count += 1
                    
                enriched_items.append(item)
                
            duplicate_groups.append({
                "hash": img_hash,
                "items": enriched_items
            })

    # Sort groups so the ones with the most potential saved bytes appear first
    duplicate_groups.sort(key=lambda g: sum(item["total_size_bytes"] for item in g["items"] if not item["is_keep"]), reverse=True)

    return {
        "success": True,
        "total_scanned": scanned_count,
        "total_duplicates": total_duplicate_count,
        "reclaimable_space_bytes": reclaimable_space,
        "groups": duplicate_groups
    }

def delete_wallpapers(file_names):
    """Delete files from disk (main + cache) and remove from index.json entirely."""
    if not os.path.exists(index_path):
        return False, "index.json not found"
        
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
    except Exception as e:
        return False, f"Failed to load index.json: {e}"

    entries_map = {e.get("file_name"): e for e in index_data}
    
    deleted_count = 0
    errors = []
    
    for file_name in file_names:
        entry = entries_map.get(file_name)
        if not entry:
            continue
            
        c_name = entry.get("file_cache_name")
        m_name = entry.get("file_main_name")
        
        if c_name:
            c_path = os.path.join(cache_dir, c_name)
            if os.path.exists(c_path):
                try:
                    os.remove(c_path)
                except Exception as e:
                    errors.append(f"Cache file delete failed ({c_name}): {e}")
                    
        if m_name:
            m_path = os.path.join(main_dir, m_name)
            if os.path.exists(m_path):
                try:
                    os.remove(m_path)
                except Exception as e:
                    errors.append(f"Main file delete failed ({m_name}): {e}")
                    
        deleted_count += 1

    # Remove entries from index.json
    index_data = [e for e in index_data if e.get("file_name") not in file_names]
    
    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=4)
    except Exception as e:
        errors.append(f"Failed to update index.json: {e}")

    # Remove from hash cache if applicable
    if os.path.exists(hash_cache_path):
        try:
            with open(hash_cache_path, 'r', encoding='utf-8') as f:
                h_cache = json.load(f)
            
            for file_name in file_names:
                if file_name in h_cache:
                    del h_cache[file_name]
                    
            with open(hash_cache_path, 'w', encoding='utf-8') as f:
                json.dump(h_cache, f, indent=4)
        except:
            pass

    if errors:
        return False, f"Deleted {deleted_count} entries with errors: {'; '.join(errors)}"
    return True, f"Successfully deleted {deleted_count} duplicate wallpaper entries"


class DuplicateDetectorHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Print logs to console cleanly
        print(f"[HTTP] {format % args}")

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
            return
            
        elif path == "/api/scan":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            results = scan_for_duplicates()
            self.wfile.write(json.dumps(results).encode('utf-8'))
            return
            
        elif path.startswith("/main/"):
            filename = urllib.parse.unquote(os.path.basename(path))
            filepath = os.path.join(main_dir, filename)
            self.serve_static_file(filepath, "image/png")
            return
            
        elif path.startswith("/cache/"):
            filename = urllib.parse.unquote(os.path.basename(path))
            filepath = os.path.join(cache_dir, filename)
            self.serve_static_file(filepath, "image/webp")
            return
            
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        if path == "/api/delete":
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error(400, "Bad Request: Empty Body")
                return
                
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode('utf-8'))
                file_names = data.get("file_names", [])
            except Exception as e:
                self.send_error(400, f"Bad Request: Invalid JSON: {e}")
                return
                
            if not file_names:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "message": "No file_names specified"}).encode('utf-8'))
                return
                
            success, message = delete_wallpapers(file_names)
            self.send_response(200 if success else 500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": success, "message": message}).encode('utf-8'))
            return
            
        else:
            self.send_error(404, "Not Found")

    def serve_static_file(self, filepath, content_type):
        if not os.path.exists(filepath):
            self.send_error(404, "File Not Found")
            return
            
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(os.path.getsize(filepath)))
            self.end_headers()
            with open(filepath, 'rb') as f:
                self.wfile.write(f.read())
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {e}")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wallpaper Duplicate Finder</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0813;
            --surface-color: rgba(26, 21, 44, 0.45);
            --surface-hover: rgba(40, 32, 66, 0.6);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f0fa;
            --text-secondary: rgba(243, 240, 250, 0.6);
            --accent-primary: #9b5de5;
            --accent-glow: rgba(155, 93, 229, 0.4);
            --success-color: #00f5d4;
            --warning-color: #f15bb5;
            --danger-color: #ff5c5c;
            --keep-color: #2ec4b6;
            --delete-color: #e71d36;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(155, 93, 229, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(0, 245, 212, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(26, 21, 44, 0.3) 0%, #0b0813 100%);
            background-attachment: fixed;
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            line-height: 1.5;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(11, 8, 19, 0.8);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(155, 93, 229, 0.3);
            border-radius: 5px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(155, 93, 229, 0.6);
        }

        /* Glassmorphism Container */
        .glass-panel {
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        header {
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 20px 20px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }

        .brand h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #f3f0fa 30%, var(--accent-primary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -1px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .brand p {
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-top: 4px;
        }

        .actions-bar {
            display: flex;
            gap: 12px;
        }

        .btn {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 0.95rem;
            padding: 12px 24px;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--accent-primary) 0%, #7b2cbf 100%);
            color: white;
            box-shadow: 0 4px 15px var(--accent-glow);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(155, 93, 229, 0.6);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateY(-2px);
        }

        .btn-danger {
            background: linear-gradient(135deg, var(--delete-color) 0%, #b00f22 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(231, 29, 54, 0.3);
        }

        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(231, 29, 54, 0.5);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
            box-shadow: none !important;
        }

        /* Stats Panel */
        .stats-container {
            max-width: 1400px;
            margin: 0 auto 30px auto;
            padding: 0 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }

        .stat-card {
            padding: 24px;
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .stat-icon {
            width: 54px;
            height: 54px;
            border-radius: 12px;
            background: rgba(155, 93, 229, 0.15);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.8rem;
            color: var(--accent-primary);
            border: 1px solid rgba(155, 93, 229, 0.2);
        }

        .stat-card:nth-child(2) .stat-icon {
            background: rgba(241, 91, 181, 0.15);
            color: var(--warning-color);
            border-color: rgba(241, 91, 181, 0.2);
        }

        .stat-card:nth-child(3) .stat-icon {
            background: rgba(0, 245, 212, 0.15);
            color: var(--success-color);
            border-color: rgba(0, 245, 212, 0.2);
        }

        .stat-details h3 {
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .stat-details p {
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 4px;
        }

        /* Main Workspace */
        main {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 20px 60px 20px;
        }

        .loader-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 100px 20px;
        }

        .spinner {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(155, 93, 229, 0.1);
            border-radius: 50%;
            border-left-color: var(--accent-primary);
            animation: spin 1s linear infinite;
            box-shadow: 0 0 15px var(--accent-glow);
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loader-container p {
            margin-top: 24px;
            color: var(--text-secondary);
            font-weight: 500;
        }

        /* Groups List */
        .duplicate-groups-list {
            display: flex;
            flex-direction: column;
            gap: 40px;
        }

        .group-container {
            padding: 24px;
            animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .group-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
            flex-wrap: wrap;
            gap: 12px;
        }

        .group-title {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .group-badge {
            background: rgba(241, 91, 181, 0.12);
            color: var(--warning-color);
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid rgba(241, 91, 181, 0.2);
        }

        .group-header h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.35rem;
            font-weight: 700;
        }

        .group-meta {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .group-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
        }

        /* Image Card styling */
        .img-card {
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.02);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            display: flex;
            flex-direction: column;
        }

        .img-card:hover {
            transform: translateY(-4px);
            border-color: rgba(155, 93, 229, 0.25);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }

        .img-card.keep-card {
            border-color: rgba(46, 196, 182, 0.2);
        }

        .img-card.keep-card:hover {
            border-color: rgba(46, 196, 182, 0.45);
        }

        .img-preview-container {
            width: 100%;
            height: 220px;
            background: #000;
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
        }

        .img-preview {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.5s ease;
        }

        .img-card:hover .img-preview {
            transform: scale(1.06);
        }

        .card-badge {
            position: absolute;
            top: 12px;
            left: 12px;
            padding: 5px 12px;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            backdrop-filter: blur(4px);
            z-index: 2;
        }

        .badge-keep {
            background: rgba(46, 196, 182, 0.2);
            color: var(--keep-color);
            border: 1px solid rgba(46, 196, 182, 0.4);
        }

        .badge-duplicate {
            background: rgba(231, 29, 54, 0.15);
            color: var(--delete-color);
            border: 1px solid rgba(231, 29, 54, 0.3);
        }

        .card-details {
            padding: 16px;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .card-filename {
            font-weight: 600;
            font-size: 0.95rem;
            word-break: break-all;
            margin-bottom: 12px;
            color: #fff;
        }

        .metadata-row {
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-bottom: 16px;
            font-size: 0.85rem;
        }

        .metadata-item {
            display: flex;
            justify-content: space-between;
            color: var(--text-secondary);
        }

        .metadata-value {
            color: var(--text-primary);
            font-weight: 500;
        }

        .card-actions {
            display: flex;
            gap: 10px;
        }

        .card-actions .btn {
            flex-grow: 1;
            padding: 8px 16px;
            font-size: 0.8rem;
            justify-content: center;
        }

        /* Modal Preview */
        .modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(11, 8, 19, 0.9);
            backdrop-filter: blur(8px);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
            padding: 20px;
        }

        .modal.active {
            opacity: 1;
            pointer-events: auto;
        }

        .modal-content {
            max-width: 90%;
            max-height: 90%;
            position: relative;
            transform: scale(0.9);
            transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .modal.active .modal-content {
            transform: scale(1);
        }

        .modal-img {
            max-width: 100%;
            max-height: 80vh;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            border: 1px solid var(--border-color);
        }

        .modal-caption {
            margin-top: 15px;
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            text-align: center;
            color: #fff;
        }

        .modal-close {
            position: absolute;
            top: -40px;
            right: 0;
            background: none;
            border: none;
            color: #fff;
            font-size: 2rem;
            cursor: pointer;
        }

        /* Toast notifications */
        .toast-container {
            position: fixed;
            top: 30px;
            right: 30px;
            z-index: 2000;
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-width: 400px;
        }

        .toast {
            background: rgba(26, 21, 44, 0.9);
            border-left: 4px solid var(--accent-primary);
            border-top: 1px solid var(--border-color);
            border-bottom: 1px solid var(--border-color);
            border-right: 1px solid var(--border-color);
            backdrop-filter: blur(12px);
            padding: 16px 20px;
            border-radius: 8px;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            animation: slideIn 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) both;
            cursor: pointer;
        }

        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        .toast-success {
            border-left-color: var(--success-color);
        }

        .toast-error {
            border-left-color: var(--danger-color);
        }

        .empty-state {
            text-align: center;
            padding: 80px 20px;
        }

        .empty-icon {
            font-size: 4rem;
            color: var(--text-secondary);
            margin-bottom: 20px;
        }

        .empty-state h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.6rem;
            margin-bottom: 8px;
        }

        .empty-state p {
            color: var(--text-secondary);
        }

        /* Card removing animation */
        .card-removing {
            transform: scale(0.8) !important;
            opacity: 0;
            transition: all 0.4s ease;
        }
        
        .group-removing {
            opacity: 0;
            max-height: 0;
            padding-top: 0;
            padding-bottom: 0;
            margin-bottom: 0;
            overflow: hidden;
            transition: all 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <h1>Wallpaper Duplicate Finder</h1>
            <p>Scanning files visually via dHash & aHash. Keeping newer, purging older.</p>
        </div>
        <div class="actions-bar">
            <button class="btn btn-secondary" onclick="scanDuplicates()">Refresh Scan</button>
            <button class="btn btn-primary" id="btn-clean-all" onclick="cleanAllOlder()" disabled>Clean All Older</button>
        </div>
    </header>

    <div class="stats-container">
        <div class="glass-panel stat-card">
            <div class="stat-icon">📷</div>
            <div class="stat-details">
                <h3>Total Scanned</h3>
                <p id="stat-scanned">-</p>
            </div>
        </div>
        <div class="glass-panel stat-card">
            <div class="stat-icon">📁</div>
            <div class="stat-details">
                <h3>Duplicate Copies</h3>
                <p id="stat-duplicates">-</p>
            </div>
        </div>
        <div class="glass-panel stat-card">
            <div class="stat-icon">💾</div>
            <div class="stat-details">
                <h3>Reclaimable Space</h3>
                <p id="stat-space">-</p>
            </div>
        </div>
    </div>

    <main>
        <!-- Loader -->
        <div class="loader-container" id="loader">
            <div class="spinner"></div>
            <p id="loader-text">Analyzing image collection... This might take a moment on the first run.</p>
        </div>

        <!-- Groups List -->
        <div class="duplicate-groups-list" id="groups-list"></div>
    </main>

    <!-- Fullscreen Preview Modal -->
    <div class="modal" id="preview-modal" onclick="closeModal()">
        <div class="modal-content" onclick="event.stopPropagation()">
            <button class="modal-close" onclick="closeModal()">&times;</button>
            <img src="" class="modal-img" id="modal-image">
            <div class="modal-caption" id="modal-caption"></div>
        </div>
    </div>

    <!-- Toast Container -->
    <div class="toast-container" id="toast-container"></div>

    <script>
        let duplicateData = null;

        function showToast(message, type = 'success') {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = `toast toast-${type}`;
            
            let icon = '⚡';
            if (type === 'success') icon = '✅';
            if (type === 'error') icon = '❌';

            toast.innerHTML = `<div>${icon}</div><div>${message}</div>`;
            toast.onclick = () => toast.remove();
            container.appendChild(toast);

            setTimeout(() => {
                toast.style.opacity = '0';
                toast.style.transform = 'translateX(100%)';
                toast.style.transition = 'all 0.3s ease';
                setTimeout(() => toast.remove(), 300);
            }, 4000);
        }

        function formatBytes(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        function formatDate(dateStr) {
            if (!dateStr) return 'Unknown';
            try {
                const date = new Date(dateStr);
                return date.toLocaleString();
            } catch(e) {
                return dateStr;
            }
        }

        async function scanDuplicates() {
            document.getElementById('loader').style.display = 'flex';
            document.getElementById('groups-list').style.display = 'none';
            document.getElementById('btn-clean-all').disabled = true;

            try {
                const response = await fetch('/api/scan');
                const result = await response.json();

                if (!result.success) {
                    showToast(result.message, 'error');
                    document.getElementById('loader').style.display = 'none';
                    return;
                }

                duplicateData = result;
                
                // Update stats
                document.getElementById('stat-scanned').innerText = result.total_scanned;
                document.getElementById('stat-duplicates').innerText = result.total_duplicates;
                document.getElementById('stat-space').innerText = formatBytes(result.reclaimable_space_bytes);

                const btnCleanAll = document.getElementById('btn-clean-all');
                if (result.total_duplicates > 0) {
                    btnCleanAll.disabled = false;
                    renderGroups(result.groups);
                } else {
                    btnCleanAll.disabled = true;
                    renderEmptyState();
                }

            } catch (error) {
                showToast(`Scan failed: ${error}`, 'error');
            } finally {
                document.getElementById('loader').style.display = 'none';
                document.getElementById('groups-list').style.display = 'flex';
            }
        }

        function renderEmptyState() {
            const list = document.getElementById('groups-list');
            list.innerHTML = `
                <div class="glass-panel empty-state">
                    <div class="empty-icon">🎉</div>
                    <h2>No Duplicate Wallpapers Found</h2>
                    <p>Your database is perfectly clean. High five!</p>
                </div>
            `;
        }

        function renderGroups(groups) {
            const list = document.getElementById('groups-list');
            list.innerHTML = '';

            groups.forEach((group, groupIdx) => {
                const groupEl = document.createElement('div');
                groupEl.className = 'glass-panel group-container';
                groupEl.id = `group-container-${groupIdx}`;

                // Calculate group reclaimable space
                const olderItems = group.items.filter(item => !item.is_keep);
                const totalGroupSize = olderItems.reduce((acc, item) => acc + item.total_size_bytes, 0);
                const deleteFileNames = olderItems.map(item => item.file_name);

                groupEl.innerHTML = `
                    <div class="group-header">
                        <div class="group-title">
                            <h2>Group #${groupIdx + 1}</h2>
                            <span class="group-badge">${group.items.length} copies found</span>
                        </div>
                        <div style="display:flex; align-items:center; gap: 16px;">
                            <span class="group-meta">Reclaimable: <b>${formatBytes(totalGroupSize)}</b></span>
                            <button class="btn btn-secondary btn-danger" style="padding: 6px 12px; font-size: 0.75rem;" 
                                onclick="cleanGroup('${group.hash}', ${groupIdx}, ${JSON.stringify(deleteFileNames).replace(/"/g, '&quot;')})">
                                Purge Older
                            </button>
                        </div>
                    </div>
                    <div class="group-grid" id="group-grid-${groupIdx}">
                    </div>
                `;

                list.appendChild(groupEl);

                const grid = document.getElementById(`group-grid-${groupIdx}`);
                group.items.forEach((item, itemIdx) => {
                    const card = document.createElement('div');
                    card.className = `img-card ${item.is_keep ? 'keep-card' : ''}`;
                    card.id = `card-${groupIdx}-${itemIdx}`;

                    // Cache source or main source if cache doesn't exist
                    const imgSrc = item.file_cache_name ? `/cache/${encodeURIComponent(item.file_cache_name)}` : `/main/${encodeURIComponent(item.file_main_name)}`;
                    const previewPath = `/main/${encodeURIComponent(item.file_main_name)}`;

                    card.innerHTML = `
                        <div class="img-preview-container" onclick="openPreview('${previewPath}', '${item.file_name}')">
                            <span class="card-badge ${item.is_keep ? 'badge-keep' : 'badge-duplicate'}">
                                ${item.is_keep ? 'Newest (Keep)' : 'Older (Duplicate)'}
                            </span>
                            <img src="${imgSrc}" class="img-preview" loading="lazy">
                        </div>
                        <div class="card-details">
                            <div>
                                <div class="card-filename" title="${item.file_name}">${item.file_name}</div>
                                <div class="metadata-row">
                                    <div class="metadata-item">
                                        <span>Resolution:</span>
                                        <span class="metadata-value">${item.resolution || 'Unknown'} (${item.width || '?'}x${item.height || '?'})</span>
                                    </div>
                                    <div class="metadata-item">
                                        <span>Total Size:</span>
                                        <span class="metadata-value">${formatBytes(item.total_size_bytes)}</span>
                                    </div>
                                    <div class="metadata-item">
                                        <span>Created:</span>
                                        <span class="metadata-value">${formatDate(item.timestamp)}</span>
                                    </div>
                                    <div class="metadata-item">
                                        <span>Category:</span>
                                        <span class="metadata-value">${item.category || 'None'}</span>
                                    </div>
                                </div>
                            </div>
                            <div class="card-actions">
                                <button class="btn btn-secondary ${item.is_keep ? 'btn-danger' : 'btn-danger'}" 
                                    onclick="deleteIndividual('${item.file_name}', ${groupIdx}, ${itemIdx})">
                                    Delete
                                </button>
                            </div>
                        </div>
                    `;
                    grid.appendChild(card);
                });
            });
        }

        async function deleteIndividual(filename, groupIdx, itemIdx) {
            const confirmed = confirm(`Are you sure you want to permanently delete the wallpaper "${filename}"? This deletes both original and cached images and clears it from index.json.`);
            if (!confirmed) return;

            try {
                const response = await fetch('/api/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file_names: [filename] })
                });

                const result = await response.json();
                if (result.success) {
                    showToast(`Successfully deleted ${filename}`);
                    
                    const card = document.getElementById(`card-${groupIdx}-${itemIdx}`);
                    card.classList.add('card-removing');
                    
                    // After fadeout, let's refresh scan data to update stats and remaining items
                    setTimeout(() => {
                        scanDuplicates();
                    }, 400);
                } else {
                    showToast(result.message, 'error');
                }
            } catch (error) {
                showToast(`Delete failed: ${error}`, 'error');
            }
        }

        async function cleanGroup(hash, groupIdx, fileNames) {
            if (!fileNames || fileNames.length === 0) {
                showToast("No older duplicates to delete.", "info");
                return;
            }
            
            const confirmed = confirm(`Are you sure you want to delete all ${fileNames.length} older duplicates in Group #${groupIdx + 1}?`);
            if (!confirmed) return;

            try {
                const response = await fetch('/api/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file_names: fileNames })
                });

                const result = await response.json();
                if (result.success) {
                    showToast(`Group #${groupIdx + 1} cleaned successfully!`);
                    
                    const container = document.getElementById(`group-container-${groupIdx}`);
                    container.classList.add('group-removing');
                    
                    setTimeout(() => {
                        scanDuplicates();
                    }, 600);
                } else {
                    showToast(result.message, 'error');
                }
            } catch (error) {
                showToast(`Group cleaning failed: ${error}`, 'error');
            }
        }

        async function cleanAllOlder() {
            // Find all file names marked as duplicates (items where is_keep = false)
            let filesToDelete = [];
            duplicateData.groups.forEach(group => {
                group.items.forEach(item => {
                    if (!item.is_keep) {
                        filesToDelete.push(item.file_name);
                    }
                });
            });

            if (filesToDelete.length === 0) {
                showToast("No older duplicates found.", "info");
                return;
            }

            const confirmed = confirm(`CAUTION: You are about to permanently delete ${filesToDelete.length} older duplicate wallpapers. This action cannot be undone. Do you wish to proceed?`);
            if (!confirmed) return;

            try {
                const response = await fetch('/api/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file_names: filesToDelete })
                });

                const result = await response.json();
                if (result.success) {
                    showToast(`Successfully deleted ${filesToDelete.length} older wallpapers!`);
                    
                    const list = document.getElementById('groups-list');
                    list.querySelectorAll('.group-container').forEach(el => {
                        el.classList.add('group-removing');
                    });
                    
                    setTimeout(() => {
                        scanDuplicates();
                    }, 600);
                } else {
                    showToast(result.message, 'error');
                }
            } catch (error) {
                showToast(`Bulk delete failed: ${error}`, 'error');
            }
        }

        function openPreview(imgUrl, caption) {
            document.getElementById('modal-image').src = imgUrl;
            document.getElementById('modal-caption').innerText = caption;
            document.getElementById('preview-modal').classList.add('active');
        }

        function closeModal() {
            document.getElementById('preview-modal').classList.remove('active');
        }

        // Initialize scan
        scanDuplicates();
    </script>
</body>
</html>
"""

def main():
    # Keep relative resources anchored to this checkout on every platform.
    os.chdir(PROJECT_DIR)
    
    # Initialize port binding
    class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        pass

    server_address = ('', PORT)
    
    # Simple check if port is in use and auto-increment if so
    port = PORT
    while port < PORT + 100:
        try:
            httpd = ThreadingHTTPServer(('', port), DuplicateDetectorHandler)
            break
        except OSError:
            port += 1
            
    url = f"http://localhost:{port}/"
    print("=" * 60)
    print(" WALLPAPER DUPLICATE DETECTOR AND PURGE WEB UTILITY")
    print("=" * 60)
    print(f" * Server started at: {url}")
    print(" * Scan cache: dHash (9x8 bilinear) + aHash (8x8 average)")
    print(" * Press Ctrl+C to stop the server.")
    print("=" * 60)
    
    # Open browser automatically
    webbrowser.open_new_tab(url)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        sys.exit(0)

if __name__ == "__main__":
    main()
