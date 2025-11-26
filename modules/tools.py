import subprocess
import webbrowser
import urllib.parse
import os
from ddgs import DDGS

class Toolbox:
    def __init__(self):
        self.ddgs = DDGS()
        
        # Define System Commands
        # Note: "start" is for Windows.
        self.apps = {
            "chrome": "start chrome", 
            "google": "start chrome", 
            "brave": "start brave", 
            "notepad": "start notepad",
            "calculator": "calc", 
            "camera": "start microsoft.windows.camera:",
            "files": "explorer", 
            "file manager": "explorer",
            "cmd": "start cmd",
            "terminal": "start wt",
            "code": "code",
            "vscode": "code",
            "spotify": "start spotify:",
            "whatsapp": "start whatsapp:",
            "youtube": "start https://www.youtube.com",
            "music": "start https://music.youtube.com"
        }

    def open_app(self, prompt):
        """
        Scans the prompt for known app keywords and launches them.
        Returns True if an app was launched, False otherwise.
        """
        clean_prompt = prompt.lower().strip()
        
        # 1. MUSIC / YOUTUBE SPECIFIC
        # Handles "play [song name]"
        if "play" in clean_prompt:
            try:
                # Extract song name: "play believing by journey" -> "believing by journey"
                parts = clean_prompt.split("play", 1)
                if len(parts) > 1:
                    song = parts[1].strip()
                    if song:
                        query = urllib.parse.quote(song)
                        url = f"https://music.youtube.com/search?q={query}"
                        print(f"[TOOL] Playing Music: {url}")
                        webbrowser.open(url)
                        return True
            except Exception as e:
                print(f"[TOOL ERROR] Music launch failed: {e}")

        # 2. CHECK KNOWN APPS (Keyword Search)
        # We check if any app key (e.g. "spotify") is inside the user's prompt
        for app_name, command in self.apps.items():
            if app_name in clean_prompt:
                print(f"[TOOL] Launching App: {app_name.upper()}")
                try:
                    # Shell=True is required for 'start' commands on Windows
                    subprocess.Popen(command, shell=True)
                    return True
                except Exception as e: 
                    print(f"[TOOL ERROR] Failed to launch {app_name}: {e}")
                    # Fallback for web apps (urls)
                    if "http" in command:
                        try:
                            url = command.split("start ")[1]
                            webbrowser.open(url)
                            return True
                        except:
                            pass
                    return False

        # 3. GENERIC WEB SEARCH
        # Handles "search for [topic]" or "google [topic]"
        if "search" in clean_prompt or "google" in clean_prompt:
            query = clean_prompt.replace("search", "").replace("google", "").replace("for", "").strip()
            if query:
                url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
                print(f"[TOOL] Opening Search: {url}")
                webbrowser.open(url)
                return True
            
        # 4. DIRECT URL HANDLING
        words = clean_prompt.split()
        for w in words:
            if "." in w and not w.endswith(".") and len(w) > 4: 
                if w.startswith("open"): continue
                
                url = f"https://{w}" if not w.startswith("http") else w
                print(f"[TOOL] Opening URL: {url}")
                webbrowser.open(url)
                return True

        return False

    def web_search(self, query):
        """
        Performs a text search using DuckDuckGo and returns a summary string.
        """
        print(f"[TOOL] Reading about: {query}")
        try:
            results = list(self.ddgs.text(query, max_results=2))
            
            if not results: 
                return "I couldn't find any information on that."
            
            summary = "Search Results:\n"
            for i, res in enumerate(results, 1):
                title = res.get('title', 'No Title')
                body = res.get('body', 'No Description')
                summary += f"{i}. {title}: {body}\n"
            
            return summary.strip()
            
        except Exception as e:
            print(f"[SEARCH ERROR] {e}")
            return "I'm having trouble accessing the internet right now."