import google.generativeai as genai
from PIL import Image
import cv2
import random
import re
from . import config
from .memory import TextMemory, BiometricMemory
from .tools import Toolbox

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")


"""
=========================================================
      GEMINI BRAIN (TOOLS + SMART CONTEXT)
=========================================================
"""

# -------------------------------------------------------
#  MODEL SELECTOR
# -------------------------------------------------------
def find_model():
    print("[SYSTEM] Scanning for valid models...")
    try:
        my_models = [
            m.name for m in genai.list_models()
            if 'generateContent' in m.supported_generation_methods
        ]

        prefs = [
            "models/gemini-2.5-flash",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-pro"
        ]

        for p in prefs:
            if p in my_models:
                return p

        return my_models[0] if my_models else "models/gemini-1.5-flash"

    except Exception:
        return "models/gemini-1.5-flash"


# -------------------------------------------------------
#  GEMINI BRAIN (CONVERSATION + NAME EXTRACTION)
# -------------------------------------------------------
class GeminiBrain:
    def __init__(self):
        # Configure API key
        if config.GEMINI_API_KEY:
            genai.configure(api_key=config.GEMINI_API_KEY)

        # Pick a model
        self.model_name = find_model()
        print(f"[SYSTEM] Brain Connected: {self.model_name}")

        safety = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        self.model = genai.GenerativeModel(self.model_name, safety_settings=safety)

        # Load memories
        self.text_mem = TextMemory()
        self.bio_mem = BiometricMemory()

        # Tools
        self.tools = Toolbox()

        # Rolling chat history (6 turns)
        self.chat_history = []

    # -------------------------------------------------------
    # EXTRACT NAME (LLM)
    # -------------------------------------------------------
    def extract_name(self, user_message):
        """
        Extract the name of the person being introduced or identifying themselves.
        """

        prompt = f"""
        Analyze this sentence and extract the name of the person being introduced or identifying themselves.
        User said: "{user_message}"

        Examples:
        - "I am Parth" -> Parth
        - "My name is Sarah" -> Sarah
        - "This is my friend Alex" -> Alex
        - "Meet Rohit, he is new" -> Rohit
        - "Everyone, this is Dr. Strange" -> Dr. Strange
        - "Call me TARS" -> TARS

        Rules:
        - If the user explicitly states a name, return ONLY that name.
        - Return just the raw name, nothing else.
        - If you cannot find a clear name, return EXACTLY: None
        """

        try:
            res = self.model.generate_content(prompt)
            text = res.text.strip()

            # Clean output
            text = text.replace(".", "").strip()

            # If model replied something extra, try to isolate first word
            if "\n" in text:
                text = text.split("\n")[0].strip()

            # Treat "None" or empty as no-name
            if text.lower() in ["none", "null", "no name", ""]:
                return None

            # name must be 1-3 words max
            if len(text.split()) > 3:
                return None

            # Alphabetic check
            if not re.match(r"^[A-Za-z][A-Za-z\s\-]*$", text):
                return None

            return text

        except Exception as e:
            print(f"[NAME EXTRACTION ERROR] {e}")
            return None

    # -------------------------------------------------------
    #  EXTRACT USER FACTS (SMARTER LLM VERSION)
    # -------------------------------------------------------
    def extract_facts_from_text(self, user_text):
        """
        Uses LLM to decide if the user stated a permanent fact.
        Prevents gibberish from being saved.
        """
        # Quick filter: Only bother the LLM if it sounds like a fact statement
        triggers = ["i like", "i am", "my", "i hate", "i love", "he is", "she is"]
        if not any(t in user_text.lower() for t in triggers):
            return []

        prompt = f"""
        Extract any permanent or semi-permanent facts about the user from this sentence.
        If the sentence is just chatter, greetings, or questions, return "None".

        User said: "{user_text}"

        Rules:
        1. Return ONLY the fact (e.g., "User likes coffee").
        2. Do not include commands or noise (e.g., ignore "recognize my voice").
        3. If no clear fact, return "None".
        """
        try:
            res = self.model.generate_content(prompt)
            text = res.text.strip()
            if "none" in text.lower():
                return []
            return [text]
        except:
            return []

    # -------------------------------------------------------
    #  MAIN CHAT PROCESSOR
    # -------------------------------------------------------
    def process(self, cv2_img, prompt, detections, active_user_context, face_encoding, depth_dist, state, mode):
        """
        Generates a normal conversational response.
        """

        # 1. CHECK TOOLS FIRST (Fix for Spotify/Commands)
        tool_result = ""
        if prompt:
            # Try to open app or run command
            if self.tools.open_app(prompt):
                tool_result = f"[SYSTEM ACTION: Successfully executed command based on user input: '{prompt}']"
            elif "search" in prompt.lower():
                # Fallback to web search if explicit
                search_res = self.tools.web_search(prompt)
                tool_result = f"[SYSTEM SEARCH RESULT: {search_res}]"

        # Convert CV2 to PIL
        pil_img = None
        if cv2_img is not None:
            try:
                rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
            except:
                pass

        # -------------------------------------------------------
        # Assemble internal context
        # -------------------------------------------------------
        user_profile = ""
        if "People:" in active_user_context:
            # Clean up the context string to get raw names
            # e.g. "People: Parth (happy)" -> ["Parth (happy)"]
            names = active_user_context.replace("People:", "").strip().split("|")
            for n in names:
                # Extract name part before parenthesis
                n = n.split("(")[0].strip()
                if n != "Unknown":
                    user_profile += self.text_mem.get_user_context(n) + "\n"
        
        history_text = ""
        if self.chat_history:
            history_text = "RECENT CHAT HISTORY:\n" + "\n".join(self.chat_history) + "\n"

        dist_msg = "Unknown distance"
        if isinstance(depth_dist, int) and depth_dist < 9000:
            dist_msg = f"{depth_dist/1000:.1f} meters"

        # -------------------------------------------------------
        # System Persona Prompt
        # -------------------------------------------------------
        sys_prompt = f"""
        You are E.D.I (Enhanced Digital Intelligence) ( Pronounced as Edie ) — a warm, expressive, witty, emotional AI companion.
        Keep responses natural, and conversational.
        Do NOT describe the environment, vision, objects, distance, or technical details unless the user explicitly asks.

        --- CRITICAL IDENTITY RULES ---
        1. TRUST THE SENSORS: Check 'People Presence' below. If a name is listed (e.g., 'Parth'), THAT IS who you are talking to.
        2. NO DENIALS: Never say "I don't recognize you" or "I don't have eyes" if a name is provided. You DO know them. Treat them as a recognized friend.
        3. MEMORY: Use the 'USER PROFILE' section to reference facts you know about them naturally.

        Start every message with one of these emotion tags:
        [EMOTION: HAPPY]
        [EMOTION: SURPRISED]
        [EMOTION: SKEPTICAL]
        [EMOTION: IDLE]
        [EMOTION: HUMMING]
        [EMOTION: THINKING]

        Respond naturally, like a human friend.
        Do NOT generate command tags like [LEARN_FACE].

        --- ENVIRONMENT CONTEXT ---
        Current State: {state}
        Detected Objects: {detections}
        Estimated Distance: {dist_msg}
        People Presence: {active_user_context}  <-- LOOK HERE FOR NAME

        --- TOOL OUTPUT ---
        {tool_result}

        --- USER PROFILE(S) ---
        {user_profile}

        --- CHAT HISTORY ---
        {history_text}

        Now respond to:
        User said: "{prompt}"

        If the Tool Output indicates an action was taken, acknowledge it naturally.
        """

        # -------------------------------------------------------
        # Generate Response
        # -------------------------------------------------------
        full_prompt = sys_prompt

        try:
            if pil_img:
                out = self.model.generate_content([full_prompt, pil_img])
            else:
                out = self.model.generate_content(full_prompt)

            response_text = out.text if out.parts else "[EMOTION: IDLE] I'm not sure what to say."

        except Exception as e:
            print(f"[BRAIN ERROR] {e}")
            return "[EMOTION: CONFUSED] Something glitched."

        # -------------------------------------------------------
        # Update history (max 6 entries)
        # -------------------------------------------------------
        if prompt:
            clean_history_line = response_text.replace("[", "").replace("]", "")
            self.chat_history.append(f"User: {prompt}")
            self.chat_history.append(f"Friday: {clean_history_line}")
            if len(self.chat_history) > 6:
                self.chat_history = self.chat_history[-6:]

        return response_text