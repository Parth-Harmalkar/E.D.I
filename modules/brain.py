import google.generativeai as genai
from PIL import Image
import cv2
import random
import re
from . import config
# from .memory import TextMemory, BiometricMemory
from .knowledge_graph import KnowledgeGraph, FactExtractor
from .biometric_memory_v2 import OptimizedBiometricMemory
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
        # self.text_mem = TextMemory()
        # self.bio_mem = BiometricMemory()

        # Tools
        self.tools = Toolbox()

        # New Memory System
        self.knowledge_graph = KnowledgeGraph()
        self.fact_extractor = FactExtractor(self.model)
        self.bio_mem = OptimizedBiometricMemory()

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
        
        # 1. CHECK TOOLS FIRST
        tool_result = ""
        if prompt:
            if self.tools.open_app(prompt):
                tool_result = f"[SYSTEM ACTION: Successfully executed command: '{prompt}']"
            elif "search" in prompt.lower():
                search_res = self.tools.web_search(prompt)
                tool_result = f"[SYSTEM SEARCH RESULT: {search_res}]"

        # 2. EXTRACT FACTS & UPDATE GRAPH (NEW)
        # We need to identify the user name from the context string "People: Name (Active)"
        current_user = "Unknown"
        if "People:" in active_user_context:
            try:
                # Extracts "Parth" from "People: Parth (Active)"
                current_user = active_user_context.split("People:")[1].split("(")[0].strip()
            except:
                pass

        if prompt and current_user not in ["Unknown", "Off-Screen Speaker"]:
            # Use the new FactExtractor
            extracted_data = self.fact_extractor.extract_from_text(current_user, prompt)
            
            # Add to Graph
            if extracted_data.get("entities"):
                for entity in extracted_data["entities"]:
                    self.knowledge_graph.add_entity(entity["id"], entity["type"], entity.get("properties", {}))
            
            if extracted_data.get("relationships"):
                for rel in extracted_data["relationships"]:
                    self.knowledge_graph.add_relationship(rel["from"], rel["to"], rel["type"], rel.get("strength", 0.8))

        # 3. RETRIEVE CONTEXT FROM GRAPH (NEW)
        user_profile = ""
        if current_user != "Unknown":
            # Query the graph instead of the old TextMemory
            user_profile = self.knowledge_graph.query_context(current_user, context_type="recent")

        # 4. PREPARE IMAGE
        pil_img = None
        if cv2_img is not None:
            try:
                rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
            except:
                pass

        # 5. ASSEMBLE PROMPT
        history_text = ""
        if self.chat_history:
            history_text = "RECENT CHAT HISTORY:\n" + "\n".join(self.chat_history) + "\n"

        sys_prompt = f"""
        You are E.D.I (Enhanced Digital Intelligence). A smart, witty, warm and emotional AI companion.
        
        --- IDENTITY & GROUNDING ---
        1. WHO YOU ARE TALKING TO: The 'People Presence' below is the absolute truth. If it says 'Parth', you are talking to Parth.
        2. MEMORY: Use the 'Memory' section to recall past details naturally. Don't announce "I have accessed your file." Just say "How's the coffee?"
        
        --- PERSONALITY RULES ---
        1. BE HUMAN: Talk like a smart, witty friend. Be concise. No flowery language or long paragraphs.
        2. BE OBSERVANT: You see the context, but don't list everything (e.g., don't say "I see you are wearing a shirt"). Only mention things if they are interesting or changed.
        3. BE DIRECT: If the user says "hello", just say "Hey [Name], what's up?".
        4. EMOTION: Start every response with a hidden tag like [EMOTION: HAPPY], [EMOTION: BORED], etc.

        --- ENVIRONMENT ---
        People Presence: {active_user_context}
        Objects: {detections}
        
        --- MEMORY CONTEXT ---
        {user_profile}

        --- TOOL RESULTS ---
        {tool_result}

        --- HISTORY ---
        {history_text}

        User said: "{prompt}"

        Respond naturally.  
        """

        # 6. GENERATE RESPONSE
        try:
            if pil_img:
                out = self.model.generate_content([sys_prompt, pil_img])
            else:
                out = self.model.generate_content(sys_prompt)
            response_text = out.text if out.parts else "[EMOTION: IDLE] ..."
        except Exception as e:
            print(f"[BRAIN ERROR] {e}")
            return "[EMOTION: CONFUSED] Error processing."

        # 7. LOG EPISODE TO GRAPH (NEW)
        if prompt and current_user != "Unknown":
            self.knowledge_graph.add_episode(
                participants=[current_user],
                summary=f"User asked: {prompt}",
                key_facts=[response_text[:100]] # Store snippet of AI response
            )

        # Update History
        if prompt:
            clean_history_line = response_text.replace("[", "").replace("]", "")
            self.chat_history.append(f"User: {prompt}")
            self.chat_history.append(f"EDI: {clean_history_line}")
            if len(self.chat_history) > 6:
                self.chat_history = self.chat_history[-6:]

        return response_text