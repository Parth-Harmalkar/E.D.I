"""
=========================================================
  AUTONOMOUS KNOWLEDGE GRAPH ENGINE
  Neo4j-Style Relational Memory with Temporal Logic
=========================================================

This replaces the flat string storage in TextMemory with:
  ✓ Entity-Relationship model
  ✓ Temporal decay (recent facts weighted higher)
  ✓ Bidirectional inference
  ✓ Episodic conversation memory
  ✓ Context-aware fact retrieval
"""

import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import google.generativeai as genai
from . import config

class KnowledgeGraph:
    def __init__(self, storage_path="None"):
        if storage_path is None:
            self.storage_path = config.MEM_DIR / "knowledge_graph.json"
        else:
            self.storage_path = Path(storage_path)
            
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Graph Structure
        self.entities = {}  # {entity_id: {type, properties, created, last_updated}}
        self.relationships = []  # [{from, to, type, strength, created, context}]
        self.episodes = []  # [{timestamp, participants, summary, key_facts}]
        
        self.load()
    
    # ===================================================================
    # PERSISTENCE
    # ===================================================================
    def save(self):
        data = {
            "entities": self.entities,
            "relationships": self.relationships,
            "episodes": self.episodes
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                self.entities = data.get("entities", {})
                self.relationships = data.get("relationships", [])
                self.episodes = data.get("episodes", [])
                print(f"[KNOWLEDGE] Loaded {len(self.entities)} entities, {len(self.relationships)} relationships")
            except Exception as e:
                print(f"[KNOWLEDGE ERROR] {e}")
    
    # ===================================================================
    # ENTITY MANAGEMENT
    # ===================================================================
    def add_entity(self, entity_id, entity_type, properties=None):
        """
        Add or update an entity.
        Example: add_entity("person_parth", "Person", {"name": "Parth", "role": "user"})
        """
        if properties is None:
            properties = {}
        
        now = datetime.now().isoformat()
        
        if entity_id not in self.entities:
            self.entities[entity_id] = {
                "type": entity_type,
                "properties": properties,
                "created": now,
                "last_updated": now
            }
            print(f"[KNOWLEDGE] Created entity: {entity_id} ({entity_type})")
        else:
            # Update existing
            self.entities[entity_id]["properties"].update(properties)
            self.entities[entity_id]["last_updated"] = now
            print(f"[KNOWLEDGE] Updated entity: {entity_id}")
        
        self.save()
        return entity_id
    
    def get_entity(self, entity_id):
        return self.entities.get(entity_id)
    
    def find_entities_by_type(self, entity_type):
        """Find all entities of a given type (e.g., all Person entities)"""
        return {eid: e for eid, e in self.entities.items() if e["type"] == entity_type}
    
    # ===================================================================
    # RELATIONSHIP MANAGEMENT
    # ===================================================================
    def add_relationship(self, from_id, to_id, rel_type, strength=1.0, context=None):
        """
        Create a relationship between two entities.
        Example: add_relationship("person_parth", "drink_coffee", "LIKES", 0.9)
        """
        now = datetime.now().isoformat()
        
        # Check if relationship already exists
        for rel in self.relationships:
            if (rel["from"] == from_id and 
                rel["to"] == to_id and 
                rel["type"] == rel_type):
                # Update strength (weighted average)
                rel["strength"] = (rel["strength"] * 0.7) + (strength * 0.3)
                rel["last_reinforced"] = now
                print(f"[KNOWLEDGE] Reinforced: {from_id} → {rel_type} → {to_id}")
                self.save()
                return
        
        # Create new relationship
        rel = {
            "from": from_id,
            "to": to_id,
            "type": rel_type,
            "strength": strength,
            "created": now,
            "last_reinforced": now,
            "context": context or {}
        }
        self.relationships.append(rel)
        print(f"[KNOWLEDGE] New relationship: {from_id} → {rel_type} → {to_id}")
        
        # Bidirectional inference
        self._create_inverse_relationship(from_id, to_id, rel_type, strength, context)
        
        self.save()
    
    def _create_inverse_relationship(self, from_id, to_id, rel_type, strength, context):
        """
        Auto-create inverse relationships where logical.
        Example: "Parth SIBLING_OF Sarah" → "Sarah SIBLING_OF Parth"
        """
        bidirectional_types = {
            "SIBLING_OF": "SIBLING_OF",
            "FRIEND_OF": "FRIEND_OF",
            "COLLEAGUE_OF": "COLLEAGUE_OF",
            "SPOUSE_OF": "SPOUSE_OF"
        }
        
        inverse_types = {
            "PARENT_OF": "CHILD_OF",
            "CHILD_OF": "PARENT_OF",
            "OWNS": "OWNED_BY",
            "MANAGES": "MANAGED_BY"
        }
        
        if rel_type in bidirectional_types:
            inverse_type = bidirectional_types[rel_type]
            self.relationships.append({
                "from": to_id,
                "to": from_id,
                "type": inverse_type,
                "strength": strength,
                "created": datetime.now().isoformat(),
                "last_reinforced": datetime.now().isoformat(),
                "context": context or {},
                "inferred": True
            })
        elif rel_type in inverse_types:
            inverse_type = inverse_types[rel_type]
            self.relationships.append({
                "from": to_id,
                "to": from_id,
                "type": inverse_type,
                "strength": strength,
                "created": datetime.now().isoformat(),
                "last_reinforced": datetime.now().isoformat(),
                "context": context or {},
                "inferred": True
            })
    
    def get_relationships(self, entity_id, rel_type=None):
        """
        Get all relationships for an entity.
        If rel_type specified, filter by relationship type.
        """
        results = []
        for rel in self.relationships:
            if rel["from"] == entity_id:
                if rel_type is None or rel["type"] == rel_type:
                    results.append(rel)
        return results
    
    # ===================================================================
    # EPISODIC MEMORY (CONVERSATION TRACKING)
    # ===================================================================
    def add_episode(self, participants, summary, key_facts=None):
        """
        Store a conversation episode with participants and key takeaways.
        """
        episode = {
            "timestamp": datetime.now().isoformat(),
            "participants": participants,
            "summary": summary,
            "key_facts": key_facts or []
        }
        self.episodes.append(episode)
        
        # Keep only last 50 episodes
        if len(self.episodes) > 50:
            self.episodes = self.episodes[-50:]
        
        self.save()
    
    def get_recent_episodes(self, participant=None, days=7):
        """
        Retrieve recent episodes, optionally filtered by participant.
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        results = []
        for ep in reversed(self.episodes):
            ep_time = datetime.fromisoformat(ep["timestamp"])
            if ep_time < cutoff:
                break
            
            if participant is None or participant in ep["participants"]:
                results.append(ep)
        
        return results
    
    # ===================================================================
    # TEMPORAL DECAY & RELEVANCE SCORING
    # ===================================================================
    def get_temporal_weight(self, timestamp_str):
        """
        Calculate relevance weight based on age.
        Recent = 1.0, Old = exponential decay
        """
        try:
            ts = datetime.fromisoformat(timestamp_str)
            age_days = (datetime.now() - ts).days
            
            # Exponential decay: weight = e^(-age/30)
            # Facts from today: 1.0
            # Facts from 30 days ago: 0.37
            # Facts from 90 days ago: 0.05
            import math
            weight = math.exp(-age_days / 30.0)
            return max(0.05, weight)  # Minimum 5% weight
        except:
            return 0.5
    
    # ===================================================================
    # CONTEXT-AWARE RETRIEVAL (SMART QUERIES)
    # ===================================================================
    def query_context(self, person_name, context_type="full"):
        """
        Build a rich context string for a person.
        
        context_type:
          - "full": Everything we know
          - "recent": Only facts from last 7 days
          - "preferences": Only likes/dislikes
        """
        person_id = f"person_{person_name.lower()}"
        
        if person_id not in self.entities:
            return f"I don't have any information about {person_name} yet."
        
        context_lines = []
        context_lines.append(f"=== PROFILE: {person_name} ===")
        
        # Basic properties
        props = self.entities[person_id]["properties"]
        for key, val in props.items():
            if key != "name":
                context_lines.append(f"- {key}: {val}")
        
        # Relationships (with temporal weighting)
        rels = self.get_relationships(person_id)
        
        if context_type == "preferences":
            pref_types = ["LIKES", "DISLIKES", "ENJOYS", "HATES"]
            rels = [r for r in rels if r["type"] in pref_types]
        
        if context_type == "recent":
            rels = [r for r in rels if self.get_temporal_weight(r["created"]) > 0.5]
        
        if rels:
            context_lines.append("\n=== RELATIONSHIPS ===")
            for rel in rels:
                to_entity = self.get_entity(rel["to"])
                if to_entity:
                    to_name = to_entity["properties"].get("name", rel["to"])
                    strength = rel["strength"]
                    age_weight = self.get_temporal_weight(rel["created"])
                    context_lines.append(
                        f"- {rel['type']} {to_name} "
                        f"(confidence: {strength:.2f}, relevance: {age_weight:.2f})"
                    )
        
        # Recent episodes
        episodes = self.get_recent_episodes(person_name, days=7)
        if episodes and context_type in ["full", "recent"]:
            context_lines.append("\n=== RECENT CONVERSATIONS ===")
            for ep in episodes[:3]:
                context_lines.append(f"- {ep['timestamp'][:10]}: {ep['summary']}")
        
        return "\n".join(context_lines)
    
    # ===================================================================
    # INFERENCE ENGINE (PATH FINDING)
    # ===================================================================
    def find_connection(self, entity1_id, entity2_id, max_depth=3):
        """
        Find connection path between two entities.
        Example: "How is Parth connected to Coffee?"
        → Parth → WORKS_AT → Office → HAS → Coffee Machine → SERVES → Coffee
        """
        # BFS to find shortest path
        from collections import deque
        
        queue = deque([(entity1_id, [entity1_id])])
        visited = {entity1_id}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current == entity2_id:
                return path
            
            # Explore neighbors
            for rel in self.get_relationships(current):
                neighbor = rel["to"]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None  # No connection found
    
    def explain_connection(self, path):
        """
        Convert a connection path into human-readable explanation.
        """
        if not path or len(path) < 2:
            return "No connection found."
        
        explanation = []
        for i in range(len(path) - 1):
            from_id = path[i]
            to_id = path[i + 1]
            
            from_entity = self.get_entity(from_id)
            to_entity = self.get_entity(to_id)
            
            from_name = from_entity["properties"].get("name", from_id) if from_entity else from_id
            to_name = to_entity["properties"].get("name", to_id) if to_entity else to_id
            
            # Find the relationship
            for rel in self.get_relationships(from_id):
                if rel["to"] == to_id:
                    explanation.append(f"{from_name} → {rel['type']} → {to_name}")
                    break
        
        return " | ".join(explanation)


# ===================================================================
# LLM-POWERED FACT EXTRACTOR
# ===================================================================
class FactExtractor:
    """
    Uses Gemini to extract structured entities and relationships
    from natural language conversations.
    """
    
    def __init__(self, model):
        self.model = model
    
    def extract_from_text(self, user_name, user_text):
        """
        Returns: {
            "entities": [{"id": "drink_coffee", "type": "Beverage", "properties": {...}}],
            "relationships": [{"from": "person_parth", "to": "drink_coffee", "type": "LIKES", "strength": 0.9}]
        }
        """
        prompt = f"""
You are a knowledge extraction system. Extract entities and relationships from this conversation.

Speaker: {user_name}
Text: "{user_text}"

Output ONLY valid JSON with this structure:
{{
  "entities": [
    {{"id": "entity_id", "type": "Person|Place|Object|Concept", "properties": {{"name": "..."}}}}
  ],
  "relationships": [
    {{"from": "entity1_id", "to": "entity2_id", "type": "LIKES|WORKS_AT|KNOWS|etc", "strength": 0.0-1.0}}
  ]
}}

Rules:
1. Always create entity IDs as: type_name (e.g., "person_parth", "place_office", "drink_coffee")
2. Relationship types: LIKES, DISLIKES, WORKS_AT, LIVES_IN, KNOWS, OWNS, etc.
3. Strength 0.8-1.0 for definite facts, 0.5-0.7 for implied, 0.0-0.4 for weak
4. If no meaningful entities, return empty arrays
"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Clean markdown fences
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(text)
            return data
        except Exception as e:
            print(f"[FACT EXTRACTION ERROR] {e}")
            return {"entities": [], "relationships": []}