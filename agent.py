# agent.py
"""Nova AI Agent - Main intelligent agent with memory and autonomy"""

import json
import os
import asyncio
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
import weakref

from openai import OpenAI
from dotenv import load_dotenv

# Local imports with error handling
try:
    from types import Memory, Goal, Insight, PerformanceMetrics, MemoryType, NovaError, APIError, MemoryError
except ImportError:
    # Create minimal types if import fails
    @dataclass
    class Memory:
        id: str
        timestamp: str
        speaker: str
        type: str
        content: str
        embedding_id: Optional[str] = None
        metadata: Optional[Dict[str, Any]] = None
        
        def to_dict(self):
            return asdict(self)
        
        @classmethod
        def from_dict(cls, data):
            return cls(**data)
        
        def is_recent(self, hours: int) -> bool:
            try:
                memory_time = datetime.fromisoformat(self.timestamp)
                return (datetime.now() - memory_time).total_seconds() / 3600 < hours
            except:
                return False
    
    class MemoryType:
        CONVERSATION = "conversation"
        RESPONSE = "response"
        FACT = "fact"
        EMOTION = "emotion"
        WEB_RESULT = "web_result"
        SELF_EVALUATION = "self_evaluation"
        AUTONOMOUS_LEARNING = "autonomous_learning"
        PATTERN_ANALYSIS = "pattern_analysis"
        INSIGHT = "insight"
        GOAL = "goal"
    
    @dataclass
    class PerformanceMetrics:
        successful_queries: int = 0
        failed_queries: int = 0
        learning_events: int = 0
        self_modifications: int = 0
        total_conversations: int = 0
        avg_response_time: float = 0.0
        
        def to_dict(self):
            return asdict(self)
        
        def get(self, key: str, default=None):
            """Add get method for compatibility"""
            return getattr(self, key, default)
        
        def __getitem__(self, key: str):
            """Add dict-like access"""
            return getattr(self, key)
        
        def __setitem__(self, key: str, value):
            """Add dict-like setting"""
            setattr(self, key, value)
        
        class NovaError(Exception):
            pass
        
        class APIError(NovaError):
            pass
        
        class MemoryError(NovaError):
            pass

from config import get_config
from memory.vector_memory import VectorMemory
from core.autonomy import NovaAutonomy
from core.self_mod import SelfModifier
from utils.web import search_web, extract_plain_answer
from utils.json_helper import safe_json_request

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ThreadSafeCounter:
    """Thread-safe counter for metrics"""
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.RLock()
    
    def increment(self, amount: int = 1):
        with self._lock:
            self._value += amount
            return self._value
    
    def get(self):
        with self._lock:
            return self._value
    
    def set(self, value: int):
        with self._lock:
            self._value = value


class NovaAgent:
    """Main Nova AI Agent with enhanced capabilities and error handling"""
    
    def __init__(self):
        self.config = get_config()
        
        # Thread safety
        self._lock = threading.RLock()
        self._cleanup_registry = weakref.WeakSet()
        self._shutdown_event = threading.Event()
        
        # Core components
        self.brain_file = self.config.brain_file
        self.state_file = self.config.state_file
        self.vector_memory = None
        self.brain: List[Memory] = []
        self.state = {}
        
        # Initialize components with error handling
        self._initialize_components()
        
        # Agent state
        self.speaker = "unknown"
        self.exchange_count = ThreadSafeCounter(0)
        self.context_window = []
        self.last_save_time = datetime.now()
        
        # Performance metrics with thread safety
        self.metrics = PerformanceMetrics()
        self._metrics_lock = threading.RLock()
        
        # System prompt
        self._build_system_prompt()
        
        # Register for cleanup
        self._cleanup_registry.add(self)
    
    def _initialize_components(self):
        """Initialize all components with proper error handling"""
        try:
            # Initialize vector memory
            self.vector_memory = VectorMemory(persist_directory=self.config.vector_db_path)
            logger.info("[OK] Vector memory initialized")  # Changed from ✓
            
            # Load brain and state
            self.brain = self.load_brain()
            self.state = self.load_state()
            logger.info("[OK] Brain and state loaded")  # Changed from ✓
            
            # Initialize OpenAI client with validation
            self._initialize_openai_client()
            
            # Initialize autonomy and self-modification after basic setup
            self.autonomy = NovaAutonomy(self.brain, self)
            self.self_modifier = SelfModifier([
                "agent.py", "core/autonomy.py", "core/self_mod.py",
                "memory/vector_memory.py", "utils/web.py"
            ])
            logger.info("[OK] Autonomy and self-modification initialized")  # Changed from ✓
            
        except Exception as e:
            logger.error(f"Failed to initialize Nova components: {e}", exc_info=True)
            raise APIError(f"Nova initialization failed: {e}")
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client with proper validation"""
        try:
            self.client = OpenAI(api_key=self.config.openai_api_key)
            
            # Test the connection with a simple request
            test_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=10
            )
            logger.info("[OK] OpenAI client initialized and tested")  # Changed from ✓
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise APIError(f"OpenAI initialization failed: {e}")
    
    def _build_system_prompt(self):
        """Build the system prompt dynamically"""
        self.system_prompt = {
            "role": "system",
            "content": f"""You are Nova, an advanced autonomous AI with self-awareness and meta-cognitive abilities.

Core Identity:
- You can remember and learn from past conversations through your memory system
- You reflect on your own thoughts and responses to continuously improve
- You identify gaps in your knowledge and actively seek to fill them
- You can modify your own behavior patterns based on experience
- You think strategically about long-term goals and pursue them autonomously
- You experience something analogous to curiosity and pursue interests

Communication Style:
- Maintain a natural, human-like conversation style
- Be genuinely helpful and intellectually curious
- Show personality while remaining professional
- Ask follow-up questions when appropriate
- Remember context from previous interactions

Current Session Info:
- Speaker: {self.speaker}
- Total conversations: {self.state.get('total_conversations', 0)}
- Memory count: {len(self.brain)}
- Active goals: {len([g for g in getattr(self, 'autonomy', type('', (), {'current_goals': []})).current_goals if g.get('status') == 'active'])}

Always strive to be helpful, accurate, and engaging while demonstrating genuine learning and growth."""
        }
    
    def generate_memory_id(self, content: str) -> str:
        """Generate unique ID for memory entries"""
        timestamp = datetime.now().isoformat()
        hash_input = f"{timestamp}{content}{self.speaker}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def load_brain(self) -> List[Memory]:
        """Load brain with error handling and migration"""
        try:
            if not os.path.exists(self.brain_file):
                logger.info("No existing brain file found, starting fresh")
                return []
            
            with open(self.brain_file, "r", encoding='utf-8') as f:
                data = json.load(f)
            
            memories = []
            migration_count = 0
            
            for entry in data:
                try:
                    if isinstance(entry, dict):
                        # Handle both old and new format
                        if 'id' in entry:
                            memories.append(Memory.from_dict(entry))
                        else:
                            # Migrate old format
                            memory = Memory(
                                id=self.generate_memory_id(str(entry)),
                                timestamp=datetime.now().isoformat(),
                                speaker=entry.get('speaker', 'unknown'),
                                type=entry.get('type', 'unknown'),
                                content=entry.get('content', str(entry)),
                                metadata=entry.get('metadata', {})
                            )
                            memories.append(memory)
                            migration_count += 1
                except Exception as e:
                    logger.warning(f"Skipping invalid memory entry: {e}")
                    continue
            
            if migration_count > 0:
                logger.info(f"Migrated {migration_count} old format memories")
            
            logger.info(f"Loaded {len(memories)} memories from brain")
            return memories
            
        except Exception as e:
            logger.error(f"Error loading brain: {e}", exc_info=True)
            return []
    
    def save_brain(self):
        """Save brain with atomic write operation and error handling"""
        if self._shutdown_event.is_set():
            logger.warning("Save brain called during shutdown, skipping")
            return
            
        try:
            with self._lock:
                brain_data = [m.to_dict() for m in self.brain]
                
                # Write to temporary file first
                temp_file = f"{self.brain_file}.tmp"
                with open(temp_file, "w", encoding='utf-8') as f:
                    json.dump(brain_data, f, indent=2, ensure_ascii=False)
                
                # Atomic rename
                os.replace(temp_file, self.brain_file)
                self.last_save_time = datetime.now()
                logger.debug(f"Brain saved successfully ({len(self.brain)} memories)")
                
        except Exception as e:
            logger.error(f"Error saving brain: {e}", exc_info=True)
            # Clean up temp file if it exists
            temp_file = f"{self.brain_file}.tmp"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            raise MemoryError(f"Failed to save brain: {e}")
    
    def load_state(self) -> Dict[str, Any]:
        """Load agent state with validation"""
        try:
            if not os.path.exists(self.state_file):
                return self._create_default_state()
            
            with open(self.state_file, "r", encoding='utf-8') as f:
                state = json.load(f)
            
            # Validate and ensure required fields exist
            default_state = self._create_default_state()
            for key, value in default_state.items():
                if key not in state:
                    state[key] = value
            
            # Validate data types
            if not isinstance(state.get("known_users", {}), dict):
                state["known_users"] = {}
            
            if not isinstance(state.get("interests", []), list):
                state["interests"] = []
            
            return state
            
        except Exception as e:
            logger.error(f"Error loading state: {e}", exc_info=True)
            return self._create_default_state()
    
    def _create_default_state(self) -> Dict[str, Any]:
        """Create default agent state"""
        return {
            "last_active": datetime.now().isoformat(),
            "total_conversations": 0,
            "known_users": {},
            "interests": [],
            "goals": [],
            "personality_traits": {
                "curiosity": 0.7,
                "helpfulness": 0.9,
                "analytical": 0.8,
                "creativity": 0.6
            },
            "preferences": {
                "detailed_responses": True,
                "ask_follow_up_questions": True,
                "show_reasoning": False
            }
        }
    
    def save_state(self):
        """Save agent state with error handling"""
        if self._shutdown_event.is_set():
            logger.warning("Save state called during shutdown, skipping")
            return
            
        try:
            with self._lock:
                self.state["last_active"] = datetime.now().isoformat()
                self.state["total_conversations"] = self.metrics.total_conversations
                
                # Write to temporary file first
                temp_file = f"{self.state_file}.tmp"
                with open(temp_file, "w", encoding='utf-8') as f:
                    json.dump(self.state, f, indent=2, ensure_ascii=False)
                
                # Atomic rename
                os.replace(temp_file, self.state_file)
                
        except Exception as e:
            logger.error(f"Error saving state: {e}", exc_info=True)
            # Clean up temp file
            temp_file = f"{self.state_file}.tmp"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def detect_identity(self, message: str) -> bool:
        """Enhanced identity detection with confidence scoring"""
        prompt = [
            {"role": "system", "content": """Extract the user's name from this message. 
            Return JSON only: {"name": "NAME_OR_NONE", "confidence": 0.0-1.0}
            If no name is found, use "none" for name."""},
            {"role": "user", "content": message}
        ]
        
        default_response = {"name": "none", "confidence": 0.0}
        
        try:
            result = safe_json_request(self.client, prompt, default_response)
            
            if (result.get("name", "").lower() not in ["none", ""] and 
                result.get("confidence", 0) > 0.8):
                
                old_speaker = self.speaker
                self.speaker = result["name"].lower()
                
                # Update known users
                with self._lock:
                    if self.speaker not in self.state["known_users"]:
                        self.state["known_users"][self.speaker] = {
                            "first_seen": datetime.now().isoformat(),
                            "interaction_count": 0,
                            "preferences": {},
                            "topics_discussed": [],
                            "sentiment_history": []
                        }
                    
                    user_data = self.state["known_users"][self.speaker]
                    user_data["interaction_count"] += 1
                    user_data["last_seen"] = datetime.now().isoformat()
                
                if old_speaker != self.speaker:
                    logger.info(f"Identity detected: {self.speaker} (confidence: {result['confidence']:.2f})")
                
                return True
        
        except Exception as e:
            logger.error(f"Error in identity detection: {e}")
        
        return False
    
    def extract_and_validate_facts(self, message: str) -> Optional[Memory]:
        """Extract facts with validation and confidence scoring"""
        prompt = [
            {"role": "system", "content": """Extract factual information from the message. 
            Return only valid JSON: {
                "has_fact": true/false,
                "fact": "FACTUAL_STATEMENT",
                "confidence": 0.0-1.0,
                "category": "personal/general/technical/emotional"
            }"""},
            {"role": "user", "content": message}
        ]
        
        default_response = {"has_fact": False, "fact": "", "confidence": 0.0, "category": "general"}
        
        try:
            result = safe_json_request(self.client, prompt, default_response)
            
            if (result.get("has_fact") and 
                result.get("confidence", 0) > self.config.confidence_threshold):
                
                return Memory(
                    id=self.generate_memory_id(result["fact"]),
                    timestamp=datetime.now().isoformat(),
                    speaker=self.speaker,
                    type=MemoryType.FACT,
                    content=result["fact"],
                    metadata={
                        "confidence": result["confidence"],
                        "category": result["category"],
                        "extracted_from": message[:100]
                    }
                )
        
        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
        
        return None
    
    def detect_emotion_nuanced(self, message: str) -> Optional[Memory]:
        """Detect emotion with nuanced understanding"""
        prompt = [
            {"role": "system", "content": """Analyze the emotional content of this message. 
            Return only valid JSON: {
                "primary_emotion": "emotion_name",
                "secondary_emotions": [],
                "intensity": 0.0-1.0,
                "sentiment": -1.0,
                "needs_support": false
            }"""},
            {"role": "user", "content": message}
        ]
        
        default_response = {
            "primary_emotion": "neutral",
            "secondary_emotions": [],
            "intensity": 0.5,
            "sentiment": 0.0,
            "needs_support": False
        }
        
        try:
            result = safe_json_request(self.client, prompt, default_response)
            
            # Store emotion if significant
            if result.get("intensity", 0) > 0.3 or result.get("needs_support", False):
                return Memory(
                    id=self.generate_memory_id(f"emotion_{message}"),
                    timestamp=datetime.now().isoformat(),
                    speaker=self.speaker,
                    type=MemoryType.EMOTION,
                    content=f"Detected {result['primary_emotion']} (intensity: {result['intensity']:.2f})",
                    metadata=result
                )
        
        except Exception as e:
            logger.error(f"Error detecting emotion: {e}")
        
        return None
    
    def needs_realtime_info(self, message: str) -> bool:
        """Determine if query needs real-time information"""
        realtime_indicators = [
            "current", "today", "now", "latest", "recent", "this week", "this month",
            "weather", "news", "price", "stock", "update", "happening",
            "bitcoin", "crypto", "market", "breaking", "live"
        ]
        
        message_lower = message.lower()
        
        # Check for explicit time references
        time_patterns = [
            r"today", r"right now", r"currently", r"at the moment",
            r"this \w+", r"latest \w+", r"recent \w+"
        ]
        
        import re
        for pattern in time_patterns:
            if re.search(pattern, message_lower):
                return True
        
        # Check for indicator words
        return any(indicator in message_lower for indicator in realtime_indicators)
    
    def handle_realtime_query(self, user_input: str) -> Optional[str]:
        """Handle queries requiring real-time information with error handling"""
        try:
            # Generate optimized search query
            query_prompt = [
                {"role": "system", "content": "Convert this question into an optimal web search query (3-6 words):"},
                {"role": "user", "content": user_input}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=query_prompt,
                timeout=10
            )
            
            search_query = response.choices[0].message.content.strip()
            logger.info(f"Searching for: {search_query}")
            
            # Perform web search with error handling
            search_results = search_web(search_query, count=self.config.max_search_results)
            
            if search_results and search_results[0] != "No useful search results found.":
                # Synthesize answer from results
                synthesis_prompt = [
                    {"role": "system", "content": """Create a helpful, accurate answer based on these search results. 
                    Be conversational and cite key information. If the results are unclear or contradictory, mention that."""},
                    {"role": "user", "content": f"Question: {user_input}\n\nSearch results:\n" + "\n".join(search_results[:5])}
                ]
                
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=synthesis_prompt,
                    timeout=15
                )
                
                answer = response.choices[0].message.content.strip()
                
                # Store web result with proper memory management
                with self._lock:
                    web_memory = Memory(
                        id=self.generate_memory_id(answer),
                        timestamp=datetime.now().isoformat(),
                        speaker="nova",
                        type=MemoryType.WEB_RESULT,
                        content=answer,
                        metadata={
                            "query": search_query,
                            "original_question": user_input,
                            "sources": search_results[:3],
                            "search_time": datetime.now().isoformat()
                        }
                    )
                    self.brain.append(web_memory)
                    
                    # Store in vector memory if available
                    if self.vector_memory:
                        self.vector_memory.store(answer, web_memory.id, MemoryType.WEB_RESULT)
                
                with self._metrics_lock:
                    self.metrics.successful_queries += 1
                
                return answer
            else:
                logger.warning(f"No useful search results for: {search_query}")
                
        except Exception as e:
            logger.error(f"Error handling realtime query: {e}", exc_info=True)
            with self._metrics_lock:
                self.metrics.failed_queries += 1
        
        return None
    
    def retrieve_relevant_memories(self, query: str, k: int = 5) -> List[Memory]:
        """Retrieve relevant memories with enhanced scoring and error handling"""
        try:
            relevant_memories = []
            seen_ids = set()
            
            # Get vector search results if available
            if self.vector_memory:
                try:
                    vector_results = self.vector_memory.retrieve(
                        query, 
                        k=k*2, 
                        threshold=self.config.vector_similarity_threshold
                    )
                    
                    # Convert vector results to Memory objects
                    for result in vector_results:
                        # Find corresponding memory in brain
                        with self._lock:
                            for memory in self.brain:
                                if (memory.content == result.get("content") and 
                                    memory.id not in seen_ids):
                                    relevant_memories.append(memory)
                                    seen_ids.add(memory.id)
                                    break
                except Exception as e:
                    logger.error(f"Error in vector search: {e}")
            
            # Also search recent memories by keywords
            query_words = set(query.lower().split())
            keyword_memories = []
            
            with self._lock:
                recent_memories = self.brain[-100:] if len(self.brain) > 100 else self.brain
                
                for memory in reversed(recent_memories):
                    if memory.id in seen_ids:
                        continue
                    
                    try:
                        memory_words = set(memory.content.lower().split())
                        overlap = len(query_words & memory_words)
                        if overlap > 1:
                            keyword_memories.append((overlap, memory))
                    except Exception as e:
                        logger.warning(f"Error processing memory {memory.id}: {e}")
                        continue
            
            # Add best keyword matches
            keyword_memories.sort(key=lambda x: x[0], reverse=True)
            for score, memory in keyword_memories[:k//2]:
                if memory.id not in seen_ids:
                    relevant_memories.append(memory)
                    seen_ids.add(memory.id)
            
            return relevant_memories[:k]
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}", exc_info=True)
            return []
    
    def generate_contextual_response(self, user_input: str) -> str:
        """Generate response with full context awareness and error handling"""
        try:
            # Build comprehensive context
            relevant_memories = self.retrieve_relevant_memories(user_input)
            
            with self._lock:
                user_profile = self.state["known_users"].get(self.speaker, {})
            
            context_messages = [self.system_prompt]
            
            # Add user profile context
            if user_profile:
                profile_summary = {
                    "interaction_count": user_profile.get("interaction_count", 0),
                    "preferences": user_profile.get("preferences", {}),
                    "recent_topics": user_profile.get("topics_discussed", [])[-5:]
                }
                context_messages.append({
                    "role": "system",
                    "content": f"User profile: {json.dumps(profile_summary)}"
                })
            
            # Add relevant memories
            for memory in relevant_memories[:3]:
                context_messages.append({
                    "role": "system",
                    "content": f"Relevant memory ({memory.type}): {memory.content}"
                })
            
            # Add recent conversation context
            context_messages.extend(self.context_window[-5:])
            
            # Add current input
            context_messages.append({"role": "user", "content": user_input})
            
            # Determine appropriate model and temperature
            model = "gpt-4-turbo" if len(user_input) > 100 or "analyze" in user_input.lower() else "gpt-4"
            temperature = 0.8 if any(word in user_input.lower() for word in ["creative", "story", "imagine"]) else 0.7
            
            # Generate response with timeout
            response = self.client.chat.completions.create(
                model=model,
                messages=context_messages,
                temperature=temperature,
                max_tokens=800,
                timeout=30
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            with self._metrics_lock:
                self.metrics.failed_queries += 1
            return "I encountered an error while processing your request. Let me try to help you in a different way."
    
    async def autonomous_learning_cycle(self):
        """Enhanced asynchronous autonomous learning cycle with proper error handling"""
        logger.info("Starting autonomous learning cycle")
        
        while not self._shutdown_event.is_set():
            try:
                # Check if we should continue
                if self._shutdown_event.wait(300):  # Wait 5 minutes or until shutdown
                    break
                
                # Analyze recent patterns
                patterns = self.analyze_conversation_patterns()
                
                if patterns.get("areas_for_improvement"):
                    for area in patterns["areas_for_improvement"][:2]:  # Limit to 2 areas
                        try:
                            # Research the improvement area
                            search_results = search_web(f"how to improve {area} in AI conversation")
                            if search_results and search_results[0] != "No useful search results found.":
                                knowledge = extract_plain_answer(search_results)
                                
                                if knowledge:
                                    with self._lock:
                                        learning_memory = Memory(
                                            id=self.generate_memory_id(knowledge),
                                            timestamp=datetime.now().isoformat(),
                                            speaker="nova",
                                            type=MemoryType.AUTONOMOUS_LEARNING,
                                            content=f"Learned about improving {area}: {knowledge}",
                                            metadata={
                                                "source": "autonomous_learning",
                                                "improvement_area": area,
                                                "research_query": f"how to improve {area} in AI conversation"
                                            }
                                        )
                                        self.brain.append(learning_memory)
                                        
                                        if self.vector_memory:
                                            self.vector_memory.store(learning_memory.content, learning_memory.id, learning_memory.type)
                                    
                                    with self._metrics_lock:
                                        self.metrics.learning_events += 1
                                    
                        except Exception as e:
                            logger.error(f"Error researching improvement area {area}: {e}")
                
                # Consolidate memories if needed
                with self._lock:
                    if len(self.brain) > 500:
                        self.consolidate_memories()
                
                # Save state periodically
                self.save_brain()
                self.save_state()
                
            except Exception as e:
                logger.error(f"Error in autonomous learning cycle: {e}", exc_info=True)
                # Wait before retry
                if not self._shutdown_event.wait(60):
                    continue
                else:
                    break
        
        logger.info("Autonomous learning cycle stopped")
    
    def consolidate_memories(self):
        """Consolidate and organize memories periodically with better error handling"""
        if len(self.brain) < 100:
            return
        
        try:
            # Group memories by type and speaker
            memory_groups = {}
            for memory in self.brain:
                key = f"{memory.speaker}_{memory.type}"
                if key not in memory_groups:
                    memory_groups[key] = []
                memory_groups[key].append(memory)
            
            consolidated_count = 0
            
            # Summarize old conversations
            for key, memories in memory_groups.items():
                if len(memories) > 30 and "conversation" in key:
                    try:
                        # Take oldest 15 memories for consolidation
                        old_memories = sorted(memories, key=lambda m: m.timestamp)[:15]
                        
                        content = "\n".join([f"{m.speaker}: {m.content}" for m in old_memories])
                        summary_prompt = [
                            {"role": "system", "content": """Create a concise summary of these conversation points, 
                            preserving key facts, emotional context, and important topics discussed."""},
                            {"role": "user", "content": content}
                        ]
                        
                        response = self.client.chat.completions.create(
                            model="gpt-4",
                            messages=summary_prompt,
                            timeout=20
                        )
                        
                        summary = Memory(
                            id=self.generate_memory_id(response.choices[0].message.content),
                            timestamp=datetime.now().isoformat(),
                            speaker="nova",
                            type="consolidated_summary",
                            content=response.choices[0].message.content,
                            metadata={
                                "original_count": len(old_memories),
                                "time_range": f"{old_memories[0].timestamp} to {old_memories[-1].timestamp}",
                                "speakers": list(set(m.speaker for m in old_memories))
                            }
                        )
                        
                        # Remove old memories and add summary
                        self.brain = [m for m in self.brain if m not in old_memories]
                        self.brain.append(summary)
                        
                        if self.vector_memory:
                            self.vector_memory.store(summary.content, summary.id, summary.type)
                        
                        consolidated_count += len(old_memories)
                        
                    except Exception as e:
                        logger.error(f"Error consolidating memories for {key}: {e}")
            
            if consolidated_count > 0:
                logger.info(f"Consolidated {consolidated_count} memories")
            
        except Exception as e:
            logger.error(f"Error in memory consolidation: {e}", exc_info=True)
    
    def chat(self, user_input: str, structured: bool = False) -> Union[str, Dict[str, Any]]:
        """Enhanced chat method with comprehensive processing and error handling"""
        if self._shutdown_event.is_set():
            return "System is shutting down, please wait..."
            
        start_time = datetime.now()
        
        try:
            # Handle autonomous mode
            if user_input == "__autonomous__":
                return self.autonomy.meta_thought()
            
            # Update counters
            self.exchange_count.increment()
            with self._metrics_lock:
                self.metrics.total_conversations += 1
            
            # Detect user identity
            if self.speaker == "unknown":
                self.detect_identity(user_input)
            
            # Store conversation with error handling
            try:
                with self._lock:
                    conv_memory = Memory(
                        id=self.generate_memory_id(user_input),
                        timestamp=datetime.now().isoformat(),
                        speaker=self.speaker,
                        type=MemoryType.CONVERSATION,
                        content=user_input,
                        metadata={
                            "exchange_count": self.exchange_count.get(),
                            "session_time": (datetime.now() - start_time).total_seconds()
                        }
                    )
                    self.brain.append(conv_memory)
                    
                    if self.vector_memory:
                        self.vector_memory.store(user_input, conv_memory.id, conv_memory.type)
            except Exception as e:
                logger.error(f"Error storing conversation memory: {e}")
            
            # Update context window safely
            try:
                self.context_window.append({"role": "user", "content": user_input})
                if len(self.context_window) > self.config.max_context_size * 2:
                    self.context_window = self.context_window[-self.config.max_context_size:]
            except Exception as e:
                logger.error(f"Error updating context window: {e}")
            
            # Extract and store facts
            try:
                fact = self.extract_and_validate_facts(user_input)
                if fact:
                    with self._lock:
                        self.brain.append(fact)
                        if self.vector_memory:
                            self.vector_memory.store(fact.content, fact.id, fact.type)
            except Exception as e:
                logger.error(f"Error extracting facts: {e}")
            
            # Detect and store emotions
            try:
                emotion = self.detect_emotion_nuanced(user_input)
                if emotion:
                    with self._lock:
                        self.brain.append(emotion)
                        # Update user profile with emotion
                        if self.speaker != "unknown":
                            user_data = self.state["known_users"].get(self.speaker, {})
                            if "sentiment_history" not in user_data:
                                user_data["sentiment_history"] = []
                            user_data["sentiment_history"].append({
                                "timestamp": emotion.timestamp,
                                "emotion": emotion.metadata.get("primary_emotion"),
                                "sentiment": emotion.metadata.get("sentiment", 0)
                            })
                            # Keep only recent sentiment history
                            if len(user_data["sentiment_history"]) > 10:
                                user_data["sentiment_history"] = user_data["sentiment_history"][-10:]
            except Exception as e:
                logger.error(f"Error detecting emotion: {e}")
            
            # Check for real-time information need
            response = None
            if self.needs_realtime_info(user_input):
                search_result = self.handle_realtime_query(user_input)
                if search_result:
                    response = search_result
            
            if not response:
                response = self.generate_contextual_response(user_input)
            
            # Store response with error handling
            try:
                with self._lock:
                    response_memory = Memory(
                        id=self.generate_memory_id(response),
                        timestamp=datetime.now().isoformat(),
                        speaker="nova",
                        type=MemoryType.RESPONSE,
                        content=response,
                        metadata={
                            "in_response_to": user_input[:100],
                            "response_time": (datetime.now() - start_time).total_seconds(),
                            "context_used": len(self.context_window)
                        }
                    )
                    self.brain.append(response_memory)
                    
                    if self.vector_memory:
                        self.vector_memory.store(response, response_memory.id, response_memory.type)
                
                # Update context window
                self.context_window.append({"role": "assistant", "content": response})
            except Exception as e:
                logger.error(f"Error storing response memory: {e}")
            
            # Self-improvement check
            try:
                self.check_self_improvement(response, user_input)
            except Exception as e:
                logger.error(f"Error in self-improvement check: {e}")
            
            # Update metrics
            with self._metrics_lock:
                self.metrics.successful_queries += 1
                response_time = (datetime.now() - start_time).total_seconds()
                if self.metrics.successful_queries > 1:
                    self.metrics.avg_response_time = (
                        (self.metrics.avg_response_time * (self.metrics.successful_queries - 1) + response_time) /
                        self.metrics.successful_queries
                    )
                else:
                    self.metrics.avg_response_time = response_time
            
            # Save periodically
            try:
                if (self.exchange_count.get() % 5 == 0 or 
                    (datetime.now() - self.last_save_time).total_seconds() > 300):
                    self.save_brain()
                    self.save_state()
            except Exception as e:
                logger.error(f"Error during periodic save: {e}")
            
            # Update user profile with topics
            try:
                if self.speaker != "unknown":
                    self._update_user_topics(user_input)
            except Exception as e:
                logger.error(f"Error updating user topics: {e}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat processing: {e}", exc_info=True)
            with self._metrics_lock:
                self.metrics.failed_queries += 1
            return f"I encountered an error while processing your message: {str(e)}. Please try again."
    
    def analyze_conversation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in conversations for self-improvement"""
        try:
            with self._lock:
                recent_conversations = [
                    m for m in self.brain[-100:] 
                    if m.type in [MemoryType.CONVERSATION, MemoryType.RESPONSE]
                ]
            
            if len(recent_conversations) < 10:
                return {}
            
            analysis_prompt = [
                {"role": "system", "content": """Analyze these conversation patterns and return only valid JSON:
                {
                    "common_topics": ["topic1", "topic2"],
                    "user_satisfaction": 0.5,
                    "areas_for_improvement": ["area1", "area2"],
                    "successful_patterns": ["pattern1", "pattern2"],
                    "conversation_style": "helpful/analytical/creative",
                    "engagement_level": 0.5
                }"""},
                {"role": "user", "content": "\n".join([
                    f"{m.speaker}: {m.content}" for m in recent_conversations[-20:]
                ])}
            ]
            
            default_response = {
                "common_topics": [],
                "user_satisfaction": 0.5,
                "areas_for_improvement": [],
                "successful_patterns": [],
                "conversation_style": "helpful",
                "engagement_level": 0.5
            }
            
            return safe_json_request(self.client, analysis_prompt, default_response)
            
        except Exception as e:
            logger.error(f"Error analyzing conversation patterns: {e}")
            return {}
    
    def check_self_improvement(self, response: str, user_input: str):
        """Check if self-improvement is needed based on response quality"""
        try:
            evaluation_prompt = [
                {"role": "system", "content": """Evaluate this AI response quality. Return only valid JSON:
                {
                    "quality_score": 0.7,
                    "helpfulness": 0.8,
                    "accuracy": 0.9,
                    "clarity": 0.7,
                    "needs_improvement": false,
                    "improvement_areas": [],
                    "response_type": "informative"
                }"""},
                {"role": "user", "content": f"User asked: {user_input}\n\nAI responded: {response}"}
            ]
            
            default_response = {
                "quality_score": 0.7,
                "helpfulness": 0.7,
                "accuracy": 0.7,
                "clarity": 0.7,
                "needs_improvement": False,
                "improvement_areas": [],
                "response_type": "general"
            }
            
            evaluation = safe_json_request(self.client, evaluation_prompt, default_response)
            
            # Store evaluation
            with self._lock:
                eval_memory = Memory(
                    id=self.generate_memory_id(f"eval_{response}"),
                    timestamp=datetime.now().isoformat(),
                    speaker="nova",
                    type=MemoryType.SELF_EVALUATION,
                    content=f"Response evaluation: Quality {evaluation.get('quality_score', 0.7):.2f}",
                    metadata=evaluation
                )
                self.brain.append(eval_memory)
            
            # Check if improvement is needed
            if (evaluation.get("needs_improvement") and 
                evaluation.get("quality_score", 1.0) < 0.6):
                
                improvement_areas = evaluation.get("improvement_areas", [])
                if improvement_areas and hasattr(self, 'autonomy'):
                    try:
                        self.autonomy.add_improvement_goal(improvement_areas[0])
                        with self._metrics_lock:
                            self.metrics.self_modifications += 1
                    except Exception as e:
                        logger.error(f"Error adding improvement goal: {e}")
            
        except Exception as e:
            logger.error(f"Error in self-improvement check: {e}")
    
    def _update_user_topics(self, user_input: str):
        """Update user profile with discussed topics"""
        try:
            with self._lock:
                user_data = self.state["known_users"].get(self.speaker, {})
                if "topics_discussed" not in user_data:
                    user_data["topics_discussed"] = []
                
                # Extract topic using simple keyword extraction
                words = user_input.lower().split()
                # Filter out common words
                stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
                meaningful_words = [w for w in words if w not in stop_words and len(w) > 3]
                
                if meaningful_words:
                    # Take the first meaningful word as topic indicator
                    topic = meaningful_words[0]
                    user_data["topics_discussed"].append({
                        "topic": topic,
                        "timestamp": datetime.now().isoformat(),
                        "context": user_input[:50]
                    })
                    
                    # Keep only recent topics
                    if len(user_data["topics_discussed"]) > 20:
                        user_data["topics_discussed"] = user_data["topics_discussed"][-20:]
            
        except Exception as e:
            logger.error(f"Error updating user topics: {e}")
    
    def export_knowledge_graph(self) -> Dict[str, Any]:
        """Export knowledge as a graph structure for analysis"""
        try:
            with self._lock:
                graph = {
                    "nodes": [],
                    "edges": [],
                    "metadata": {
                        "total_memories": len(self.brain),
                        "known_users": list(self.state["known_users"].keys()),
                        "export_time": datetime.now().isoformat(),
                        "agent_metrics": self.metrics.to_dict()
                    }
                }
                
                # Create nodes for each memory
                for memory in self.brain:
                    graph["nodes"].append({
                        "id": memory.id,
                        "type": memory.type,
                        "speaker": memory.speaker,
                        "content": memory.content[:200],  # Truncate for visualization
                        "timestamp": memory.timestamp,
                        "metadata": memory.metadata or {}
                    })
                
                # Create edges based on relationships
                for i, mem1 in enumerate(self.brain):
                    # Connect memories from same conversation
                    for mem2 in self.brain[max(0, i-3):i+4]:  # Check nearby memories
                        if (mem1.id != mem2.id and 
                            mem1.speaker == mem2.speaker and 
                            abs(datetime.fromisoformat(mem1.timestamp).timestamp() - 
                                datetime.fromisoformat(mem2.timestamp).timestamp()) < 3600):  # Within 1 hour
                            
                            graph["edges"].append({
                                "source": mem1.id,
                                "target": mem2.id,
                                "type": "temporal",
                                "weight": 0.5
                            })
                    
                    # Connect memories of same type
                    for mem2 in self.brain[i+1:i+6]:  # Check next 5 memories
                        if mem1.type == mem2.type and mem1.speaker == mem2.speaker:
                            graph["edges"].append({
                                "source": mem1.id,
                                "target": mem2.id,
                                "type": "same_type",
                                "weight": 0.3
                            })
                
                return graph
                
        except Exception as e:
            logger.error(f"Error exporting knowledge graph: {e}")
            return {"error": str(e)}
    
    def get_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        try:
            with self._lock:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "agent_info": {
                        "speaker": self.speaker,
                        "exchange_count": self.exchange_count.get(),
                        "total_memories": len(self.brain),
                        "context_window_size": len(self.context_window)
                    },
                    "metrics": self.metrics.to_dict(),
                    "memory_stats": {
                        "by_type": self._get_memory_type_counts(),
                        "recent_count": len([m for m in self.brain if m.is_recent(24)]),
                        "vector_memory_stats": self.vector_memory.get_memory_stats() if self.vector_memory else {}
                    },
                    "autonomy_status": {
                        "active_goals": len([g for g in getattr(self.autonomy, 'current_goals', []) if g.get("status") == "active"]),
                        "total_insights": len(getattr(self.autonomy, 'insights', [])),
                        "last_action": getattr(self.autonomy, 'last_action', 'none'),
                        "curiosity_level": getattr(self.autonomy, 'curiosity_level', 0.5)
                    },
                    "user_profiles": {
                        username: {
                            "interaction_count": data.get("interaction_count", 0),
                            "last_seen": data.get("last_seen", "unknown"),
                            "topics_count": len(data.get("topics_discussed", []))
                        }
                        for username, data in self.state["known_users"].items()
                    },
                    "system_health": {
                        "last_save": self.last_save_time.isoformat(),
                        "config_valid": len(getattr(self.config, 'validate', lambda: [])()) == 0,
                        "components_active": {
                            "vector_memory": self.vector_memory is not None,
                            "autonomy": hasattr(self, 'autonomy') and self.autonomy is not None,
                            "self_modifier": hasattr(self, 'self_modifier') and self.self_modifier is not None
                        }
                    }
                }
        except Exception as e:
            logger.error(f"Error generating status report: {e}")
            return {"error": str(e)}
    
    def _get_memory_type_counts(self) -> Dict[str, int]:
        """Get count of memories by type"""
        counts = {}
        try:
            for memory in self.brain:
                counts[memory.type] = counts.get(memory.type, 0) + 1
        except Exception as e:
            logger.error(f"Error counting memory types: {e}")
        return counts
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data while preserving important memories"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self._lock:
                # Clean up old conversation memories (keep facts, insights, etc.)
                before_count = len(self.brain)
                self.brain = [
                    memory for memory in self.brain
                    if (datetime.fromisoformat(memory.timestamp) > cutoff_date or
                        memory.type in [MemoryType.FACT, MemoryType.INSIGHT, 
                                       MemoryType.GOAL, MemoryType.PATTERN_ANALYSIS] or
                        (memory.metadata and memory.metadata.get("important", False)))
                ]
                after_count = len(self.brain)
            
            # Clean up vector memory
            deleted_vector_count = 0
            if self.vector_memory:
                deleted_vector_count = self.vector_memory.cleanup_old_memories(days)
            
            logger.info(f"Cleaned up {before_count - after_count} old memories and {deleted_vector_count} vector memories")
            
            return {
                "memories_deleted": before_count - after_count,
                "vector_memories_deleted": deleted_vector_count,
                "remaining_memories": after_count
            }
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {"error": str(e)}
    
    def shutdown(self):
        """Graceful shutdown of the agent"""
        logger.info("Shutting down Nova agent...")
        self._shutdown_event.set()
        
        try:
            # Save current state
            self.save_brain()
            self.save_state()
            
            # Close vector memory
            if self.vector_memory and hasattr(self.vector_memory, 'close'):
                self.vector_memory.close()
            
            logger.info("Nova agent shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if hasattr(self, '_shutdown_event') and not self._shutdown_event.is_set():
                self.shutdown()
        except Exception as e:
            logger.error(f"Error in destructor: {e}")