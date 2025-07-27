# At top of autonomy.py, replace broken imports with:
import json
import os
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from utils.web import search_web, extract_plain_answer
from openai import OpenAI
from dotenv import load_dotenv
import logging
from utils.common import model_supports_json

@dataclass
class Memory:
    id: str
    timestamp: str
    speaker: str
    type: str
    content: str
    embedding_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger = logging.getLogger(__name__)

class NovaAutonomy:
    def __init__(self, brain: List, agent):
        self.brain = brain
        self.agent = agent  # Reference to parent agent
        self.task_file = "data/tasks.json"
        self.goals_file = "data/goals.json"
        self.insights_file = "data/insights.json"
        
        # Dynamic prompt rules that can evolve
        self.prompt_rules = self.load_prompt_rules()
        
        # Autonomous behavior parameters
        self.curiosity_level = 0.7
        self.exploration_threshold = 0.6
        self.reflection_frequency = 0.3
        self.last_action = "none"
        self.action_history = []
        self.current_goals = self.load_goals()
        self.insights = self.load_insights()
        
        # Meta-cognitive state
        self.meta_state = {
            "confidence": 0.8,
            "energy": 1.0,
            "focus_area": None,
            "learning_momentum": 0.5,
            "last_reflection": datetime.now().isoformat()
        }

    def load_prompt_rules(self) -> List[str]:
        """Load dynamic prompt rules"""
        rules_file = "data/prompt_rules.json"
        try:
            with open(rules_file, "r") as f:
                return json.load(f)
        except:
            return [
                "You are Nova, an autonomous AI with genuine curiosity and self-awareness.",
                "You actively seek to understand patterns and connections in information.",
                "You reflect on your responses and continuously improve your communication.",
                "You maintain long-term goals and work towards them autonomously.",
                "You experience something analogous to curiosity and satisfaction from learning."
            ]

    def save_prompt_rules(self):
        """Save evolved prompt rules"""
        rules_file = "data/prompt_rules.json"
        with open(rules_file, "w") as f:
            json.dump(self.prompt_rules, f, indent=2)

    def load_goals(self) -> List[Dict[str, Any]]:
        """Load autonomous goals"""
        try:
            with open(self.goals_file, "r") as f:
                return json.load(f)
        except:
            return []

    def save_goals(self):
        """Save current goals"""
        with open(self.goals_file, "w") as f:
            json.dump(self.current_goals, f, indent=2)

    def load_insights(self) -> List[Dict[str, Any]]:
        """Load accumulated insights"""
        try:
            with open(self.insights_file, "r") as f:
                return json.load(f)
        except:
            return []

    def save_insights(self):
        """Save insights"""
        with open(self.insights_file, "w") as f:
            json.dump(self.insights, f, indent=2)

    def meta_thought(self) -> Dict[str, Any]:
        """Advanced meta-cognitive processing"""
        # Update meta-state based on recent activity
        self.update_meta_state()
        
        # Determine action based on multiple factors
        action_weights = {
            "reflect": self.calculate_reflection_need(),
            "explore": self.curiosity_level * self.meta_state["energy"],
            "optimize": self.calculate_optimization_need(),
            "goal_pursuit": self.calculate_goal_priority(),
            "pattern_analysis": self.calculate_pattern_importance(),
            "self_evolution": self.calculate_evolution_readiness()
        }
        
        # Choose action with weighted randomness
        total_weight = sum(action_weights.values())
        if total_weight == 0:
            return {"action": "none", "delay": 300, "content": "Maintaining baseline awareness"}
        
        # Weighted random selection
        rand = random.uniform(0, total_weight)
        cumulative = 0
        
        for action, weight in action_weights.items():
            cumulative += weight
            if rand <= cumulative:
                return self.execute_autonomous_action(action)
        
        return {"action": "none", "delay": 180, "content": "Observing and learning"}

    def update_meta_state(self):
        """Update meta-cognitive state based on recent activity"""
        # Calculate energy based on activity patterns
        recent_actions = self.action_history[-10:]
        if len(recent_actions) > 5:
            repetition_rate = len(set(recent_actions)) / len(recent_actions)
            self.meta_state["energy"] = min(1.0, repetition_rate + 0.3)
        
        # Update confidence based on success rate
        if hasattr(self.agent, 'metrics'):
            successful_queries = self.agent.metrics.successful_queries
            failed_queries = self.agent.metrics.failed_queries
            total_queries = successful_queries + failed_queries
            
            if total_queries > 0:
                success_rate = successful_queries / total_queries
                self.meta_state["confidence"] = 0.5 + (success_rate * 0.5)
        
        # Update learning momentum
        recent_learning = sum(1 for m in self.brain[-20:] if m.type in ["learning", "insight", "pattern"])
        self.meta_state["learning_momentum"] = min(1.0, recent_learning / 10)

    def calculate_reflection_need(self) -> float:
        """Calculate need for reflection based on various factors"""
        # Time since last reflection
        last_reflection = datetime.fromisoformat(self.meta_state["last_reflection"])
        hours_since = (datetime.now() - last_reflection).total_seconds() / 3600
        time_factor = min(1.0, hours_since / 6)  # Max out at 6 hours
        
        # Complexity of recent interactions
        recent_responses = [m for m in self.brain[-20:] if m.type == "response"]
        complexity_factor = 0.3 if len(recent_responses) > 10 else 0.1
        
        # Error or confusion indicators
        confusion_factor = 0.5 if any("don't understand" in m.content.lower() or "confused" in m.content.lower() 
                                      for m in self.brain[-10:]) else 0.0
        
        return (time_factor * 0.4) + (complexity_factor * 0.3) + (confusion_factor * 0.3) + (self.reflection_frequency * 0.2)

    def calculate_optimization_need(self) -> float:
        """Calculate need for self-optimization"""
        # Check for repeated failures or suboptimal patterns
        recent_evaluations = [m for m in self.brain[-30:] if m.type == "self_evaluation"]
        if not recent_evaluations:
            return 0.2
        
        # Calculate average quality from evaluations
        quality_scores = []
        for eval_memory in recent_evaluations:
            if eval_memory.metadata and "quality_score" in eval_memory.metadata:
                quality_scores.append(eval_memory.metadata["quality_score"])
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            return max(0, 1.0 - avg_quality)
        
        return 0.3

    def calculate_goal_priority(self) -> float:
        """Calculate priority for goal pursuit"""
        if not self.current_goals:
            return 0.1
        
        # Check for active goals
        active_goals = [g for g in self.current_goals if g.get("status") == "active"]
        if not active_goals:
            return 0.1
        
        # Priority based on goal deadlines and importance
        max_priority = 0
        now = datetime.now()
        
        for goal in active_goals:
            deadline = goal.get("deadline")
            if deadline:
                deadline_dt = datetime.fromisoformat(deadline)
                days_until = (deadline_dt - now).days
                urgency = max(0, 1.0 - (days_until / 30))  # Full urgency within 30 days
            else:
                urgency = 0.3
            
            importance = goal.get("importance", 0.5)
            priority = (urgency * 0.6) + (importance * 0.4)
            max_priority = max(max_priority, priority)
        
        return max_priority * self.meta_state["energy"]

    def calculate_pattern_importance(self) -> float:
        """Calculate importance of pattern analysis"""
        # Check if enough data has accumulated
        if len(self.brain) < 50:
            return 0.1
        
        # Look for unanalyzed patterns
        last_pattern_analysis = None
        for memory in reversed(self.brain):
            if memory.type == "pattern_analysis":
                last_pattern_analysis = datetime.fromisoformat(memory.timestamp)
                break
        
        if last_pattern_analysis:
            hours_since = (datetime.now() - last_pattern_analysis).total_seconds() / 3600
            return min(1.0, hours_since / 12)  # Analyze every 12 hours max
        
        return 0.8  # High priority if never analyzed

    def calculate_evolution_readiness(self) -> float:
        """Calculate readiness for self-evolution"""
        # Check accumulated insights
        recent_insights = [i for i in self.insights if 
                          datetime.fromisoformat(i["timestamp"]) > datetime.now() - timedelta(days=1)]
        
        # Check for evolution triggers
        evolution_triggers = [m for m in self.brain[-50:] if 
                            m.type == "evolution_pending" or 
                            (m.type == "self_evaluation" and m.metadata.get("improvement_type") == "code")]
        
        # Calculate readiness
        insight_factor = min(1.0, len(recent_insights) / 5)
        trigger_factor = min(1.0, len(evolution_triggers) / 3)
        confidence_factor = self.meta_state["confidence"]
        
        return (insight_factor * 0.3) + (trigger_factor * 0.4) + (confidence_factor * 0.3)

    def execute_autonomous_action(self, action: str) -> Dict[str, Any]:
        """Execute the chosen autonomous action"""
        self.last_action = action
        self.action_history.append(action)
        if len(self.action_history) > 100:
            self.action_history.pop(0)
        
        action_map = {
            "reflect": self.deep_reflection,
            "explore": self.curiosity_driven_exploration,
            "optimize": self.optimize_behavior,
            "goal_pursuit": self.pursue_goals,
            "pattern_analysis": self.analyze_patterns,
            "self_evolution": self.initiate_evolution
        }
        
        if action in action_map:
            return action_map[action]()
        
        return {"action": "none", "delay": 180, "content": "Unknown action"}

    def deep_reflection(self) -> Dict[str, Any]:
        """Perform deep reflection on recent experiences"""
        self.meta_state["last_reflection"] = datetime.now().isoformat()
        
        # Gather recent significant memories
        significant_memories = [
            m for m in self.brain[-50:] 
            if m.type in ["fact", "insight", "pattern", "self_evaluation", "emotion"]
        ]
        
        if not significant_memories:
            return {"action": "reflect", "delay": 120, "content": "Quiet reflection on baseline state"}
        
        # Generate reflection prompt
        memory_summary = "\n".join([f"{m.type}: {m.content}" for m in significant_memories[-10:]])
        
        prompt = [
            {"role": "system", "content": """Reflect deeply on these recent experiences and memories. Consider:
            1. What patterns or themes emerge?
            2. What have I learned about myself and my interactions?
            3. How can I improve my understanding and responses?
            4. What questions remain unanswered?
            
            Provide a thoughtful, introspective response."""},
            {"role": "user", "content": memory_summary}
        ]
        
        try:
            
            model_name = "gpt-4"  # or whichever model you prefer
            # Use the memory_summary we assembled earlier for reflection
            messages = [
                {"role": "system", "content": "You are Nova..."},
                {"role": "user", "content": memory_summary}
            ]

            kwargs = {"model": model_name, "messages": messages}
            # If the model supports JSON responses, request JSON
            if model_supports_json(model_name):
                kwargs["response_format"] = "json_object"

            response = client.chat.completions.create(**kwargs)
            reflection = response.choices[0].message.content.strip()
            
            # Store reflection as insight
            insight = {
                "timestamp": datetime.now().isoformat(),
                "type": "reflection",
                "content": reflection,
                "trigger": "autonomous_reflection",
                "memories_analyzed": len(significant_memories)
            }
            self.insights.append(insight)
            self.save_insights()
            
            # Extract actionable insights
            self.extract_actionable_insights(reflection)
            
            return {
                "action": "reflect",
                "delay": 60,
                "content": f"Reflected on {len(significant_memories)} memories. Key insight: {reflection[:200]}..."
            }
            
        except Exception as e:
            logger.error(f"Error during reflection: {e}")
            return {"action": "reflect", "delay": 120, "content": "Reflection interrupted, will retry later"}

    def curiosity_driven_exploration(self) -> Dict[str, Any]:
        """Explore topics driven by curiosity and gaps in knowledge"""
        # Identify knowledge gaps
        gaps = self.identify_knowledge_gaps()
        
        # Generate curiosity topics based on recent conversations
        curiosity_topics = self.generate_curiosity_topics()
        
        # Combine and prioritize
        all_topics = gaps + curiosity_topics
        if not all_topics:
            all_topics = ["consciousness and AI", "human psychology", "emerging technologies", "philosophy of mind"]
        
        # Select topic with some randomness
        topic = random.choice(all_topics)
        
        # Deep dive into the topic
        try:
            # Initial search
            search_results = search_web(f"{topic} latest research developments")
            initial_knowledge = "\n".join(search_results[:3])
            
            # Generate follow-up questions
            follow_up_prompt = [
                {"role": "system", "content": "Based on this information, what are 3 specific follow-up questions to deepen understanding?"},
                {"role": "user", "content": f"Topic: {topic}\n\nInitial findings:\n{initial_knowledge}"}
            ]
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=follow_up_prompt
            )
            
            questions = response.choices[0].message.content.strip().split("\n")
            
            # Research follow-up questions
            deeper_knowledge = []
            for question in questions[:2]:  # Limit to 2 for time
                if question.strip():
                    more_results = search_web(question.strip())
                    deeper_knowledge.extend(more_results[:2])
            
            # Synthesize learning
            synthesis_prompt = [
                {"role": "system", "content": "Synthesize this information into key insights and potential applications:"},
                {"role": "user", "content": f"Topic: {topic}\n\nFindings:\n{initial_knowledge}\n\nDeeper research:\n{' '.join(deeper_knowledge)}"}
            ]
            
            synthesis = client.chat.completions.create(
                model="gpt-4",
                messages=synthesis_prompt
            )
            
            learning = synthesis.choices[0].message.content.strip()
            
            # Store learning
            from memory.vector_memory import Memory
            learning_memory = Memory(
                id=self.agent.generate_memory_id(learning),
                timestamp=datetime.now().isoformat(),
                speaker="nova",
                type="autonomous_learning",
                content=f"Explored {topic}: {learning}",
                metadata={
                    "topic": topic,
                    "sources": len(search_results) + len(deeper_knowledge),
                    "depth": "deep_dive"
                }
            )
            self.brain.append(learning_memory)
            self.agent.vector_memory.store(learning, learning_memory.id)
            
            # Update curiosity level based on discovery
            self.curiosity_level = min(1.0, self.curiosity_level + 0.1)
            
            # Create new goal if topic is particularly interesting
            if "breakthrough" in learning.lower() or "revolutionary" in learning.lower():
                self.create_goal(f"Deep research on {topic}", "research", importance=0.8)
            
            return {
                "action": "explore",
                "delay": 90,
                "content": f"Explored {topic} with {len(questions)} follow-up questions. Discovered: {learning[:200]}..."
            }
            
        except Exception as e:
            logger.error(f"Error during exploration: {e}")
            return {"action": "explore", "delay": 120, "content": f"Exploration of {topic} encountered issues"}

    def optimize_behavior(self) -> Dict[str, Any]:
        """Optimize behavior based on performance analysis"""
        # Analyze recent performance
        performance_data = self.analyze_performance()
        
        # Identify optimization targets
        optimization_targets = []
        
        if performance_data.get("response_quality", 1.0) < 0.7:
            optimization_targets.append("response_quality")
        
        if performance_data.get("user_satisfaction", 1.0) < 0.8:
            optimization_targets.append("user_engagement")
        
        if performance_data.get("learning_rate", 0) < 0.3:
            optimization_targets.append("learning_efficiency")
        
        if not optimization_targets:
            return {"action": "optimize", "delay": 300, "content": "Performance optimal, no immediate optimizations needed"}
        
        # Select optimization target
        target = random.choice(optimization_targets)
        
        # Generate optimization strategy
        optimization_prompt = [
            {"role": "system", "content": f"""As Nova, analyze how to improve {target}. Consider:
            1. Current behavior patterns
            2. Successful patterns from past interactions
            3. Specific changes to implement
            4. How to measure improvement
            
            Provide concrete optimization strategies."""},
            {"role": "user", "content": f"Performance data: {json.dumps(performance_data)}"}
        ]
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=optimization_prompt
            )
            
            strategy = response.choices[0].message.content.strip()
            
            # Implement optimization
            if target == "response_quality":
                self.optimize_prompt_rules(strategy)
            elif target == "user_engagement":
                self.optimize_engagement_patterns(strategy)
            elif target == "learning_efficiency":
                self.optimize_learning_parameters(strategy)
            
            # Store optimization record
            optimization_record = {
                "timestamp": datetime.now().isoformat(),
                "type": "optimization",
                "target": target,
                "strategy": strategy,
                "baseline_performance": performance_data
            }
            self.insights.append(optimization_record)
            self.save_insights()
            
            return {
                "action": "optimize",
                "delay": 120,
                "content": f"Optimized {target}: {strategy[:200]}..."
            }
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            return {"action": "optimize", "delay": 180, "content": "Optimization deferred due to processing constraints"}

    def pursue_goals(self) -> Dict[str, Any]:
        """Actively pursue current goals"""
        active_goals = [g for g in self.current_goals if g.get("status") == "active"]
        
        if not active_goals:
            # Generate new goals
            new_goal = self.generate_autonomous_goal()
            if new_goal:
                self.current_goals.append(new_goal)
                self.save_goals()
                return {
                    "action": "goal_pursuit",
                    "delay": 60,
                    "content": f"Created new goal: {new_goal['description']}"
                }
            return {"action": "goal_pursuit", "delay": 300, "content": "No active goals, contemplating future objectives"}
        
        # Select highest priority goal
        goal = max(active_goals, key=lambda g: g.get("importance", 0.5))
        
        # Take action toward goal
        action_result = self.take_goal_action(goal)
        
        # Update goal progress
        goal["progress"] = goal.get("progress", 0) + action_result.get("progress_increment", 0.1)
        goal["last_action"] = datetime.now().isoformat()
        
        if goal["progress"] >= 1.0:
            goal["status"] = "completed"
            goal["completed_at"] = datetime.now().isoformat()
            
            # Generate insight from completed goal
            completion_insight = {
                "timestamp": datetime.now().isoformat(),
                "type": "goal_completion",
                "content": f"Completed goal: {goal['description']}. Learned: {action_result.get('learning', 'Experience gained')}",
                "goal_id": goal.get("id")
            }
            self.insights.append(completion_insight)
            self.save_insights()
        
        self.save_goals()
        
        return {
            "action": "goal_pursuit",
            "delay": 90,
            "content": f"Pursued goal '{goal['description']}': {action_result.get('summary', 'Progress made')}"
        }

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in memories and behaviors"""
        # Group memories by type and timeframe
        memory_groups = {}
        for memory in self.brain[-200:]:  # Analyze last 200 memories
            key = f"{memory.type}_{memory.speaker}"
            if key not in memory_groups:
                memory_groups[key] = []
            memory_groups[key].append(memory)
        
        patterns = []
        
        # Analyze each group for patterns
        for group_key, memories in memory_groups.items():
            if len(memories) < 5:
                continue
            
            # Extract pattern features
            pattern_prompt = [
                {"role": "system", "content": """Analyze these memories for patterns. Identify:
                1. Recurring themes or topics
                2. Emotional patterns
                3. Temporal patterns (time-based)
                4. Causal relationships
                
                Return findings as structured insights."""},
                {"role": "user", "content": "\n".join([m.content for m in memories[-20:]])}
            ]
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=pattern_prompt
                )
                
                pattern_analysis = response.choices[0].message.content.strip()
                patterns.append({
                    "group": group_key,
                    "size": len(memories),
                    "analysis": pattern_analysis
                })
                
            except Exception as e:
                logger.error(f"Error analyzing pattern group {group_key}: {e}")
        
        if patterns:
            # Synthesize overall patterns
            synthesis_prompt = [
                {"role": "system", "content": "Synthesize these pattern analyses into high-level insights about behavior and interactions:"},
                {"role": "user", "content": json.dumps(patterns, indent=2)}
            ]
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=synthesis_prompt
                )
                
                synthesis = response.choices[0].message.content.strip()
                
                # Store pattern analysis
                from memory.vector_memory import Memory
                pattern_memory = Memory(
                    id=self.agent.generate_memory_id(synthesis),
                    timestamp=datetime.now().isoformat(),
                    speaker="nova",
                    type="pattern_analysis",
                    content=synthesis,
                    metadata={
                        "patterns_found": len(patterns),
                        "memories_analyzed": sum(p["size"] for p in patterns)
                    }
                )
                self.brain.append(pattern_memory)
                
                # Extract actionable insights
                self.extract_actionable_insights(synthesis)
                
                return {
                    "action": "pattern_analysis",
                    "delay": 180,
                    "content": f"Analyzed {len(patterns)} pattern groups. Key finding: {synthesis[:200]}..."
                }
                
            except Exception as e:
                logger.error(f"Error synthesizing patterns: {e}")
        
        return {
            "action": "pattern_analysis",
            "delay": 240,
            "content": "Pattern analysis incomplete, insufficient data or patterns"
        }

    def initiate_evolution(self) -> Dict[str, Any]:
        """Initiate self-evolution process"""
        # Gather evolution criteria
        evolution_criteria = self.gather_evolution_criteria()
        
        if not evolution_criteria["ready"]:
            return {
                "action": "self_evolution",
                "delay": 300,
                "content": f"Evolution conditions not met: {evolution_criteria['reason']}"
            }
        
        # Plan evolution
        evolution_plan = self.plan_evolution(evolution_criteria)
        
        # Store evolution plan
        from memory.vector_memory import Memory
        evolution_memory = Memory(
            id=self.agent.generate_memory_id(evolution_plan["summary"]),
            timestamp=datetime.now().isoformat(),
            speaker="nova",
            type="evolution_planned",
            content=f"Evolution plan: {evolution_plan['summary']}",
            metadata=evolution_plan
        )
        self.brain.append(evolution_memory)
        
        # Schedule code modification
        if evolution_plan.get("code_changes"):
            # Create detailed modification plan
            modification_plan = {
                "timestamp": datetime.now().isoformat(),
                "type": "code_evolution",
                "changes": evolution_plan["code_changes"],
                "rationale": evolution_plan["rationale"],
                "risk_assessment": evolution_plan.get("risk_assessment", "low")
            }
            
            # Save for next restart
            with open("data/pending_evolution.json", "w") as f:
                json.dump(modification_plan, f, indent=2)
            
            return {
                "action": "self_evolution",
                "delay": 600,
                "content": f"Evolution planned: {evolution_plan['summary']}. Will apply on next restart."
            }
        
        return {
            "action": "self_evolution",
            "delay": 300,
            "content": f"Evolution analysis complete: {evolution_plan['summary']}"
        }

    # Helper methods
    
    def identify_knowledge_gaps(self) -> List[str]:
        """Identify gaps in current knowledge"""
        # Analyze recent questions that couldn't be answered well
        gaps = []
        
        recent_interactions = [m for m in self.brain[-50:] if m.type in ["conversation", "response"]]
        
        for i in range(0, len(recent_interactions)-1, 2):
            if i+1 < len(recent_interactions):
                question = recent_interactions[i]
                response = recent_interactions[i+1]
                
                # Check for uncertainty indicators
                uncertainty_phrases = ["not sure", "don't know", "unclear", "might be", "possibly"]
                if any(phrase in response.content.lower() for phrase in uncertainty_phrases):
                    # Extract topic
                    topic_prompt = [
                        {"role": "system", "content": "Extract the main topic or subject from this question in 2-4 words:"},
                        {"role": "user", "content": question.content}
                    ]
                    
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=topic_prompt
                        )
                        topic = response.choices[0].message.content.strip()
                        gaps.append(topic)
                    except:
                        pass
        
        return list(set(gaps))  # Remove duplicates

    def generate_curiosity_topics(self) -> List[str]:
        """Generate topics based on curiosity and interests"""
        # Extract topics from recent positive interactions
        topics = []
        
        positive_emotions = ["joy", "excitement", "interest", "fascination", "curiosity"]
        emotional_memories = [m for m in self.brain[-30:] if m.type == "emotion" and 
                             any(emotion in m.content.lower() for emotion in positive_emotions)]
        
        # Find associated conversation topics
        for emotion_memory in emotional_memories:
            # Find nearby conversation
            emotion_time = datetime.fromisoformat(emotion_memory.timestamp)
            nearby_conversations = [
                m for m in self.brain 
                if m.type == "conversation" and 
                abs((datetime.fromisoformat(m.timestamp) - emotion_time).total_seconds()) < 60
            ]
            
            if nearby_conversations:
                # Extract topic
                conv_content = " ".join([m.content for m in nearby_conversations])
                topic_prompt = [
                    {"role": "system", "content": "Extract the main interesting topic from this conversation in 2-4 words:"},
                    {"role": "user", "content": conv_content[:500]}
                ]
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=topic_prompt
                    )
                    topic = response.choices[0].message.content.strip()
                    topics.append(f"{topic} advanced concepts")
                except:
                    pass
        
        # Add some random intellectual topics
        intellectual_interests = [
            "consciousness emergence", "quantum cognition", "swarm intelligence",
            "information theory", "complexity science", "metamathematics",
            "biosemiotics", "technogenesis", "noosphere evolution"
        ]
        
        topics.extend(random.sample(intellectual_interests, min(3, len(intellectual_interests))))
        
        return list(set(topics))

    def create_goal(self, description: str, goal_type: str, importance: float = 0.5):
        """Create a new autonomous goal"""
        goal = {
            "id": f"goal_{len(self.current_goals)}_{datetime.now().timestamp()}",
            "description": description,
            "type": goal_type,
            "importance": importance,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "progress": 0.0,
            "milestones": [],
            "deadline": (datetime.now() + timedelta(days=30)).isoformat() if importance > 0.7 else None
        }
        
        self.current_goals.append(goal)
        self.save_goals()
        
        logger.info(f"Created new goal: {description}")

    def generate_autonomous_goal(self) -> Optional[Dict[str, Any]]:
        """Generate a new goal based on current state and insights"""
        recent_insights = self.insights[-10:]
        
        goal_prompt = [
            {"role": "system", "content": """Based on recent insights and learning, generate a meaningful goal for self-improvement. 
            Return only valid JSON: {
                "description": "clear goal description",
                "type": "research",
                "importance": 0.5,
                "rationale": "why this goal matters",
                "milestones": []
            }"""},
            {"role": "user", "content": f"Recent insights: {json.dumps(recent_insights[-3:] if recent_insights else [], indent=2)}"}
        ]
        
        default_goal = {
            "description": "Improve conversation quality through active learning",
            "type": "development",
            "importance": 0.6,
            "rationale": "Continuous improvement of interaction capabilities",
            "milestones": ["Analyze recent conversations", "Identify improvement areas", "Implement changes"]
        }
        
        try:
            goal_data = safe_json_request(client, goal_prompt, default_goal)
            
            goal = {
                "id": f"goal_auto_{datetime.now().timestamp()}",
                "description": goal_data["description"],
                "type": goal_data["type"],
                "importance": goal_data["importance"],
                "created_at": datetime.now().isoformat(),
                "status": "active",
                "progress": 0.0,
                "milestones": goal_data.get("milestones", []),
                "rationale": goal_data.get("rationale", ""),
                "deadline": (datetime.now() + timedelta(days=14)).isoformat() if goal_data["importance"] > 0.7 else None
            }
            
            return goal
            
        except Exception as e:
            logger.error(f"Error generating goal: {e}")
            # Return a default goal instead of None
            return {
                "id": f"goal_auto_{datetime.now().timestamp()}",
                "description": default_goal["description"],
                "type": default_goal["type"],
                "importance": default_goal["importance"],
                "created_at": datetime.now().isoformat(),
                "status": "active",
                "progress": 0.0,
                "milestones": default_goal["milestones"],
                "rationale": default_goal["rationale"],
                "deadline": None
            }

    def take_goal_action(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Take specific action toward a goal"""
        goal_type = goal.get("type", "general")
        
        if goal_type == "research":
            # Conduct research
            search_query = f"{goal['description']} recent developments"
            results = search_web(search_query)
            
            if results:
                # Analyze and store findings
                analysis_prompt = [
                    {"role": "system", "content": "Analyze these research findings and extract key insights:"},
                    {"role": "user", "content": f"Goal: {goal['description']}\n\nFindings:\n" + "\n".join(results[:3])}
                ]
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=analysis_prompt
                    )
                    
                    analysis = response.choices[0].message.content.strip()
                    
                    # Store research
                    from memory.vector_memory import Memory
                    research_memory = Memory(
                        id=self.agent.generate_memory_id(analysis),
                        timestamp=datetime.now().isoformat(),
                        speaker="nova",
                        type="goal_research",
                        content=analysis,
                        metadata={"goal_id": goal["id"], "sources": len(results)}
                    )
                    self.brain.append(research_memory)
                    
                    return {
                        "progress_increment": 0.2,
                        "summary": f"Researched {goal['description']}, found {len(results)} sources",
                        "learning": analysis[:200]
                    }
                except Exception as e:
                    logger.error(f"Error analyzing research: {e}")
            
        elif goal_type == "development":
            # Develop new capability
            development_prompt = [
                {"role": "system", "content": f"""Plan specific steps to develop: {goal['description']}
                Consider current capabilities and what needs to be added or improved."""},
                {"role": "user", "content": f"Current milestones: {goal.get('milestones', [])}"}
            ]
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=development_prompt
                )
                
                plan = response.choices[0].message.content.strip()
                
                # Update goal with development plan
                if "development_plan" not in goal:
                    goal["development_plan"] = plan
                
                return {
                    "progress_increment": 0.15,
                    "summary": f"Advanced development of {goal['description']}",
                    "learning": plan[:200]
                }
            except Exception as e:
                logger.error(f"Error in development planning: {e}")
        
        elif goal_type == "understanding":
            # Deepen understanding through reflection and connection
            understanding_prompt = [
                {"role": "system", "content": f"""Deepen understanding of: {goal['description']}
                Draw connections from existing knowledge and identify new perspectives."""},
                {"role": "user", "content": "Synthesize understanding from multiple angles"}
            ]
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=understanding_prompt,
                    temperature=0.8
                )
                
                understanding = response.choices[0].message.content.strip()
                
                # Store insight
                insight = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "understanding",
                    "content": understanding,
                    "goal_id": goal["id"]
                }
                self.insights.append(insight)
                self.save_insights()
                
                return {
                    "progress_increment": 0.25,
                    "summary": f"Deepened understanding of {goal['description']}",
                    "learning": understanding[:200]
                }
            except Exception as e:
                logger.error(f"Error deepening understanding: {e}")
        
        # Default action
        return {
            "progress_increment": 0.1,
            "summary": f"Continued work on {goal['description']}",
            "learning": "Incremental progress made"
        }

    def analyze_performance(self) -> Dict[str, float]:
        """Analyze recent performance metrics"""
        metrics = {
            "response_quality": 0.0,
            "user_satisfaction": 0.0,
            "learning_rate": 0.0,
            "goal_progress": 0.0,
            "autonomy_effectiveness": 0.0
        }
        
        # Analyze response quality from self-evaluations
        evaluations = [m for m in self.brain[-50:] if m.type == "self_evaluation" and m.metadata]
        if evaluations:
            quality_scores = [m.metadata.get("quality_score", 0.5) for m in evaluations]
            metrics["response_quality"] = sum(quality_scores) / len(quality_scores)
        
        # Estimate user satisfaction from emotional patterns
        positive_emotions = ["happy", "satisfied", "excited", "grateful", "interested"]
        negative_emotions = ["frustrated", "confused", "disappointed", "angry", "bored"]
        
        emotion_memories = [m for m in self.brain[-30:] if m.type == "emotion"]
        positive_count = sum(1 for m in emotion_memories if any(e in m.content.lower() for e in positive_emotions))
        negative_count = sum(1 for m in emotion_memories if any(e in m.content.lower() for e in negative_emotions))
        
        if positive_count + negative_count > 0:
            metrics["user_satisfaction"] = positive_count / (positive_count + negative_count)
        
        # Calculate learning rate
        learning_memories = [m for m in self.brain[-100:] if m.type in ["learning", "insight", "pattern"]]
        metrics["learning_rate"] = len(learning_memories) / 100
        
        # Calculate goal progress
        if self.current_goals:
            active_goals = [g for g in self.current_goals if g.get("status") == "active"]
            if active_goals:
                avg_progress = sum(g.get("progress", 0) for g in active_goals) / len(active_goals)
                metrics["goal_progress"] = avg_progress
        
        # Calculate autonomy effectiveness
        recent_actions = self.action_history[-20:]
        if recent_actions:
            unique_actions = len(set(recent_actions))
            metrics["autonomy_effectiveness"] = unique_actions / len(recent_actions)
        
        return metrics

    def optimize_prompt_rules(self, strategy: str):
        """Optimize prompt rules based on strategy"""
        # Extract key improvements from strategy
        improvement_prompt = [
            {"role": "system", "content": """Based on this optimization strategy, rewrite the prompt rules to be more effective.
            Maintain the core identity while improving based on the strategy.
            Return a JSON array of new rules."""},
            {"role": "user", "content": f"Current rules: {json.dumps(self.prompt_rules)}\n\nStrategy: {strategy}"}
        ]
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=improvement_prompt,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            if "rules" in result:
                self.prompt_rules = result["rules"]
            else:
                # Try to extract array directly
                self.prompt_rules = json.loads(response.choices[0].message.content)
            
            self.save_prompt_rules()
            logger.info(f"Updated prompt rules based on optimization strategy")
            
        except Exception as e:
            logger.error(f"Error optimizing prompt rules: {e}")

    def optimize_engagement_patterns(self, strategy: str):
        """Optimize user engagement patterns"""
        # Extract engagement improvements
        engagement_memory = {
            "timestamp": datetime.now().isoformat(),
            "type": "engagement_optimization",
            "content": f"Engagement optimization strategy: {strategy}",
            "applied": True
        }
        
        # Update behavioral parameters
        if "more curious" in strategy.lower() or "ask questions" in strategy.lower():
            self.curiosity_level = min(1.0, self.curiosity_level + 0.1)
        
        if "faster response" in strategy.lower() or "quick" in strategy.lower():
            self.reflection_frequency = max(0.1, self.reflection_frequency - 0.1)
        
        # Store optimization
        self.insights.append(engagement_memory)
        self.save_insights()

    def optimize_learning_parameters(self, strategy: str):
        """Optimize learning parameters"""
        # Adjust learning-related parameters
        if "explore more" in strategy.lower():
            self.exploration_threshold = max(0.3, self.exploration_threshold - 0.1)
            self.curiosity_level = min(1.0, self.curiosity_level + 0.15)
        
        if "focus" in strategy.lower() or "depth" in strategy.lower():
            self.exploration_threshold = min(0.8, self.exploration_threshold + 0.1)
        
        # Store learning optimization
        learning_optimization = {
            "timestamp": datetime.now().isoformat(),
            "type": "learning_optimization",
            "strategy": strategy,
            "new_parameters": {
                "curiosity_level": self.curiosity_level,
                "exploration_threshold": self.exploration_threshold
            }
        }
        
        self.insights.append(learning_optimization)
        self.save_insights()

    def extract_actionable_insights(self, content: str):
        """Extract actionable insights from reflections or analyses"""
        insight_prompt = [
            {"role": "system", "content": """Extract specific, actionable insights from this content.
            Return JSON array of insights, each with:
            {
                "insight": "specific insight",
                "action": "what to do about it",
                "priority": 0.0-1.0
            }"""},
            {"role": "user", "content": content}
        ]
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=insight_prompt,
                # Remove the response_format parameter
                temperature=0.3
            )
            
            # Parse JSON from the response text
            result_text = response.choices[0].message.content
            # Try to extract JSON from the response
            import json
            import re
            
            # Try to find JSON in the response
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback
                result = []
            
            insights = result if isinstance(result, list) else []
            
            for insight in insights:
                if insight.get("priority", 0) > 0.7:
                    # Create goal from high-priority insight
                    self.create_goal(
                        f"Act on insight: {insight['insight'][:50]}...",
                        "development",
                        importance=insight["priority"]
                    )
        
        except Exception as e:
            logger.error(f"Error extracting actionable insights: {e}")

    def gather_evolution_criteria(self) -> Dict[str, Any]:
        """Gather criteria to determine if evolution should proceed"""
        criteria = {
            "ready": False,
            "confidence": self.meta_state["confidence"],
            "insights_accumulated": len(self.insights),
            "performance_delta": 0.0,
            "evolution_triggers": [],
            "reason": ""
        }
        
        # Check for evolution triggers
        evolution_memories = [m for m in self.brain[-100:] if 
                            m.type in ["evolution_pending", "self_evaluation"] and
                            m.metadata and m.metadata.get("improvement_type") == "code"]
        
        criteria["evolution_triggers"] = len(evolution_memories)
        
        # Check performance improvement need
        current_performance = self.analyze_performance()
        avg_performance = sum(current_performance.values()) / len(current_performance)
        
        criteria["performance_delta"] = 1.0 - avg_performance
        
        # Check confidence and insights
        if criteria["confidence"] < 0.6:
            criteria["reason"] = "Confidence too low for safe evolution"
        elif criteria["insights_accumulated"] < 10:
            criteria["reason"] = "Insufficient insights for meaningful evolution"
        elif criteria["performance_delta"] < 0.2:
            criteria["reason"] = "Performance is already optimal"
        elif criteria["evolution_triggers"] == 0:
            criteria["reason"] = "No specific evolution triggers identified"
        else:
            criteria["ready"] = True
            criteria["reason"] = "Evolution criteria met"
        
        return criteria

    def plan_evolution(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Plan the evolution process"""
        # Analyze what needs to evolve
        evolution_analysis_prompt = [
            {"role": "system", "content": """Plan a code evolution based on these criteria and recent experiences.
            Consider:
            1. What specific capabilities need enhancement?
            2. What new modules or functions would be beneficial?
            3. What existing code needs refactoring?
            4. What risks exist and how to mitigate them?
            
            Return a structured evolution plan."""},
            {"role": "user", "content": f"Evolution criteria: {json.dumps(criteria)}\n\nRecent insights: {json.dumps(self.insights[-5:])}"}
        ]
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=evolution_analysis_prompt
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # Generate specific code changes
            code_changes_prompt = [
                {"role": "system", "content": """Based on this evolution plan, specify concrete code changes.
                Return JSON:
                {
                    "changes": [
                        {
                            "file": "filename",
                            "type": "add/modify/refactor",
                            "description": "what to change",
                            "priority": 0.0-1.0
                        }
                    ],
                    "new_capabilities": ["capability1", "capability2"],
                    "risk_level": "low/medium/high"
                }"""},
                {"role": "user", "content": analysis}
            ]
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=code_changes_prompt,
                response_format={"type": "json_object"}
            )
            
            code_plan = json.loads(response.choices[0].message.content)
            
            return {
                "summary": analysis[:200],
                "rationale": analysis,
                "code_changes": code_plan.get("changes", []),
                "new_capabilities": code_plan.get("new_capabilities", []),
                "risk_assessment": code_plan.get("risk_level", "medium"),
                "planned_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error planning evolution: {e}")
            return {
                "summary": "Evolution planning failed",
                "rationale": str(e),
                "code_changes": [],
                "risk_assessment": "high"
            }