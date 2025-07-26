import asyncio
import threading
import signal
import sys
import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional
from agent import NovaAgent
from core.self_mod import SelfModifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/nova.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global state
nova: Optional[NovaAgent] = None
orchestrator: Optional['NovaOrchestrator'] = None
autonomy_active = True
user_interface_active = True
shutdown_event = threading.Event()


class NovaOrchestrator:
    """Orchestrates all Nova components and processes"""
    
    def __init__(self):
        self.nova = NovaAgent()
        self.self_modifier = SelfModifier([
            "agent.py",
            "core/autonomy.py",
            "core/self_mod.py",
            "memory/vector_memory.py",
            "utils/web.py"
        ])
        self.startup_time = datetime.now()
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = datetime.now()
        
    def startup_sequence(self):
        """Execute startup sequence"""
        logger.info("🚀 Nova startup sequence initiated")
        
        # Check for restart state
        self.check_restart_state()
        
        # Load and validate components
        self.validate_components()
        
        # Check for pending evolutions
        self.check_pending_evolutions()
        
        # Initialize async event loop for background tasks
        self.setup_async_loop()
        
        # Start health monitoring
        self.start_health_monitoring()
        
        logger.info("✨ Nova is fully operational")
        
    def check_restart_state(self):
        """Check if this is a restart with pending state"""
        restart_state_file = "data/restart_state.json"
        if os.path.exists(restart_state_file):
            try:
                with open(restart_state_file, "r") as f:
                    state = json.load(f)
                
                logger.info(f"Resuming from restart: {state}")
                
                # Clean up restart state
                os.remove(restart_state_file)
                
            except Exception as e:
                logger.error(f"Error loading restart state: {e}")
    
    def validate_components(self):
        """Validate all components are properly initialized"""
        components = {
            "Agent": self.nova,
            "Vector Memory": self.nova.vector_memory,
            "Autonomy": self.nova.autonomy,
            "Self Modifier": self.self_modifier
        }
        
        for name, component in components.items():
            if component is None:
                logger.error(f"Component {name} failed to initialize")
                raise RuntimeError(f"Component {name} initialization failed")
            else:
                logger.info(f"✓ {name} initialized successfully")
    
    def check_pending_evolutions(self):
        """Check and apply any pending evolutions"""
        if os.path.exists("data/pending_evolution.json"):
            logger.info("Found pending evolution plan, applying...")
            try:
                self.self_modifier.reflect_and_patch()
                logger.info("Evolution applied successfully")
            except Exception as e:
                logger.error(f"Error applying evolution: {e}")
    
    def setup_async_loop(self):
        """Setup async event loop for background tasks"""
        self.loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(
            target=self._run_async_loop,
            daemon=True,
            name="AsyncEventLoop"
        )
        self.async_thread.start()
    
    def _run_async_loop(self):
        """Run the async event loop"""
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.nova.autonomous_learning_cycle())
        except Exception as e:
            logger.error(f"Error in async loop: {e}")
    
    def start_health_monitoring(self):
        """Start health monitoring thread"""
        self.health_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="HealthMonitor"
        )
        self.health_thread.start()
    
    def _health_monitor_loop(self):
        """Monitor system health"""
        while not shutdown_event.is_set():
            try:
                if (datetime.now() - self.last_health_check).seconds > self.health_check_interval:
                    self.perform_health_check()
                    self.last_health_check = datetime.now()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
    
    def perform_health_check(self):
        """Perform comprehensive health check"""
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "uptime": str(datetime.now() - self.startup_time),
            "components": {},
            "metrics": {},
            "warnings": []
        }
        
        try:
            # Check memory usage
            memory_stats = self.nova.vector_memory.get_memory_stats()
            health_report["metrics"]["total_memories"] = memory_stats["total_memories"]
            health_report["metrics"]["memory_health"] = memory_stats["memory_health"]
            
            # Check agent metrics
            health_report["metrics"]["agent_metrics"] = self.nova.metrics
            
            # Check autonomy state
            health_report["components"]["autonomy"] = {
                "last_action": self.nova.autonomy.last_action,
                "meta_state": self.nova.autonomy.meta_state
            }
            
            # Check for warnings
            if memory_stats["total_memories"] > 10000:
                health_report["warnings"].append("Memory approaching capacity limits")
            
            if self.nova.metrics.get("failed_queries", 0) > self.nova.metrics.get("successful_queries", 0) * 0.2:
                health_report["warnings"].append("High query failure rate detected")
            
            # Log health report
            logger.info(f"Health check: {json.dumps(health_report, indent=2)}")
            
            # Save health report
            health_file = f"data/health_reports/health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(health_file), exist_ok=True)
            with open(health_file, "w") as f:
                json.dump(health_report, f, indent=2)
            
            # Take corrective actions if needed
            if health_report["warnings"]:
                self.handle_health_warnings(health_report["warnings"])
                
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
    
    def handle_health_warnings(self, warnings: list):
        """Handle health warnings with corrective actions"""
        for warning in warnings:
            if "Memory approaching capacity" in warning:
                # Trigger memory consolidation
                logger.warning("Triggering memory consolidation due to capacity warning")
                self.nova.consolidate_memories()
                
                # Clean up old memories
                deleted = self.nova.vector_memory.cleanup_old_memories(days=30)
                logger.info(f"Cleaned up {deleted} old memories")
            
            elif "High query failure rate" in warning:
                # Trigger behavior optimization
                logger.warning("Triggering optimization due to high failure rate")
                self.nova.autonomy.meta_thought()  # Trigger reflection


def autonomy_loop():
    """Enhanced autonomy loop with error recovery"""
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    while autonomy_active and not shutdown_event.is_set():
        try:
            # Get autonomous action
            result = nova.chat("__autonomous__", structured=True)
            
            if result and isinstance(result, dict) and result.get("action") != "none":
                logger.info(f"🤖 Autonomous Action: {result['action']}")
                if result.get("content"):
                    logger.info(f"   Content: {result['content'][:100]}...")
            
            # Reset error counter on success
            consecutive_errors = 0
            
            # Dynamic delay based on action
            delay = result.get("delay", 60) if isinstance(result, dict) else 60
            delay = max(30, min(600, delay))  # Clamp between 30s and 10min
            
            # Wait with interruption capability
            shutdown_event.wait(delay)
            
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Error in autonomy loop: {e}")
            
            if consecutive_errors >= max_consecutive_errors:
                logger.error("Too many consecutive errors in autonomy loop, pausing...")
                time.sleep(300)  # Pause for 5 minutes
                consecutive_errors = 0
            else:
                time.sleep(30)  # Short pause before retry


def user_input_loop():
    """Enhanced user interaction loop"""
    print("\n" + "="*50)
    print("✨ Welcome to Nova - Your Autonomous AI Assistant")
    print("="*50)
    print("\nCommands:")
    print("  'exit' or 'quit' - Shutdown Nova")
    print("  'status' - Show Nova's current status")
    print("  'export' - Export memories and knowledge")
    print("  'help' - Show available commands")
    print("\n" + "="*50 + "\n")
    
    conversation_context = []
    
    while user_interface_active and not shutdown_event.is_set():
        try:
            # Show prompt with context awareness
            prompt = "You: " if nova.speaker == "unknown" else f"{nova.speaker}: "
            user_input = input(prompt).strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit']:
                handle_shutdown()
                break
            
            elif user_input.lower() == 'status':
                show_status()
                continue
            
            elif user_input.lower() == 'export':
                export_data()
                continue
            
            elif user_input.lower() == 'help':
                show_help()
                continue
            
            # Regular conversation
            conversation_context.append(user_input)
            
            # Show thinking indicator for complex queries
            if any(word in user_input.lower() for word in ['analyze', 'research', 'explain', 'complex']):
                print("Nova: 🤔 Thinking deeply about this...")
            
            response = nova.chat(user_input)
            
            print(f"\nNova: {response}\n")
            
            conversation_context.append(response)
            
            # Keep context window manageable
            if len(conversation_context) > 20:
                conversation_context = conversation_context[-20:]
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'exit' to shutdown properly.")
        except EOFError:
            handle_shutdown()
            break
        except Exception as e:
            logger.error(f"Error in user input loop: {e}")
            print(f"\nNova: I encountered an error: {str(e)}. Let me try to recover...\n")


def show_status():
    """Show Nova's current status"""
    print("\n" + "="*50)
    print("📊 Nova Status Report")
    print("="*50)
    
    try:
        # Basic info
        print(f"Speaker: {nova.speaker}")
        print(f"Exchange Count: {nova.exchange_count}")
        print(f"Total Memories: {len(nova.brain)}")
        
        # Metrics
        print(f"\nMetrics:")
        for key, value in nova.metrics.items():
            print(f"  {key}: {value}")
        
        # Autonomy state
        print(f"\nAutonomy:")
        print(f"  Last Action: {nova.autonomy.last_action}")
        print(f"  Curiosity Level: {nova.autonomy.curiosity_level:.2f}")
        print(f"  Active Goals: {len([g for g in nova.autonomy.current_goals if g.get('status') == 'active'])}")
        
        # Memory stats
        memory_stats = nova.vector_memory.get_memory_stats()
        print(f"\nMemory Statistics:")
        print(f"  Total Vector Memories: {memory_stats['total_memories']}")
        print(f"  Memory Health Score: {memory_stats['memory_health']['diversity_score']:.2f}")
        
        # Uptime
        if orchestrator:
            uptime = datetime.now() - orchestrator.startup_time
            print(f"\nSystem:")
            print(f"  Uptime: {str(uptime).split('.')[0]}")
        
    except Exception as e:
        print(f"Error displaying status: {e}")
    
    print("="*50 + "\n")


def show_help():
    """Show help information"""
    print("\n" + "="*50)
    print("📚 Nova Help")
    print("="*50)
    print("\nAvailable Commands:")
    print("  'exit'/'quit' - Shutdown Nova gracefully")
    print("  'status'      - Show current status and metrics")
    print("  'export'      - Export all memories and knowledge")
    print("  'help'        - Show this help message")
    print("\nConversation Tips:")
    print("  - Nova remembers your name and important facts")
    print("  - Ask for current information and Nova will search the web")
    print("  - Request analysis or research for in-depth responses")
    print("  - Nova learns and improves from interactions")
    print("  - Use 'analyze', 'research', or 'explain' for deep dives")
    print("\nAdvanced Features:")
    print("  - Autonomous learning runs in background")
    print("  - Self-modification and evolution capabilities")
    print("  - Vector-based memory with semantic search")
    print("  - Real-time web research integration")
    print("="*50 + "\n")


def export_data():
    """Export Nova's memories and knowledge"""
    print("\n📦 Exporting Nova's knowledge...")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = f"data/exports/nova_export_{timestamp}"
        os.makedirs(export_dir, exist_ok=True)
        
        # Export memories
        nova.save_brain()
        memory_export = f"{export_dir}/memories.json"
        with open(nova.brain_file, "r") as src:
            with open(memory_export, "w") as dst:
                dst.write(src.read())
        
        # Export vector memories
        vector_export = f"{export_dir}/vector_memories.json"
        nova.vector_memory.export_memories(vector_export)
        
        # Export state
        state_export = f"{export_dir}/agent_state.json"
        nova.save_state()
        with open(nova.state_file, "r") as src:
            with open(state_export, "w") as dst:
                dst.write(src.read())
        
        # Export autonomy data
        autonomy_export = {
            "goals": nova.autonomy.current_goals,
            "insights": nova.autonomy.insights,
            "prompt_rules": nova.autonomy.prompt_rules,
            "meta_state": nova.autonomy.meta_state
        }
        with open(f"{export_dir}/autonomy.json", "w") as f:
            json.dump(autonomy_export, f, indent=2)
        
        # Export knowledge graph
        graph_export = f"{export_dir}/knowledge_graph.json"
        graph = nova.export_knowledge_graph()
        with open(graph_export, "w") as f:
            json.dump(graph, f, indent=2)
        
        # Create summary
        summary = {
            "export_timestamp": datetime.now().isoformat(),
            "nova_version": "1.0",
            "total_memories": len(nova.brain),
            "vector_memories": nova.vector_memory.stats["total_memories"],
            "known_users": list(nova.state["known_users"].keys()),
            "active_goals": len([g for g in nova.autonomy.current_goals if g.get("status") == "active"]),
            "total_insights": len(nova.autonomy.insights),
            "uptime": str(datetime.now() - orchestrator.startup_time) if orchestrator else "Unknown"
        }
        
        with open(f"{export_dir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Export complete! Files saved to: {export_dir}")
        print(f"   - Total memories: {summary['total_memories']}")
        print(f"   - Vector memories: {summary['vector_memories']}")
        print(f"   - Active goals: {summary['active_goals']}")
        
    except Exception as e:
        logger.error(f"Error during export: {e}")
        print(f"❌ Export failed: {str(e)}")


def handle_shutdown():
    """Gracefully shutdown Nova and clean up resources"""
    global user_interface_active, autonomy_active
    
    logger.info("Initiating graceful shutdown...")
    print("\n🔄 Shutting down Nova gracefully...")
    
    # Stop all loops
    user_interface_active = False
    autonomy_active = False
    shutdown_event.set()
    
    try:
        # Save current state
        if nova:
            print("💾 Saving memories and state...")
            nova.save_brain()
            nova.save_state()
            logger.info("Nova state saved successfully")
        
        # Wait for threads to finish (with timeout)
        if orchestrator:
            print("⏳ Waiting for background processes...")
            # Give threads time to finish gracefully
            time.sleep(2)
        
        print("✅ Shutdown complete")
        logger.info("Nova shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        print(f"⚠️ Warning: Error during shutdown: {e}")
    
    print("\n👋 Nova has been shut down. Goodbye!")
    sys.exit(0)


def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logger.info(f"Received signal {signum}")
    handle_shutdown()


def create_directories():
    """Create necessary directories for Nova operation"""
    directories = [
        "data", "data/exports", "data/health_reports", 
        "data/shutdown_reports", "data/code_backups",
        "data/vector_db", "data/web_cache"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise


def main():
    """Main entry point with enhanced error handling"""
    global nova, orchestrator
    
    try:
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Create necessary directories
        create_directories()
        
        # Initialize orchestrator and Nova
        print("🚀 Initializing Nova AI...")
        orchestrator = NovaOrchestrator()
        nova = orchestrator.nova
        
        # Run startup sequence
        orchestrator.startup_sequence()
        
        # Start autonomy loop in background
        autonomy_thread = threading.Thread(
            target=autonomy_loop,
            daemon=True,
            name="AutonomyLoop"
        )
        autonomy_thread.start()
        logger.info("Autonomy loop started")
        
        # Wait a moment for everything to initialize
        time.sleep(1)
        
        # Start user interaction loop (blocking)
        user_input_loop()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        handle_shutdown()
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {str(e)}")
        print("📋 Check data/nova.log for details")
        
        # Emergency save
        try:
            if nova:
                print("🆘 Attempting emergency save...")
                nova.save_brain()
                nova.save_state()
                print("📝 Emergency save completed")
        except Exception as save_error:
            logger.error(f"Emergency save failed: {save_error}")
            print(f"❌ Emergency save failed: {save_error}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()