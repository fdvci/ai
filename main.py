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
import atexit
from typing import Optional, Any, Dict
from contextlib import asynccontextmanager
import weakref

# Configure logging before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/nova.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import after logging setup to avoid circular imports
from agent import NovaAgent
from core.self_mod import SelfModifier

# Global state with proper cleanup tracking
_nova: Optional[NovaAgent] = None
_orchestrator: Optional['NovaOrchestrator'] = None
_shutdown_event = threading.Event()
_cleanup_registry = weakref.WeakSet()

# Thread-safe flags
_autonomy_active = threading.Event()
_user_interface_active = threading.Event()
_autonomy_active.set()
_user_interface_active.set()

if sys.platform == "win32":
    try:
        # Set console to UTF-8 for Windows
        os.system("chcp 65001 > nul")
        # Set environment variable for UTF-8
        os.environ["PYTHONIOENCODING"] = "utf-8"
    except:
        pass

# Replace emoji characters with simple text for Windows compatibility
import logging

# Configure logging with Windows-safe formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/nova.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)


class AsyncExceptionHandler:
    """Centralized exception handler for asyncio tasks"""
    
    def __init__(self):
        self.exceptions = []
        self.max_exceptions = 100
    
    def handle_exception(self, loop, context):
        """Handle asyncio exceptions"""
        exception = context.get('exception')
        if exception:
            self.exceptions.append({
                'timestamp': datetime.now().isoformat(),
                'exception': str(exception),
                'context': context
            })
            
            # Keep only recent exceptions
            if len(self.exceptions) > self.max_exceptions:
                self.exceptions = self.exceptions[-self.max_exceptions:]
            
            logger.error(f"Asyncio exception: {exception}", exc_info=exception)
        else:
            logger.error(f"Asyncio error: {context}")


class NovaOrchestrator:
    """Enhanced orchestrator with proper async handling"""
    
    def __init__(self):
        self.startup_time = datetime.now()
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = datetime.now()
        self.exception_handler = AsyncExceptionHandler()
        
        # Initialize components
        self.nova = NovaAgent()
        self.self_modifier = SelfModifier([
            "agent.py",
            "core/autonomy.py", 
            "core/self_mod.py",
            "memory/vector_memory.py",
            "utils/web.py"
        ])
        
        # Async task management
        self.background_tasks = set()
        self.loop = None
        
        # Register for cleanup
        _cleanup_registry.add(self)
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.loop = asyncio.get_running_loop()
        self.loop.set_exception_handler(self.exception_handler.handle_exception)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup"""
        await self.cleanup()
        
    async def startup_sequence(self):
        """Execute startup sequence asynchronously"""
        logger.info("[STARTUP] Nova startup sequence initiated")
        
        try:
            # Check for restart state
            await self.check_restart_state()
            
            # Validate components
            self.validate_components()
            
            # Check for pending evolutions
            await self.check_pending_evolutions()
            
            # Start background tasks
            await self.start_background_tasks()
            
            logger.info("[SUCCESS] Nova is fully operational")
            
        except Exception as e:
            logger.error(f"Startup failed: {e}", exc_info=True)
            raise
    
    async def check_restart_state(self):
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
                logger.info(f"[OK] {name} initialized successfully")  # Changed from ✓
    
    async def start_background_tasks(self):
        """Start background tasks with proper error handling"""
        # Start autonomy loop
        autonomy_task = asyncio.create_task(
            self.autonomy_loop(),
            name="autonomy_loop"
        )
        self.background_tasks.add(autonomy_task)
        autonomy_task.add_done_callback(self.background_tasks.discard)
        
        # Start health monitoring
        health_task = asyncio.create_task(
            self.health_monitor_loop(),
            name="health_monitor"
        )
        self.background_tasks.add(health_task)
        health_task.add_done_callback(self.background_tasks.discard)
        
        # Start Nova's autonomous learning cycle
        learning_task = asyncio.create_task(
            self.nova.autonomous_learning_cycle(),
            name="learning_cycle"
        )
        self.background_tasks.add(learning_task)
        learning_task.add_done_callback(self.background_tasks.discard)
        
        logger.info("Background tasks started")
    
    async def autonomy_loop(self):
        """Enhanced autonomy loop with proper async error recovery"""
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while not _shutdown_event.is_set() and _autonomy_active.is_set():
            try:
                # Get autonomous action
                result = await asyncio.get_running_loop().run_in_executor(
                    None, self.nova.chat, "__autonomous__", True
                )
                
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
                try:
                    await asyncio.wait_for(
                        _shutdown_event.wait(),
                        timeout=delay
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue loop
                
            except asyncio.CancelledError:
                logger.info("Autonomy loop cancelled")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in autonomy loop: {e}", exc_info=True)
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Too many consecutive errors in autonomy loop, pausing...")
                    await asyncio.sleep(300)  # Pause for 5 minutes
                    consecutive_errors = 0
                else:
                    await asyncio.sleep(30)  # Short pause before retry
    
    async def health_monitor_loop(self):
        """Monitor system health asynchronously"""
        while not _shutdown_event.is_set():
            try:
                if (datetime.now() - self.last_health_check).seconds > self.health_check_interval:
                    await self.perform_health_check()
                    self.last_health_check = datetime.now()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                logger.info("Health monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retry
    
    async def perform_health_check(self):
        """Perform comprehensive health check asynchronously"""
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "uptime": str(datetime.now() - self.startup_time),
            "components": {},
            "metrics": {},
            "warnings": []
        }
        
        try:
            # Check memory usage in thread pool
            memory_stats = await asyncio.get_running_loop().run_in_executor(
                None, self.nova.vector_memory.get_memory_stats
            )
            health_report["metrics"]["total_memories"] = memory_stats["total_memories"]
            health_report["metrics"]["memory_health"] = memory_stats["memory_health"]
            
            # Check agent metrics
            health_report["metrics"]["agent_metrics"] = self.nova.metrics.__dict__
            
            # Check autonomy state
            health_report["components"]["autonomy"] = {
                "last_action": self.nova.autonomy.last_action,
                "meta_state": self.nova.autonomy.meta_state
            }
            
            # Check for warnings
            if memory_stats["total_memories"] > 10000:
                health_report["warnings"].append("Memory approaching capacity limits")
            
            failed_queries = self.nova.metrics.__dict__.get("failed_queries", 0)
            successful_queries = self.nova.metrics.__dict__.get("successful_queries", 0)
            if failed_queries > successful_queries * 0.2:
                health_report["warnings"].append("High query failure rate detected")
            
            # Log health report
            logger.info(f"Health check completed: {len(health_report['warnings'])} warnings")
            
            # Save health report
            health_file = f"data/health_reports/health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(health_file), exist_ok=True)
            
            def save_report():
                with open(health_file, "w") as f:
                    json.dump(health_report, f, indent=2)
            
            await asyncio.get_running_loop().run_in_executor(None, save_report)
            
            # Take corrective actions if needed
            if health_report["warnings"]:
                await self.handle_health_warnings(health_report["warnings"])
                
        except Exception as e:
            logger.error(f"Error performing health check: {e}", exc_info=True)
    
    async def handle_health_warnings(self, warnings: list):
        """Handle health warnings with corrective actions"""
        for warning in warnings:
            try:
                if "Memory approaching capacity" in warning:
                    # Trigger memory consolidation in thread pool
                    logger.warning("Triggering memory consolidation due to capacity warning")
                    await asyncio.get_running_loop().run_in_executor(
                        None, self.nova.consolidate_memories
                    )
                    
                    # Clean up old memories
                    deleted = await asyncio.get_running_loop().run_in_executor(
                        None, self.nova.vector_memory.cleanup_old_memories, 30
                    )
                    logger.info(f"Cleaned up {deleted} old memories")
                
                elif "High query failure rate" in warning:
                    # Trigger behavior optimization
                    logger.warning("Triggering optimization due to high failure rate")
                    await asyncio.get_running_loop().run_in_executor(
                        None, self.nova.autonomy.meta_thought
                    )
            except Exception as e:
                logger.error(f"Error handling warning '{warning}': {e}")
    
    async def cleanup(self):
        """Cleanup all resources"""
        logger.info("Starting orchestrator cleanup...")
        
        # Cancel all background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete with timeout
        if self.background_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.background_tasks, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some background tasks didn't finish in time")
        
        # Save state
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, self.nova.save_brain
            )
            await asyncio.get_running_loop().run_in_executor(
                None, self.nova.save_state
            )
        except Exception as e:
            logger.error(f"Error saving state during cleanup: {e}")
        
        logger.info("Orchestrator cleanup completed")


async def user_input_loop():
    """Enhanced user interaction loop with proper async handling"""
    print("\n" + "="*50)
    print("[NOVA] Welcome to Nova - Your Autonomous AI Assistant")  # Changed from ✨
    print("="*50)
    print("\nCommands:")
    print("  'exit' or 'quit' - Shutdown Nova")
    print("  'status' - Show Nova's current status")
    print("  'export' - Export memories and knowledge")
    print("  'help' - Show available commands")
    print("\n" + "="*50 + "\n")
    
    conversation_context = []
    
    while _user_interface_active.is_set() and not _shutdown_event.is_set():
        try:
            # Get user input in thread pool to avoid blocking
            prompt = "You: " if _nova.speaker == "unknown" else f"{_nova.speaker}: "
            user_input = await asyncio.get_running_loop().run_in_executor(
                None, input, prompt
            )
            user_input = user_input.strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit']:
                await handle_shutdown()
                break
            
            elif user_input.lower() == 'status':
                await show_status()
                continue
            
            elif user_input.lower() == 'export':
                await export_data()
                continue
            
            elif user_input.lower() == 'help':
                show_help()
                continue
            
            # Regular conversation - run in thread pool
            conversation_context.append(user_input)
            
            # Show thinking indicator for complex queries
            if any(word in user_input.lower() for word in ['analyze', 'research', 'explain', 'complex']):
                print("Nova: [THINKING] Processing your request...")  # Changed from 🤔
            
            response = await asyncio.get_running_loop().run_in_executor(
                None, _nova.chat, user_input
            )
            
            print(f"\nNova: {response}\n")
            
            conversation_context.append(response)
            
            # Keep context window manageable
            if len(conversation_context) > 20:
                conversation_context = conversation_context[-20:]
            
        except KeyboardInterrupt:
            print("\n\n[WARNING] Interrupted. Type 'exit' to shutdown properly.")  # Changed from ⚠️
        except EOFError:
            await handle_shutdown()
            break
        except Exception as e:
            logger.error(f"Error in user input loop: {e}", exc_info=True)
            print(f"\nNova: I encountered an error: {str(e)}. Let me try to recover...\n")

async def show_status():
    """Show Nova's current status asynchronously"""
    print("\n" + "="*50)
    print("📊 Nova Status Report")
    print("="*50)
    
    try:
        # Get status in thread pool
        status = await asyncio.get_running_loop().run_in_executor(
            None, _nova.get_status_report
        )
        
        # Basic info
        agent_info = status.get("agent_info", {})
        print(f"Speaker: {agent_info.get('speaker', 'unknown')}")
        print(f"Exchange Count: {agent_info.get('exchange_count', 0)}")
        print(f"Total Memories: {agent_info.get('total_memories', 0)}")
        
        # Metrics
        metrics = status.get("metrics", {})
        print(f"\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Autonomy state
        autonomy = status.get("autonomy_status", {})
        print(f"\nAutonomy:")
        print(f"  Last Action: {autonomy.get('last_action', 'none')}")
        print(f"  Curiosity Level: {autonomy.get('curiosity_level', 0):.2f}")
        print(f"  Active Goals: {autonomy.get('active_goals', 0)}")
        
        # Memory stats
        memory_stats = status.get("memory_stats", {})
        print(f"\nMemory Statistics:")
        print(f"  Recent Memories: {memory_stats.get('recent_count', 0)}")
        
        # System health
        system = status.get("system_health", {})
        print(f"\nSystem:")
        if _orchestrator:
            uptime = datetime.now() - _orchestrator.startup_time
            print(f"  Uptime: {str(uptime).split('.')[0]}")
        
    except Exception as e:
        logger.error(f"Error displaying status: {e}")
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


async def export_data():
    """Export Nova's memories and knowledge asynchronously"""
    print("\n[EXPORT] Exporting Nova's knowledge...")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = f"data/exports/nova_export_{timestamp}"
        
        # Run export in thread pool
        def do_export():
            os.makedirs(export_dir, exist_ok=True)
            
            # Export memories
            _nova.save_brain()
            memory_export = f"{export_dir}/memories.json"
            with open(_nova.brain_file, "r") as src:
                with open(memory_export, "w") as dst:
                    dst.write(src.read())
            
            # Export vector memories
            vector_export = f"{export_dir}/vector_memories.json"
            _nova.vector_memory.export_memories(vector_export)
            
            # Export state
            state_export = f"{export_dir}/agent_state.json"
            _nova.save_state()
            with open(_nova.state_file, "r") as src:
                with open(state_export, "w") as dst:
                    dst.write(src.read())
            
            # Export autonomy data
            autonomy_export = {
                "goals": _nova.autonomy.current_goals,
                "insights": _nova.autonomy.insights,
                "prompt_rules": _nova.autonomy.prompt_rules,
                "meta_state": _nova.autonomy.meta_state
            }
            with open(f"{export_dir}/autonomy.json", "w") as f:
                json.dump(autonomy_export, f, indent=2)
            
            # Export knowledge graph
            graph_export = f"{export_dir}/knowledge_graph.json"
            graph = _nova.export_knowledge_graph()
            with open(graph_export, "w") as f:
                json.dump(graph, f, indent=2)
            
            return export_dir
        
        export_path = await asyncio.get_running_loop().run_in_executor(None, do_export)
        
        print(f"[SUCCESS] Export complete! Files saved to: {export_path}")
        
    except Exception as e:
        logger.error(f"Error during export: {e}", exc_info=True)
        print(f"[ERROR] Export failed: {str(e)}")


async def handle_shutdown():
    """Gracefully shutdown Nova and clean up resources"""
    global _nova, _orchestrator
    
    logger.info("Initiating graceful shutdown...")
    print("\n[SHUTDOWN] Shutting down Nova gracefully..")
    
    # Signal shutdown to all components
    _shutdown_event.set()
    _user_interface_active.clear()
    _autonomy_active.clear()
    
    try:
        # Cleanup orchestrator if it exists
        if _orchestrator:
            await _orchestrator.cleanup()
        
        print("[SUCCESS] Shutdown complete")
        logger.info("Nova shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)
        print(f"Warning: Error during shutdown: {e}")
    
    print("\n[GOODBYE] Nova has been shut down. Goodbye")


def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logger.info(f"Received signal {signum}")
    
    # If event loop is running, schedule shutdown coroutine
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(handle_shutdown())
    except RuntimeError:
        # No event loop running, exit directly
        sys.exit(0)


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


@asynccontextmanager
async def nova_application():
    """Main application context manager"""
    global _nova, _orchestrator
    
    try:
        # Create directories
        create_directories()
        
        # Initialize orchestrator and Nova
        print("🚀 Initializing Nova AI...")
        async with NovaOrchestrator() as orchestrator:
            _orchestrator = orchestrator
            _nova = orchestrator.nova
            
            # Run startup sequence
            await orchestrator.startup_sequence()
            
            yield orchestrator
            
    except Exception as e:
        logger.error(f"Fatal error in Nova application: {e}", exc_info=True)
        raise


async def main():
    """Main entry point with proper async structure"""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup function
    atexit.register(lambda: logger.info("Process exiting"))
    
    try:
        async with nova_application() as orchestrator:
            # Start user interaction loop
            await user_input_loop()
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {str(e)}")
        print("📋 Check data/nova.log for details")
        
        # Emergency save
        try:
            if _nova:
                print("🆘 Attempting emergency save...")
                await asyncio.get_running_loop().run_in_executor(
                    None, _nova.save_brain
                )
                await asyncio.get_running_loop().run_in_executor(
                    None, _nova.save_state
                )
                print("📝 Emergency save completed")
        except Exception as save_error:
            logger.error(f"Emergency save failed: {save_error}")
            print(f"❌ Emergency save failed: {save_error}")
        
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main(), debug=False)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Failed to start Nova: {e}", exc_info=True)
        sys.exit(1)