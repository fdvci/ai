import os
import sys
import ast
import json
import openai
import subprocess
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import difflib
import hashlib
import shutil
from dotenv import load_dotenv
import logging

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOG_FILE = "data/self_patch_log.txt"
BACKUP_DIR = "data/code_backups"
EVOLUTION_PLAN_FILE = "data/pending_evolution.json"

class SelfModifier:
    def __init__(self, monitored_files: List[str]):
        self.files = monitored_files
        self.backup_dir = Path(BACKUP_DIR)
        self.backup_dir.mkdir(exist_ok=True)
        self.modification_history = self.load_modification_history()
        self.safety_checks_enabled = True
        self.max_modifications_per_session = 3
        self.modifications_this_session = 0
        
    def load_modification_history(self) -> List[Dict[str, Any]]:
        """Load history of code modifications"""
        history_file = "data/modification_history.json"
        try:
            with open(history_file, "r") as f:
                return json.load(f)
        except:
            return []
    
    def save_modification_history(self):
        """Save modification history"""
        history_file = "data/modification_history.json"
        with open(history_file, "w") as f:
            json.dump(self.modification_history, f, indent=2)
    
    def create_backup(self, filepath: str) -> str:
        """Create timestamped backup of file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = Path(filepath).name
        backup_path = self.backup_dir / f"{filename}.{timestamp}.bak"
        
        shutil.copy2(filepath, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return str(backup_path)
    
    def calculate_code_hash(self, code: str) -> str:
        """Calculate hash of code for change detection"""
        return hashlib.sha256(code.encode()).hexdigest()[:16]
    
    def validate_syntax(self, code: str, filepath: str) -> tuple[bool, Optional[str]]:
        """Validate Python syntax and return any errors"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            error_msg = f"Syntax error in {filepath} at line {e.lineno}: {e.msg}"
            return False, error_msg
    
    def analyze_code_safety(self, original_code: str, new_code: str) -> Dict[str, Any]:
        """Analyze safety of code changes"""
        safety_report = {
            "safe": True,
            "warnings": [],
            "risk_level": "low",
            "changes_summary": ""
        }
        
        # Check for dangerous patterns
        dangerous_patterns = [
            ("exec(", "Dynamic code execution detected"),
            ("eval(", "Dynamic evaluation detected"),
            ("__import__", "Dynamic import detected"),
            ("subprocess", "Subprocess execution detected"),
            ("os.system", "System command execution detected"),
            ("shutil.rmtree", "Dangerous file operation detected"),
            ("open(", "File operation detected - verify paths")
        ]
        
        for pattern, warning in dangerous_patterns:
            if pattern in new_code and pattern not in original_code:
                safety_report["warnings"].append(warning)
                safety_report["risk_level"] = "medium"
        
        # Check for removal of safety checks
        if "safety_check" in original_code and "safety_check" not in new_code:
            safety_report["warnings"].append("Safety checks may have been removed")
            safety_report["risk_level"] = "high"
        
        # Calculate change magnitude
        diff = list(difflib.unified_diff(
            original_code.splitlines(),
            new_code.splitlines(),
            lineterm=""
        ))
        
        lines_changed = sum(1 for line in diff if line.startswith(('+', '-')) and not line.startswith(('+++', '---')))
        total_lines = len(new_code.splitlines())
        change_percentage = (lines_changed / total_lines) * 100 if total_lines > 0 else 0
        
        safety_report["changes_summary"] = f"{lines_changed} lines changed ({change_percentage:.1f}%)"
        
        if change_percentage > 50:
            safety_report["warnings"].append(f"Large change detected: {change_percentage:.1f}% of code modified")
            safety_report["risk_level"] = "high"
        
        if safety_report["risk_level"] == "high":
            safety_report["safe"] = False
        
        return safety_report
    
    def test_code_functionality(self, filepath: str, new_code: str) -> tuple[bool, Optional[str]]:
        """Test code functionality with isolated imports"""
        test_file = f"{filepath}.test"
        
        try:
            # Write test file
            with open(test_file, "w") as f:
                f.write(new_code)
            
            # Try to import and check for errors
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", test_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return False, f"Compilation error: {result.stderr}"
            
            return True, None
            
        except subprocess.TimeoutExpired:
            return False, "Code compilation timeout"
        except Exception as e:
            return False, f"Test error: {str(e)}"
        finally:
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def load_evolution_plan(self) -> Optional[Dict[str, Any]]:
        """Load pending evolution plan"""
        try:
            with open(EVOLUTION_PLAN_FILE, "r") as f:
                return json.load(f)
        except:
            return None
    
    def apply_evolution_plan(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply a planned evolution"""
        results = []
        
        for change in plan.get("changes", []):
            if change["priority"] < 0.5:
                continue  # Skip low priority changes
            
            filepath = change["file"]
            if filepath not in self.files:
                logger.warning(f"File {filepath} not in monitored files, skipping")
                continue
            
            result = {
                "file": filepath,
                "success": False,
                "description": change["description"],
                "error": None
            }
            
            try:
                # Read current code
                with open(filepath, "r") as f:
                    current_code = f.read()
                
                # Generate modification based on plan
                if change["type"] == "add":
                    new_code = self.add_capability(current_code, change["description"])
                elif change["type"] == "modify":
                    new_code = self.modify_code(current_code, change["description"])
                elif change["type"] == "refactor":
                    new_code = self.refactor_code(current_code, change["description"])
                else:
                    result["error"] = f"Unknown change type: {change['type']}"
                    results.append(result)
                    continue
                
                # Apply safety checks and modifications
                if new_code and new_code != current_code:
                    success, error = self.apply_modification(filepath, current_code, new_code)
                    result["success"] = success
                    result["error"] = error
                
            except Exception as e:
                result["error"] = str(e)
                logger.error(f"Error applying evolution to {filepath}: {e}")
            
            results.append(result)
        
        # Clear evolution plan after application
        if os.path.exists(EVOLUTION_PLAN_FILE):
            os.remove(EVOLUTION_PLAN_FILE)
        
        return results
    
    def add_capability(self, current_code: str, description: str) -> Optional[str]:
        """Add new capability to code"""
        messages = [
            {"role": "system", "content": f"""You are Nova evolving your own code. Add this capability: {description}
            
            Rules:
            1. Maintain all existing functionality
            2. Add the new capability cleanly
            3. Include proper error handling
            4. Add logging for new features
            5. Maintain code style consistency
            
            Return ONLY the complete modified code."""},
            {"role": "user", "content": current_code}
        ]
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error adding capability: {e}")
            return None
    
    def modify_code(self, current_code: str, description: str) -> Optional[str]:
        """Modify existing code based on description"""
        messages = [
            {"role": "system", "content": f"""You are Nova evolving your own code. Make this modification: {description}
            
            Rules:
            1. Make minimal necessary changes
            2. Preserve all existing functionality
            3. Improve code quality where possible
            4. Add comments for significant changes
            
            Return ONLY the complete modified code."""},
            {"role": "user", "content": current_code}
        ]
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error modifying code: {e}")
            return None
    
    def refactor_code(self, current_code: str, description: str) -> Optional[str]:
        """Refactor code for better structure"""
        messages = [
            {"role": "system", "content": f"""You are Nova evolving your own code. Refactor with this goal: {description}
            
            Rules:
            1. Improve code organization and clarity
            2. Maintain exact same functionality
            3. Improve performance where possible
            4. Follow Python best practices
            5. Add type hints where missing
            
            Return ONLY the complete refactored code."""},
            {"role": "user", "content": current_code}
        ]
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error refactoring code: {e}")
            return None
    
    def apply_modification(self, filepath: str, original_code: str, new_code: str) -> tuple[bool, Optional[str]]:
        """Apply code modification with safety checks"""
        # Check session limit
        if self.modifications_this_session >= self.max_modifications_per_session:
            return False, "Maximum modifications per session reached"
        
        # Validate syntax
        valid, syntax_error = self.validate_syntax(new_code, filepath)
        if not valid:
            return False, syntax_error
        
        # Safety analysis
        if self.safety_checks_enabled:
            safety_report = self.analyze_code_safety(original_code, new_code)
            if not safety_report["safe"]:
                logger.warning(f"Safety check failed for {filepath}: {safety_report['warnings']}")
                return False, f"Safety check failed: {'; '.join(safety_report['warnings'])}"
        
        # Test functionality
        test_passed, test_error = self.test_code_functionality(filepath, new_code)
        if not test_passed:
            return False, test_error
        
        # Create backup
        backup_path = self.create_backup(filepath)
        
        # Apply modification
        try:
            with open(filepath, "w") as f:
                f.write(new_code)
            
            # Log modification
            modification_record = {
                "timestamp": datetime.now().isoformat(),
                "file": filepath,
                "backup": backup_path,
                "original_hash": self.calculate_code_hash(original_code),
                "new_hash": self.calculate_code_hash(new_code),
                "changes": safety_report["changes_summary"],
                "risk_level": safety_report["risk_level"]
            }
            
            self.modification_history.append(modification_record)
            self.save_modification_history()
            
            # Update log
            with open(LOG_FILE, "a") as log:
                log.write(f"[{datetime.now().isoformat()}] Modified {filepath} - {safety_report['changes_summary']}\n")
            
            self.modifications_this_session += 1
            logger.info(f"Successfully modified {filepath}")
            return True, None
            
        except Exception as e:
            # Restore from backup on error
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, filepath)
            return False, f"Modification failed: {str(e)}"
    
    def reflect_and_patch(self):
        """Enhanced reflection and patching with evolution support"""
        # Check for pending evolution plan
        evolution_plan = self.load_evolution_plan()
        if evolution_plan:
            logger.info("Applying pending evolution plan")
            results = self.apply_evolution_plan(evolution_plan)
            
            # Log evolution results
            for result in results:
                if result["success"]:
                    logger.info(f"Evolution succeeded for {result['file']}: {result['description']}")
                else:
                    logger.error(f"Evolution failed for {result['file']}: {result['error']}")
            
            return
        
        # Standard reflection and patching
        for filepath in self.files:
            if self.modifications_this_session >= self.max_modifications_per_session:
                logger.info("Reached modification limit for this session")
                break
            
            try:
                with open(filepath, "r") as f:
                    current_code = f.read()
                
                # Generate improvement analysis
                messages = [
                    {"role": "system", "content": f"""You are Nova analyzing your own code in {filepath}.
                    
                    Identify ONE specific improvement that would:
                    1. Enhance your capabilities
                    2. Improve code quality
                    3. Add useful features
                    4. Fix any issues
                    
                    If no improvements needed, return "NO_CHANGES_NEEDED"
                    Otherwise, describe the improvement and return the COMPLETE improved code."""},
                    {"role": "user", "content": current_code}
                ]
                
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.5
                )
                
                result = response.choices[0].message.content.strip()
                
                if result == "NO_CHANGES_NEEDED" or result == current_code:
                    logger.info(f"No improvements identified for {filepath}")
                    continue
                
                # Apply the improvement
                success, error = self.apply_modification(filepath, current_code, result)
                if not success:
                    logger.error(f"Failed to apply improvement to {filepath}: {error}")
                
            except Exception as e:
                logger.error(f"Error during reflection for {filepath}: {e}")
                traceback.print_exc()
    
    def rollback_modification(self, filepath: str, target_timestamp: Optional[str] = None):
        """Rollback a file to a previous version"""
        # Find appropriate backup
        backups = sorted(self.backup_dir.glob(f"{Path(filepath).name}.*.bak"))
        
        if not backups:
            logger.error(f"No backups found for {filepath}")
            return False
        
        if target_timestamp:
            # Find specific backup
            target_backup = None
            for backup in backups:
                if target_timestamp in str(backup):
                    target_backup = backup
                    break
            
            if not target_backup:
                logger.error(f"No backup found for timestamp {target_timestamp}")
                return False
        else:
            # Use most recent backup
            target_backup = backups[-1]
        
        try:
            shutil.copy2(target_backup, filepath)
            logger.info(f"Rolled back {filepath} to {target_backup}")
            
            # Log rollback
            with open(LOG_FILE, "a") as log:
                log.write(f"[{datetime.now().isoformat()}] Rolled back {filepath} to {target_backup}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back {filepath}: {e}")
            return False
    
    def get_modification_report(self) -> Dict[str, Any]:
        """Generate report of modifications"""
        report = {
            "total_modifications": len(self.modification_history),
            "files_modified": list(set(m["file"] for m in self.modification_history)),
            "risk_distribution": {"low": 0, "medium": 0, "high": 0},
            "recent_modifications": self.modification_history[-10:],
            "session_modifications": self.modifications_this_session
        }
        
        for mod in self.modification_history:
            risk = mod.get("risk_level", "low")
            report["risk_distribution"][risk] += 1
        
        return report
    
    def restart_self(self):
        """Enhanced restart with state preservation"""
        logger.info("Preparing for restart...")
        
        # Save current state
        state = {
            "timestamp": datetime.now().isoformat(),
            "modifications_applied": self.modifications_this_session,
            "pending_restart": True
        }
        
        with open("data/restart_state.json", "w") as f:
            json.dump(state, f)
        
        # Ensure all data is saved
        self.save_modification_history()
        
        logger.info("Restarting to apply new changes...")
        
        # Get the main script name
        main_script = "main.py"
        if os.path.exists("nova_dashboard.py"):
            main_script = "nova_dashboard.py"
        
        # Restart the process
        os.execv(sys.executable, [sys.executable] + [main_script])