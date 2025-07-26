# utils/error_handling.py
"""Enhanced error handling and retry logic for Nova AI Agent"""

import logging
import time
import functools
import asyncio
from typing import Any, Callable, Optional, Type, Union, Tuple, Dict
from dataclasses import dataclass
import threading
import traceback
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_factor: float = 1.0


class NovaException(Exception):
    """Base exception for Nova AI Agent"""
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()


class APIError(NovaException):
    """API-related errors"""
    pass


class MemoryError(NovaException):
    """Memory management errors"""
    pass


class ConfigurationError(NovaException):
    """Configuration-related errors"""
    pass


class ValidationError(NovaException):
    """Data validation errors"""
    pass


class NetworkError(NovaException):
    """Network-related errors"""
    pass


class RateLimitError(NovaException):
    """Rate limiting errors"""
    pass


class ErrorTracker:
    """Thread-safe error tracking and analytics"""
    
    def __init__(self, max_errors: int = 1000):
        self.max_errors = max_errors
        self.errors = []
        self._lock = threading.RLock()
        self.error_counts = {}
        self.last_cleanup = datetime.now()
    
    def record_error(self, error: Exception, context: Dict[str, Any] = None):
        """Record an error with context"""
        with self._lock:
            error_record = {
                "timestamp": datetime.now().isoformat(),
                "type": type(error).__name__,
                "message": str(error),
                "context": context or {},
                "traceback": traceback.format_exc()
            }
            
            self.errors.append(error_record)
            
            # Update error counts
            error_type = type(error).__name__
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            # Cleanup old errors
            if len(self.errors) > self.max_errors:
                self.errors = self.errors[-self.max_errors:]
            
            # Periodic cleanup of old error counts
            if (datetime.now() - self.last_cleanup).total_seconds() > 3600:
                self._cleanup_old_counts()
    
    def _cleanup_old_counts(self):
        """Clean up old error counts (keep last 24 hours)"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        recent_errors = [
            e for e in self.errors
            if datetime.fromisoformat(e["timestamp"]) > cutoff_time
        ]
        
        # Recalculate counts
        new_counts = {}
        for error in recent_errors:
            error_type = error["type"]
            new_counts[error_type] = new_counts.get(error_type, 0) + 1
        
        self.error_counts = new_counts
        self.last_cleanup = datetime.now()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        with self._lock:
            return {
                "total_errors": len(self.errors),
                "error_counts": self.error_counts.copy(),
                "recent_errors": self.errors[-10:],  # Last 10 errors
                "last_cleanup": self.last_cleanup.isoformat()
            }
    
    def get_error_rate(self, error_type: str = None, hours: int = 1) -> float:
        """Get error rate for a specific type or all errors"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            if error_type:
                recent_errors = [
                    e for e in self.errors
                    if (e["type"] == error_type and 
                        datetime.fromisoformat(e["timestamp"]) > cutoff_time)
                ]
            else:
                recent_errors = [
                    e for e in self.errors
                    if datetime.fromisoformat(e["timestamp"]) > cutoff_time
                ]
            
            return len(recent_errors) / hours


# Global error tracker
error_tracker = ErrorTracker()


def retry_with_backoff(
    retry_config: RetryConfig = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None,
    on_failure: Optional[Callable] = None
):
    """Decorator for retry logic with exponential backoff"""
    
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    error_tracker.record_error(e, {
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "max_attempts": retry_config.max_attempts
                    })
                    
                    if attempt == retry_config.max_attempts - 1:
                        # Last attempt failed
                        if on_failure:
                            on_failure(e, attempt + 1)
                        break
                    
                    # Calculate delay
                    delay = min(
                        retry_config.base_delay * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay
                    )
                    
                    if retry_config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
                    
                    delay *= retry_config.backoff_factor
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{retry_config.max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    if on_retry:
                        on_retry(e, attempt + 1, delay)
                    
                    time.sleep(delay)
            
            # All attempts failed
            raise last_exception
        
        return wrapper
    return decorator


def async_retry_with_backoff(
    retry_config: RetryConfig = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None,
    on_failure: Optional[Callable] = None
):
    """Async decorator for retry logic with exponential backoff"""
    
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    error_tracker.record_error(e, {
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "max_attempts": retry_config.max_attempts,
                        "async": True
                    })
                    
                    if attempt == retry_config.max_attempts - 1:
                        # Last attempt failed
                        if on_failure:
                            if asyncio.iscoroutinefunction(on_failure):
                                await on_failure(e, attempt + 1)
                            else:
                                on_failure(e, attempt + 1)
                        break
                    
                    # Calculate delay
                    delay = min(
                        retry_config.base_delay * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay
                    )
                    
                    if retry_config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    delay *= retry_config.backoff_factor
                    
                    logger.warning(
                        f"Async attempt {attempt + 1}/{retry_config.max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    if on_retry:
                        if asyncio.iscoroutinefunction(on_retry):
                            await on_retry(e, attempt + 1, delay)
                        else:
                            on_retry(e, attempt + 1, delay)
                    
                    await asyncio.sleep(delay)
            
            # All attempts failed
            raise last_exception
        
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time = None
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.RLock()
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self._lock:
                if self._state == "OPEN":
                    if self._should_attempt_reset():
                        self._state = "HALF_OPEN"
                    else:
                        raise APIError(f"Circuit breaker is OPEN for {func.__name__}")
                
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                
                except self.expected_exception as e:
                    self._on_failure()
                    raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self._last_failure_time is None:
            return True
        
        return (time.time() - self._last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Reset circuit breaker on successful call"""
        self._failure_count = 0
        self._state = "CLOSED"
    
    def _on_failure(self):
        """Handle failure in circuit breaker"""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
    
    @property
    def state(self) -> str:
        """Get current circuit breaker state"""
        with self._lock:
            return self._state


class SafeExecutor:
    """Safe execution wrapper with comprehensive error handling"""
    
    def __init__(self, name: str = "SafeExecutor"):
        self.name = name
        self.metrics = {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "total_time": 0.0
        }
        self._lock = threading.RLock()
    
    def execute(
        self,
        func: Callable,
        *args,
        default_return: Any = None,
        log_errors: bool = True,
        context: Dict[str, Any] = None,
        **kwargs
    ) -> Any:
        """Safely execute a function with error handling"""
        start_time = time.time()
        
        with self._lock:
            self.metrics["executions"] += 1
        
        try:
            result = func(*args, **kwargs)
            
            with self._lock:
                self.metrics["successes"] += 1
                self.metrics["total_time"] += time.time() - start_time
            
            return result
            
        except Exception as e:
            with self._lock:
                self.metrics["failures"] += 1
                self.metrics["total_time"] += time.time() - start_time
            
            if log_errors:
                logger.error(f"Error in {self.name}: {e}", exc_info=True)
            
            error_tracker.record_error(e, {
                "executor": self.name,
                "function": getattr(func, "__name__", str(func)),
                "context": context or {}
            })
            
            return default_return
    
    async def execute_async(
        self,
        func: Callable,
        *args,
        default_return: Any = None,
        log_errors: bool = True,
        context: Dict[str, Any] = None,
        **kwargs
    ) -> Any:
        """Safely execute an async function with error handling"""
        start_time = time.time()
        
        with self._lock:
            self.metrics["executions"] += 1
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            with self._lock:
                self.metrics["successes"] += 1
                self.metrics["total_time"] += time.time() - start_time
            
            return result
            
        except Exception as e:
            with self._lock:
                self.metrics["failures"] += 1
                self.metrics["total_time"] += time.time() - start_time
            
            if log_errors:
                logger.error(f"Error in async {self.name}: {e}", exc_info=True)
            
            error_tracker.record_error(e, {
                "executor": self.name,
                "function": getattr(func, "__name__", str(func)),
                "context": context or {},
                "async": True
            })
            
            return default_return
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        with self._lock:
            metrics = self.metrics.copy()
            
            if metrics["executions"] > 0:
                metrics["success_rate"] = metrics["successes"] / metrics["executions"]
                metrics["failure_rate"] = metrics["failures"] / metrics["executions"]
                metrics["avg_execution_time"] = metrics["total_time"] / metrics["executions"]
            else:
                metrics["success_rate"] = 0.0
                metrics["failure_rate"] = 0.0
                metrics["avg_execution_time"] = 0.0
            
            return metrics


def handle_exception(
    exception: Exception,
    context: str = "",
    reraise: bool = False,
    log_level: int = logging.ERROR
) -> Optional[Exception]:
    """Centralized exception handling"""
    
    error_message = f"Exception in {context}: {exception}" if context else str(exception)
    
    # Log based on exception type
    if isinstance(exception, (APIError, NetworkError)):
        logger.log(log_level, f"External service error: {error_message}")
    elif isinstance(exception, (MemoryError, ValidationError)):
        logger.log(log_level, f"Internal error: {error_message}")
    elif isinstance(exception, RateLimitError):
        logger.warning(f"Rate limit exceeded: {error_message}")
    else:
        logger.log(log_level, f"Unexpected error: {error_message}", exc_info=True)
    
    # Record error
    error_tracker.record_error(exception, {"context": context})
    
    if reraise:
        raise exception
    
    return exception


class ErrorReporter:
    """Error reporting and alerting system"""
    
    def __init__(self, report_file: str = "data/error_reports.json"):
        self.report_file = report_file
        self.alert_thresholds = {
            "error_rate_per_hour": 10,
            "consecutive_failures": 5,
            "critical_errors": 1
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        summary = error_tracker.get_error_summary()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "alerts": self._check_alerts(),
            "recommendations": self._generate_recommendations(),
            "system_health": self._assess_system_health()
        }
        
        return report
    
    def _check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        
        # Check error rate
        error_rate = error_tracker.get_error_rate(hours=1)
        if error_rate > self.alert_thresholds["error_rate_per_hour"]:
            alerts.append({
                "type": "high_error_rate",
                "severity": "warning",
                "message": f"Error rate ({error_rate:.1f}/hour) exceeds threshold",
                "threshold": self.alert_thresholds["error_rate_per_hour"]
            })
        
        # Check for critical errors
        summary = error_tracker.get_error_summary()
        critical_errors = ["APIError", "MemoryError", "ConfigurationError"]
        
        for error_type in critical_errors:
            if error_type in summary["error_counts"]:
                count = summary["error_counts"][error_type]
                if count > 0:
                    alerts.append({
                        "type": "critical_error",
                        "severity": "critical",
                        "message": f"{count} {error_type} occurrences detected",
                        "error_type": error_type
                    })
        
        return alerts
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on error patterns"""
        recommendations = []
        summary = error_tracker.get_error_summary()
        
        if "APIError" in summary["error_counts"]:
            recommendations.append("Check API key validity and network connectivity")
        
        if "MemoryError" in summary["error_counts"]:
            recommendations.append("Consider increasing memory limits or cleanup frequency")
        
        if "RateLimitError" in summary["error_counts"]:
            recommendations.append("Implement request throttling or upgrade API plan")
        
        error_rate = error_tracker.get_error_rate(hours=24)
        if error_rate > 50:
            recommendations.append("High error rate detected - review system logs and consider maintenance")
        
        return recommendations
    
    def _assess_system_health(self) -> Dict[str, str]:
        """Assess overall system health"""
        summary = error_tracker.get_error_summary()
        total_errors = summary["total_errors"]
        error_rate = error_tracker.get_error_rate(hours=1)
        
        if error_rate == 0:
            health_status = "excellent"
        elif error_rate < 5:
            health_status = "good"
        elif error_rate < 10:
            health_status = "fair"
        elif error_rate < 20:
            health_status = "poor"
        else:
            health_status = "critical"
        
        return {
            "status": health_status,
            "total_errors": total_errors,
            "hourly_error_rate": error_rate
        }
    
    def save_report(self, report: Dict[str, Any] = None):
        """Save error report to file"""
        if report is None:
            report = self.generate_report()
        
        try:
            # Load existing reports
            reports = []
            if os.path.exists(self.report_file):
                try:
                    with open(self.report_file, 'r') as f:
                        reports = json.load(f)
                except Exception:
                    reports = []
            
            # Add new report
            reports.append(report)
            
            # Keep only last 100 reports
            if len(reports) > 100:
                reports = reports[-100:]
            
            # Save updated reports
            os.makedirs(os.path.dirname(self.report_file), exist_ok=True)
            with open(self.report_file, 'w') as f:
                json.dump(reports, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")


# Global error reporter
error_reporter = ErrorReporter()


def setup_error_handling():
    """Setup global error handling"""
    # Set up uncaught exception handler
    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        error_tracker.record_error(exc_value, {"uncaught": True})
    
    import sys
    sys.excepthook = handle_uncaught_exception
    
    # Set up asyncio exception handler
    def handle_asyncio_exception(loop, context):
        exception = context.get('exception')
        if exception:
            logger.error(f"Asyncio exception: {exception}", exc_info=exception)
            error_tracker.record_error(exception, {"asyncio": True, "context": context})
        else:
            logger.error(f"Asyncio error: {context}")
    
    # This will be applied when event loop is created
    return handle_asyncio_exception


# Convenience functions
def safe_call(func: Callable, *args, default=None, **kwargs):
    """Safely call a function with default return value"""
    executor = SafeExecutor("safe_call")
    return executor.execute(func, *args, default_return=default, **kwargs)


async def safe_call_async(func: Callable, *args, default=None, **kwargs):
    """Safely call an async function with default return value"""
    executor = SafeExecutor("safe_call_async")
    return await executor.execute_async(func, *args, default_return=default, **kwargs)