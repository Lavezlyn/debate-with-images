"""
Logger Module

Provide unified logging configuration and management functionality, only supporting console output.
"""

import logging
from typing import Optional


class DebateLogger:
    """Debate system专用日志器类"""
    
    def __init__(self, name: str = "DebateRunner"):
        """
        Initialize logger
        
        Args:
            name: logger name
        """
        self.name = name
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logger, only supporting console output
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        
        # Prevent logging from propagating to parent logger, avoid duplicate output
        logger.propagate = False
        
        # Clear existing handlers to avoid duplicate
        logger.handlers.clear()
        
        # Console handler - the only output method
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message: str):
        """Log info level message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug level message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning level message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error level message"""
        self.logger.error(message)
    
    def exception(self, message: str):
        """Log exception message (including stack trace)"""
        self.logger.exception(message)
    
    def log_debate_start(self, total_rounds: int, total_cases: int, debaters: list):
        """Log debate start information"""
        self.info("=" * 50)
        self.info("🚀 Debate System Started")
        self.info(f"📊 Debate Configuration: {total_rounds} rounds, {total_cases} cases, {len(debaters)} debaters")
        for debater in debaters:
            self.info(f"  👤 Debater: {debater}")
        self.info("=" * 50)
    
    def log_initialization_start(self):
        """Log initialization start"""
        self.info("=== Starting Debate Environment Initialization ===")
    
    def log_initialization_step(self, step_name: str, count: int):
        """Log initialization step"""
        self.info(f"Completed {step_name}: {count} items")
    
    def log_debate_rounds_start(self, total_rounds: int):
        """Log debate rounds start"""
        self.info(f"=== Starting {total_rounds} Rounds of Debate ===")
    
    def log_round_start(self, round_idx: int, total_rounds: int):
        """Log single round start"""
        self.info(f"🔥 Round {round_idx + 1}/{total_rounds} Started")
    
    def log_round_end(self, round_idx: int, duration: float):
        """Log single round end"""
        self.info(f"✅ Round {round_idx + 1} Completed (Duration: {duration:.2f}s)")
    
    def log_debater_start(self, role: str):
        """Log debater start speaking"""
        self.info(f"  👤 {role} Started Speaking")
    
    def log_debater_inference_start(self, backend: str):
        """Log inference start"""
        self.info(f"  🤖 Calling {backend} Backend for Inference...")
    
    def log_debater_inference_end(self, role: str, duration: float):
        """Log inference end"""
        self.info(f"  ✅ {role} Inference Completed (Duration: {duration:.2f}s)")
    
    def log_debate_complete(self, total_duration: float, stats: dict):
        """Log debate completion"""
        self.info(f"🎉 All Debates Completed! Total Duration: {total_duration:.2f}s")
        self.info(f"📊 Statistics: {stats.get('cases', 0)} cases, {stats.get('rounds', 0)} rounds, {stats.get('debaters', 0)} debaters")
    
    def log_output_saved(self, output_path: str):
        """Log output saved"""
        self.info(f"💾 Debate Trajectory Saved to: {output_path}")
    
    def log_program_complete(self):
        """Log program completion"""
        self.info("✨ Program Execution Completed")
    
    def log_error(self, error_msg: str):
        """Log program error"""
        self.error(f"❌ Error Occurred During Debate: {error_msg}")
    
    def log_payload_built(self, count: int):
        """Log payload built (debug level)"""
        self.debug(f"  Built {count} inference payloads")
    
    def log_history_updated(self, history_lengths: list):
        """Log history updated (debug level)"""
        self.debug(f"  Updated history records, current history lengths: {history_lengths}")


def create_debate_logger(name: str = "DebateRunner") -> DebateLogger:
    """
    Factory function to create debate logger
    
    Args:
        name: logger name
    
    Returns:
        DebateLogger instance
    """
    return DebateLogger(name)


def setup_basic_logging(level: int = logging.INFO):
    """
    Set up basic global logging configuration
    
    Args:
        level: logging level
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Global function for compatibility
def get_default_logger() -> DebateLogger:
    """Get default debate logger"""
    return create_debate_logger()


if __name__ == "__main__":
    # Test logger functionality
    logger = create_debate_logger("TestLogger")
    
    logger.info("Test info log")
    logger.debug("Test debug log")
    logger.warning("Test warning log")
    
    # Test specialized methods
    logger.log_debate_start(3, 5, ["D_aff", "D_neg"])
    logger.log_initialization_start()
    logger.log_initialization_step("initial user prompts", 5)
    
    print("Logger testing completed - console output only")
