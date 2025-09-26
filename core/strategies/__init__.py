"""Core strategies package for MT5 backtesting.

This module re-exports strategy classes from the main strategies package
for backward compatibility.
"""

# Import the strategies using absolute imports
try:
    from strategies.pinbar_configurable import ConfigurableTimeframePinbarStrategy
    from strategies.timed_entry import TimedEntryStrategy5m
    
    # Re-export the classes
    __all__ = ["TimedEntryStrategy5m", "ConfigurableTimeframePinbarStrategy"]
    
except ImportError as e:
    # Fallback to empty classes if imports fail
    print(f"Warning: Could not import strategies: {e}")
    
    class TimedEntryStrategy5m:
        pass
        
    class ConfigurableTimeframePinbarStrategy:
        pass
        
    __all__ = ["TimedEntryStrategy5m", "ConfigurableTimeframePinbarStrategy"]
    
    # Re-raise the import error to make it clear something is wrong
    raise ImportError(
        "Failed to import strategy classes. Make sure the project root is in your PYTHONPATH. "
        f"Current sys.path: {sys.path}"
    ) from e
