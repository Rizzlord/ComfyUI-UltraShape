import os
import sys

# Add the UltraShape-1.0 directory to sys.path so 'ultrashape' can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
ultrashape_path = os.path.join(current_dir, "UltraShape-1.0")
if ultrashape_path not in sys.path:
    sys.path.insert(0, ultrashape_path)

from .ultra_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
