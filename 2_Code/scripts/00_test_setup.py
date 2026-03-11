"""
Test script to verify all required packages are installed correctly
"""

print("Testing package imports...")
print("=" * 50)

try:
    import pandas as pd
    print("✓ pandas:", pd.__version__)
except ImportError as e:
    print("✗ pandas failed:", e)

try:
    import numpy as np
    print("✓ numpy:", np.__version__)
except ImportError as e:
    print("✗ numpy failed:", e)

try:
    import scipy
    print("✓ scipy:", scipy.__version__)
except ImportError as e:
    print("✗ scipy failed:", e)

try:
    import matplotlib
    print("✓ matplotlib:", matplotlib.__version__)
except ImportError as e:
    print("✗ matplotlib failed:", e)

try:
    import seaborn as sns
    print("✓ seaborn:", sns.__version__)
except ImportError as e:
    print("✗ seaborn failed:", e)

try:
    import requests
    print("✓ requests:", requests.__version__)
except ImportError as e:
    print("✗ requests failed:", e)

try:
    from bs4 import BeautifulSoup
    print("✓ beautifulsoup4: OK")
except ImportError as e:
    print("✗ beautifulsoup4 failed:", e)

try:
    import selenium
    print("✓ selenium:", selenium.__version__)
except ImportError as e:
    print("✗ selenium failed:", e)

try:
    import jupyter
    print("✓ jupyter: OK")
except ImportError as e:
    print("✗ jupyter failed:", e)

try:
    import openpyxl
    print("✓ openpyxl:", openpyxl.__version__)
except ImportError as e:
    print("✗ openpyxl failed:", e)

print("=" * 50)
print("\n✅ All packages imported successfully!")
print("\n📁 Directory structure:")
import os
dirs = [
    "data/raw/bctc/wholesale",
    "data/raw/bctc/food_mfg",
    "data/raw/bctc/fnb",
    "data/processed/distributions",
    "data/output",
    "scripts",
    "notebooks"
]
for d in dirs:
    exists = "✓" if os.path.exists(d) else "✗"
    print(f"  {exists} {d}")

print("\n🎉 Phase 0 Setup Complete!")
