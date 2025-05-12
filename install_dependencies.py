"""
Dependency Installation Script for Sentiment Analysis Engine
-----------------------------------------------------------
This script automatically installs all required dependencies for the sentiment
analysis engine on Windows. It handles NLTK data downloads and special package
requirements.
"""

import os
import sys
import subprocess
import time
import platform

def install_package(package, prefer_binary=False):
    """Install a package using pip with error handling"""
    
    print(f"Installing {package}...")
    cmd = [sys.executable, "-m", "pip", "install"]
    
    if prefer_binary:
        cmd.append("--prefer-binary")
    
    cmd.append(package)
    
    try:
        subprocess.check_call(cmd)
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def install_nltk_data():
    """Install required NLTK data"""
    
    print("\nDownloading required NLTK data...")
    try:
        import nltk
        
        # Download required NLTK data
        for data_pkg in ['punkt', 'stopwords']:
            try:
                nltk.download(data_pkg, quiet=True)
                print(f"✅ Successfully downloaded nltk:{data_pkg}")
            except Exception as e:
                print(f"❌ Failed to download nltk:{data_pkg} - {str(e)}")
    except ImportError:
        print("❌ NLTK not installed properly")
        return False
    
    return True

def read_requirements():
    """Read requirements.txt file"""
    
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    if not os.path.exists(req_file):
        print("❌ requirements.txt not found")
        return []
    
    with open(req_file, 'r') as f:
        lines = f.readlines()
    
    # Filter out comments and empty lines
    requirements = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)
    
    return requirements

def main():
    """Main installation function"""
    
    print("=" * 80)
    print("Sentiment Analysis Engine Dependency Installer")
    print("=" * 80)
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print("=" * 80)
    
    # Read requirements
    requirements = read_requirements()
    if not requirements:
        print("No requirements found. Exiting.")
        return 1
    
    print(f"Found {len(requirements)} packages to install")
    
    # First, install critical dependencies
    critical_deps = ["numpy", "pandas", "nltk", "streamlit"]
    
    print("\nInstalling critical dependencies...")
    for dep in critical_deps:
        matching_reqs = [req for req in requirements if req.startswith(dep)]
        if matching_reqs:
            install_package(matching_reqs[0], prefer_binary=True)
        else:
            install_package(dep, prefer_binary=True)
    
    # Install remaining requirements
    print("\nInstalling remaining dependencies...")
    for req in requirements:
        if not any(req.startswith(dep) for dep in critical_deps):
            install_package(req, prefer_binary=True)
    
    # Install NLTK data
    if not install_nltk_data():
        print("\n⚠️ Warning: Failed to download some NLTK data. Some features may not work properly.")
    
    print("\n" + "=" * 80)
    print("Installation completed!")
    print("You can now run the sentiment analysis engine with:")
    print("   python run_advanced.py dashboard")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 