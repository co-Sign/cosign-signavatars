import os
import argparse
import subprocess
import sys
import platform
import pkg_resources
import shutil

def print_section(title):
    """Print a section title."""
    print(f"\n{'=' * 80}")
    print(f"{title.center(80)}")
    print(f"{'=' * 80}\n")

def run_command(command, description=None):
    """
    Run a command and handle errors.
    
    Args:
        command (str): Command to run
        description (str, optional): Description of the command
        
    Returns:
        bool: True if successful, False otherwise
    """
    if description:
        print(f"\n> {description}")
    print(f"$ {command}")
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        return_code = process.poll()
        
        if return_code != 0:
            error = process.stderr.read()
            print(f"Error (return code {return_code}):")
            print(error)
            return False
        
        return True
    
    except Exception as e:
        print(f"Exception running command: {e}")
        return False

def check_cuda():
    """Check if CUDA is available."""
    print_section("Checking CUDA Availability")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA is available")
            device_count = torch.cuda.device_count()
            print(f"   - Number of CUDA devices: {device_count}")
            for i in range(device_count):
                print(f"   - CUDA Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"⚠️  CUDA is not available. Training will be slower on CPU.")
        
        return cuda_available
    except ImportError:
        print(f"⚠️  PyTorch is not installed. Cannot check CUDA availability.")
        return False

def create_directories():
    """Create necessary directories if they don't exist."""
    print_section("Creating Project Directories")
    
    directories = [
        'preprocessing',
        'models',
        'utils',
        'web_app',
        'web_app/templates',
        'web_app/static',
        'web_app/static/css',
        'web_app/static/js',
        'outputs',
        'data',
        'data/raw',
        'data/preprocessed'
    ]
    
    for directory in directories:
        dir_path = os.path.join(os.path.dirname(__file__), directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def check_and_install_packages():
    """Check and install required packages."""
    print_section("Checking and Installing Dependencies")
    
    # Check if requirements.txt exists
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if not os.path.exists(req_path):
        print(f"⚠️  requirements.txt not found at {req_path}")
        return False
    
    # Get installed packages
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    # Read requirements
    with open(req_path, 'r') as f:
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.strip().startswith('#')
        ]
    
    # Check each requirement
    missing_packages = []
    
    for req in requirements:
        # Handle requirements with version specifiers
        package_name = req.split('==')[0].split('>')[0].split('<')[0].split('~=')[0].strip()
        
        if package_name.lower() not in [pkg.lower() for pkg in installed_packages]:
            missing_packages.append(req)
    
    # Install missing packages
    if missing_packages:
        print(f"Found {len(missing_packages)} missing packages. Installing...")
        
        for pkg in missing_packages:
            success = run_command(f"{sys.executable} -m pip install {pkg}", f"Installing {pkg}")
            if not success:
                print(f"⚠️  Failed to install {pkg}")
    else:
        print("All required packages are already installed.")
    
    return True

def create_env_file():
    """Create a template .env file if it doesn't exist."""
    print_section("Creating Environment File")
    
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write("""# API Keys for ASL Translation
# Uncomment and add your API keys as needed

# OpenAI API key (for English to ASL gloss conversion)
# OPENAI_API_KEY=your_openai_api_key

# Speech Recognition API keys (optional)
# WIT_API_KEY=your_wit_api_key
# AZURE_SPEECH_KEY=your_azure_speech_key
# AZURE_SPEECH_REGION=your_azure_region
# IBM_SPEECH_USERNAME=your_ibm_username
# IBM_SPEECH_PASSWORD=your_ibm_password
""")
        print(f"Created template .env file at {env_path}")
    else:
        print(f".env file already exists at {env_path}")

def initialize_project():
    """Initialize the project environment and structure."""
    print_section("ASL Translation Project Initialization")
    
    # Print system information
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Create directories
    create_directories()
    
    # Check and install packages
    check_and_install_packages()
    
    # Check CUDA
    check_cuda()
    
    # Create .env file
    create_env_file()
    
    print_section("Project Initialization Complete")
    print("You can now start using the ASL Translation System.")
    print("\nNext steps:")
    print("1. Add your API keys to the .env file")
    print("2. Place your raw PKL files in the data/raw directory")
    print("3. Run preprocessing: python run_pipeline.py --raw_data_dir data/raw --output_dir outputs")
    print("\nFor more information, see the README.md file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize the ASL Translation project")
    args = parser.parse_args()
    
    initialize_project() 