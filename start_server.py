#!/usr/bin/env python3
"""
Production server startup script for Mask Grouping Server.

This script provides options for running the server in different modes:
- Development mode with Flask's built-in server
- Production mode with Waitress WSGI server
- Production mode with Gunicorn WSGI server (Linux/Mac only)
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def start_development_server(host='0.0.0.0', port=5003, debug=True):
    """Start Flask development server."""
    print("Starting Mask Grouping Server in DEVELOPMENT mode...")
    print("⚠️  This is NOT suitable for production use!")
    
    from app import app
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )

def start_waitress_server(host='0.0.0.0', port=5003, threads=4):
    """Start production server with Waitress WSGI."""
    print("Starting Mask Grouping Server with Waitress WSGI server...")
    print(f"🚀 Server will be available at: http://{host}:{port}")
    
    try:
        from waitress import serve
        from app import app
        
        serve(
            app,
            host=host,
            port=port,
            threads=threads,
            connection_limit=1000,
            cleanup_interval=30,
            channel_timeout=120
        )
    except ImportError:
        print("❌ Waitress not installed. Install with: pip install waitress")
        sys.exit(1)

def start_gunicorn_server(host='0.0.0.0', port=5003, workers=2):
    """Start production server with Gunicorn WSGI."""
    print("Starting Mask Grouping Server with Gunicorn WSGI server...")
    print(f"🚀 Server will be available at: http://{host}:{port}")
    
    try:
        import gunicorn.app.wsgiapp as wsgi
        
        # Set gunicorn arguments
        sys.argv = [
            'gunicorn',
            '--bind', f'{host}:{port}',
            '--workers', str(workers),
            '--worker-class', 'sync',
            '--timeout', '300',
            '--keep-alive', '2',
            '--max-requests', '1000',
            '--max-requests-jitter', '100',
            '--preload',
            'app:app'
        ]
        
        # Run gunicorn
        wsgi.run()
        
    except ImportError:
        print("❌ Gunicorn not installed. Install with: pip install gunicorn")
        sys.exit(1)

def check_requirements():
    """Check if all required dependencies are installed."""
    required_packages = [
        'flask',
        'torch',
        'opencv-python',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'segment_anything'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)
    
    print("✅ All required packages are installed")

def main():
    parser = argparse.ArgumentParser(
        description="Start Mask Grouping Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Development server (with auto-reload)
  python start_server.py --mode dev
  
  # Production server with Waitress
  python start_server.py --mode waitress --host 0.0.0.0 --port 5003
  
  # Production server with Gunicorn (Linux/Mac)
  python start_server.py --mode gunicorn --workers 4
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['dev', 'waitress', 'gunicorn'],
        default='waitress',
        help='Server mode (default: waitress)'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5003,
        help='Port to bind to (default: 5003)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='Number of worker processes for Gunicorn (default: 2)'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help='Number of threads for Waitress (default: 4)'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip dependency checks'
    )
    
    args = parser.parse_args()
    
    # Check requirements unless skipped
    if not args.skip_checks:
        check_requirements()
    
    # Create necessary directories
    os.makedirs('model', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("="*60)
    print("🧠 MASK GROUPING SERVER")
    print("   Advanced SAM-based mask analysis with overlap detection")
    print("   Based on mask_size_grouping.py functionality")
    print("="*60)
    
    # Start server based on mode
    if args.mode == 'dev':
        start_development_server(args.host, args.port)
    elif args.mode == 'waitress':
        start_waitress_server(args.host, args.port, args.threads)
    elif args.mode == 'gunicorn':
        start_gunicorn_server(args.host, args.port, args.workers)

if __name__ == '__main__':
    main() 