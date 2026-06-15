#!/usr/bin/env python3
"""
RAG Multi-Agent Chatbot System
Main entry point for the application
"""

import uvicorn
import logging
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.main import app

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_chatbot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main function to start the FastAPI server"""
    try:
        logger.info("Starting RAG Multi-Agent Chatbot System...")
        logger.info("Available endpoints:")
        logger.info("  - POST /chat: Main chatbot endpoint")
        logger.info("  - POST /mock_chat: Mock chatbot endpoint")
        logger.info("  - GET /health: Health check")
        logger.info("  - GET /agents: List available agents")
        logger.info("  - GET /: API documentation")

        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8501,
            log_level="info",
            reload=False  # Set to True for development
        )

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()