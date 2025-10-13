import logging
import sys
from pathlib import Path
import json
import time

# Add project root to path to allow imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from emailops.summarize_email_thread import analyze_conversation_dir, _atomic_write_text
    from emailops.utils import find_conversation_dirs
except ImportError as e:
    print(f"Error: Failed to import emailops modules. Make sure the project root is in the Python path. {e}")
    sys.exit(1)

# --- Configuration ---
LOG_DIR = project_root / "log"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"live_test_{int(time.time())}.log"
OUTLOOK_DIR = Path("C:/Users/ASUS/Desktop/Outlook")
CONVERSATION_LIMIT = 100

# --- Setup Logging ---
# Remove all handlers associated with the root logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("live_test")

def run_test():
    """
    Runs a live test on conversation directories, analyzing each one and logging the outcome.
    """
    logger.info("--- Starting Live Conversation Analysis Test ---")
    logger.info(f"Outlook Directory: {OUTLOOK_DIR}")
    logger.info(f"Conversation Limit: {CONVERSATION_LIMIT}")
    logger.info(f"Log file: {LOG_FILE}")

    if not OUTLOOK_DIR.exists() or not OUTLOOK_DIR.is_dir():
        logger.error(f"Outlook directory not found or is not a directory: {OUTLOOK_DIR}")
        return

    try:
        conversation_dirs = find_conversation_dirs(OUTLOOK_DIR)
    except Exception as e:
        logger.error(f"Failed to find conversation directories: {e}", exc_info=True)
        return

    if not conversation_dirs:
        logger.warning("No conversation directories found to process.")
        return

    logger.info(f"Found {len(conversation_dirs)} total conversations. Processing up to {CONVERSATION_LIMIT}.")

    success_count = 0
    error_count = 0
    total_processed = 0

    for i, convo_dir in enumerate(conversation_dirs[:CONVERSATION_LIMIT]):
        total_processed += 1
        logger.info(f"--- Processing Conversation {i+1}/{CONVERSATION_LIMIT}: {convo_dir.name} ---")

        try:
            # Step 1: Check if Conversation.txt exists
            if not (convo_dir / "Conversation.txt").exists():
                logger.error(f"SKIPPING: Conversation.txt not found in {convo_dir.name}")
                error_count += 1
                continue

            # Step 2: Analyze the conversation
            analysis_result = analyze_conversation_dir(thread_dir=convo_dir)

            # Step 3: Check for errors in the result
            if not analysis_result or "_metadata" not in analysis_result:
                raise ValueError("Analysis result is empty or missing metadata.")

            # Step 4: Log success and save result
            logger.info(f"SUCCESS: Analysis complete for {convo_dir.name}")
            logger.debug(f"Analysis for {convo_dir.name}: {json.dumps(analysis_result, indent=2)}")
            
            # Save the analysis to the conversation directory
            output_path = convo_dir / "summary_live_test.json"
            _atomic_write_text(output_path, json.dumps(analysis_result, indent=2, ensure_ascii=False))
            logger.info(f"Saved analysis to {output_path}")

            success_count += 1

        except Exception as e:
            logger.error(f"ERROR processing {convo_dir.name}: {e}", exc_info=True)
            error_count += 1

        logger.info(f"--- Finished Processing {convo_dir.name} ---")
        time.sleep(1) # Small delay to avoid overwhelming APIs if rate limits are tight

    logger.info("--- Live Test Summary ---")
    logger.info(f"Total Conversations Processed: {total_processed}")
    logger.info(f"Successful Analyses: {success_count}")
    logger.info(f"Failed Analyses: {error_count}")
    logger.info("--- Test Complete ---")

if __name__ == "__main__":
    run_test()
