import requests
import os
import logging
from pathlib import Path
from typing import Optional

# Log config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_valid_tle(data: str) -> bool:
    # Cheks if data are valid TLE format (3 lines per satellite)
    lines = data.strip().split('\n')
    return len(lines) >= 3 and len(lines) % 3 == 0

def fetch_tle_data(url: str, output_path: str) -> bool:
    # Dowloads TLE data from the given URL and saves it to the specified output path
    try:
        logging.info(f"Downloading data from {url}...") # instead 'print'
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Check if the request was successful, if not, it will raise an HTTPError

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(response.text)

        # Basic check
        with open(output_path, 'r') as f:
            if not is_valid_tle(f.read()):
                logging.warning("TLE format may be invalid.")

        logging.info(f"Data saved in {output_path}")
        return True

    except requests.exceptions.RequestException as e:
        logging.error(f"Network error: {e}")
        return False
    except IOError as e:
        logging.error(f"File error: {e}")
        return False

if __name__ == "__main__":
    import argparse # Allows to change the URL and output path from the command line
    parser = argparse.ArgumentParser(description="Downloading TLE data from Celestrak.")
    parser.add_argument("--url", default="https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
                        help="URL of the TLE data")
    parser.add_argument("--output", default="../data/raw/satellites.tle",
                        help="Output path for the TLE file")
    args = parser.parse_args()

    success = fetch_tle_data(args.url, args.output)
    if not success:
        logging.error("Failed to download data.")
        exit(1)
