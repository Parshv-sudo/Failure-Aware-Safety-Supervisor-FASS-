#!/usr/bin/env python
"""
Utility script to fetch OpenStreetMap (.osm) data for a given bounding box.
Using the Overpass API to download map data for CARLA.

Usage:
    python fetch_osm_map.py -b MIN_LON,MIN_LAT,MAX_LON,MAX_LAT -o maps/my_map.osm

Example:
    python fetch_osm_map.py -b -74.0135,40.7107,-74.0041,40.7183 -o maps/nyc_test.osm
"""

import argparse
import os
import requests
import sys

OVERPASS_URL = "https://overpass-api.de/api/map?bbox="

def download_osm(min_lon: float, min_lat: float, max_lon: float, max_lat: float, output_file: str):
    """
    Downloads OSM data within the specified bounding box.
    """
    # Overpass API requires bbox format: min_lon,min_lat,max_lon,max_lat
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    url = OVERPASS_URL + bbox_str
    
    print(f"Fetching from Overpass API: {url}")
    print("This may take a moment depending on the area size...")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Determine the project root from the script's path so output defaults relative to project
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # If output_file is not an absolute path, assume it's relative to the project root
        if not os.path.isabs(output_file):
            output_path = os.path.join(project_root, output_file)
        else:
            output_path = output_file
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(response.content)
            
        print(f"\n✅ Success! Saved {len(response.content)} bytes to {output_path}")
        print("\nTo use this map in the simulation, update your config/simulation_config.yaml:")
        print('  map:')
        print('    type: "real_world"')
        print(f'    source: "{output_file}"') # Provide the relative path for the config
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Failed to download OSM data: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 400:
                print("   Note: The bounding box might be too large or improperly formatted.", file=sys.stderr)
            elif e.response.status_code == 429:
                print("   Note: Overpass API rate limit exceeded. Try again later.", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Fetch OpenStreetMap map data using a bounding box.")
    
    parser.add_argument(
        "--bbox", 
        help="Bounding box: min_lon,min_lat,max_lon,max_lat (e.g., -74.0135,40.7107,-74.0041,40.7183)"
    )
    
    parser.add_argument(
        "-o", "--output", 
        default="maps/downloaded_map.osm",
        help="Output file path (e.g., maps/nyc.osm). Defaults to maps/downloaded_map.osm"
    )
    
    # Workaround for negative numbers being parsed as flags
    argv = sys.argv
    bbox_str = None
    output_path = "maps/downloaded_map.osm"
    
    for i, arg in enumerate(argv):
        if arg == "-o" or arg == "--output":
            if i + 1 < len(argv):
                output_path = argv[i+1]
        elif arg == "--bbox" or arg == "-b":
             if i + 1 < len(argv):
                 bbox_str = argv[i+1]
        elif not arg.startswith('-') and bbox_str is None and "," in arg:
            bbox_str = arg
            
    if bbox_str is None:
        print("Error: Bounding box must be provided.", file=sys.stderr)
        print("Usage: python fetch_osm_map.py --bbox min_lon,min_lat,max_lon,max_lat [-o output_file.osm]", file=sys.stderr)
        sys.exit(1)
    
    try:
        parts = bbox_str.split(',')
        if len(parts) != 4:
            raise ValueError("Bounding box must have exactly 4 values separated by commas.")
        
        min_lon = float(parts[0].strip())
        min_lat = float(parts[1].strip())
        max_lon = float(parts[2].strip())
        max_lat = float(parts[3].strip())
        
        # Quick sanity check on coords
        if min_lon > max_lon:
            print("Warning: min_lon is greater than max_lon. Swapping them.", file=sys.stderr)
            min_lon, max_lon = max_lon, min_lon
            
        if min_lat > max_lat:
            print("Warning: min_lat is greater than max_lat. Swapping them.", file=sys.stderr)
            min_lat, max_lat = max_lat, min_lat
            
    except ValueError as e:
        print(f"Error parsing bounding box: {e}", file=sys.stderr)
        sys.exit(1)
        
    download_osm(min_lon, min_lat, max_lon, max_lat, output_path)

if __name__ == "__main__":
    main()
