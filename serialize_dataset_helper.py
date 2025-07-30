#!/usr/bin/env python3
"""
Helper script to search CasualConversations transcriptions for specific keywords
"""

import json
import os
import logging
import re
from typing import List, Dict, Set, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_transcriptions(json_path: str) -> List[Dict]:
    """
    Load transcriptions from JSON file
    
    Args:
        json_path (str): Path to the transcriptions JSON file
        
    Returns:
        List[Dict]: List of transcription entries
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Transcriptions file not found: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            transcriptions = json.load(f)
        
        logger.info(f"✅ Loaded {len(transcriptions)} transcription entries from: {json_path}")
        return transcriptions
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON file: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading transcriptions: {e}")
        raise


def parse_timestamp_for_keywords(transcription_timestamp: str, keywords: List[str]) -> List[Tuple[str, float]]:
    """
    Parse transcription_timestamp to find keyword occurrences with timestamps
    Returns the NEXT timestamp after the keyword is mentioned (when expression occurs)
    
    Args:
        transcription_timestamp (str): The timestamped transcription text
        keywords (List[str]): List of keywords to search for
        
    Returns:
        List[Tuple[str, float]]: List of (keyword, next_timestamp) pairs
    """
    keyword_occurrences = []
    
    # Split the text by timestamp markers [X.XXX]
    # This regex finds all timestamp patterns like [0.000], [3.555], etc.
    timestamp_pattern = r'\[(\d+\.\d+)\]'
    
    # Find all timestamps and their positions
    timestamps = list(re.finditer(timestamp_pattern, transcription_timestamp))
    
    for i, timestamp_match in enumerate(timestamps):
        timestamp_sec = float(timestamp_match.group(1))
        start_pos = timestamp_match.end()
        
        # Find the end position (next timestamp or end of string)
        if i + 1 < len(timestamps):
            end_pos = timestamps[i + 1].start()
        else:
            end_pos = len(transcription_timestamp)
        
        # Extract the text segment between this timestamp and the next
        text_segment = transcription_timestamp[start_pos:end_pos].lower()
        
        # Check for keywords in this segment
        for keyword in keywords:
            if keyword.lower() in text_segment:
                # Get the NEXT timestamp (when expression occurs)
                if i + 1 < len(timestamps):
                    next_timestamp = float(timestamps[i + 1].group(1))
                else:
                    # If this is the last timestamp, use the current one
                    next_timestamp = timestamp_sec
                
                keyword_occurrences.append((keyword, next_timestamp))
    
    return keyword_occurrences


def search_transcriptions_for_keywords(transcriptions: List[Dict], keywords: List[str]) -> Dict[str, List[Dict]]:
    """
    Search transcriptions for specific keywords
    Requires ALL keywords to be found in the same transcript
    
    Args:
        transcriptions (List[Dict]): List of transcription entries
        keywords (List[str]): List of keywords to search for (ALL must be present)
        
    Returns:
        Dict[str, List[Dict]]: Dictionary mapping each keyword to list of matching entries with video_path and keyword occurrences
    """
    results = {keyword: [] for keyword in keywords}
    
    logger.info(f"🔍 Searching {len(transcriptions)} transcriptions for ALL keywords: {keywords}")
    
    for entry in transcriptions:
        video_path = entry.get('video_path', '')
        transcription = entry.get('transcription', '').lower()
        transcription_timestamp = entry.get('transcription_timestamp', '')
        
        # Check if ALL keywords are present in this transcript
        all_keywords_found = all(keyword.lower() in transcription for keyword in keywords)
        
        if all_keywords_found:
            # Parse timestamp occurrences for each keyword
            keyword_occurrences = parse_timestamp_for_keywords(transcription_timestamp, keywords)
            
            # Group occurrences by keyword
            occurrences_by_keyword = {}
            for keyword, timestamp in keyword_occurrences:
                if keyword not in occurrences_by_keyword:
                    occurrences_by_keyword[keyword] = []
                occurrences_by_keyword[keyword].append(timestamp)
            
            # Store entry for each keyword since all are present
            for keyword in keywords:
                results[keyword].append({
                    'video_path': video_path,
                    'subject_id': entry.get('subject_id', ''),
                    'age': entry.get('age', ''),
                    'gender': entry.get('gender', ''),
                    'skin-type': entry.get('skin-type', ''),
                    'is_dark': entry.get('is_dark', False),
                    'keyword_occurrences': occurrences_by_keyword.get(keyword, [])
                })
            logger.info(f"✅ Found ALL keywords in: {video_path}")
    
    return results


def search_keywords_in_transcriptions(transcriptions: List[Dict], keywords: List[str]) -> Dict[str, List[Dict]]:
    """
    Search for keywords in transcriptions and return videos that contain at least n-1 out of n keywords
    
    Args:
        transcriptions (List[Dict]): Loaded transcriptions data as list
        keywords (List[str]): List of keywords to search for
        
    Returns:
        Dict[str, List[Dict]]: Results grouped by keyword
    """
    results = {keyword: [] for keyword in keywords}
    
    # Track which videos contain which keywords
    video_keyword_counts = {}
    
    for entry in transcriptions:
        # Extract video info
        video_path = entry.get('video_path', '')
        subject_id = entry.get('subject_id', 'unknown')
        age = entry.get('age', 'unknown')
        gender = entry.get('gender', 'unknown')
        skin_type = entry.get('skin-type', 'unknown')
        is_dark = entry.get('is_dark', False)
        
        # Get transcription text
        transcription_text = entry.get('transcription_timestamp', '')
        
        # Check each keyword
        video_has_keywords = []
        for keyword in keywords:
            if keyword.lower() in transcription_text.lower():
                # Parse timestamps for this keyword
                keyword_occurrences = parse_timestamp_for_keywords(transcription_text, [keyword])
                
                # Extract just the timestamps for this keyword
                timestamps = [timestamp for kw, timestamp in keyword_occurrences if kw == keyword]
                
                results[keyword].append({
                    'video_path': video_path,
                    'subject_id': subject_id,
                    'age': age,
                    'gender': gender,
                    'skin-type': skin_type,
                    'is_dark': is_dark,
                    'keyword_occurrences': timestamps
                })
                video_has_keywords.append(keyword)
        
        # Track which keywords this video has
        if video_has_keywords:
            video_keyword_counts[video_path] = video_has_keywords
    
    # Filter to only include videos that have at least n-1 out of n keywords
    min_required = len(keywords) - 4  # n-4 out of n
    filtered_results = {keyword: [] for keyword in keywords}
    
    for video_path, found_keywords in video_keyword_counts.items():
        if len(found_keywords) >= min_required:
            # This video meets the criteria, include all its keyword entries
            for keyword in keywords:
                for entry in results[keyword]:
                    if entry['video_path'] == video_path:
                        filtered_results[keyword].append(entry)
    
    return filtered_results


def print_results(results: Dict[str, List[Dict]], output_json_path: str = None):
    """
    Print search results grouped by video and optionally save to JSON
    
    Args:
        results (Dict[str, List[Dict]]): Search results with video_path and keyword occurrences
        output_json_path (str): Optional path to save results as JSON
    """
    print("\n" + "="*80)
    print("🔍 TRANSCRIPTION KEYWORD SEARCH RESULTS (AT LEAST N-2 OUT OF N KEYWORDS)")
    print("="*80)
    
    # Group all entries by video_path
    videos = {}
    for keyword, entries in results.items():
        for entry in entries:
            video_path = entry['video_path']
            if video_path not in videos:
                videos[video_path] = {
                    'subject_id': entry['subject_id'],
                    'age': entry['age'],
                    'gender': entry['gender'],
                    'skin_type': entry['skin-type'],
                    'is_dark': entry['is_dark'],
                    'transcription_timestamp': entry.get('transcription_timestamp', ''),
                    'keywords': {}
                }
            videos[video_path]['keywords'][keyword] = entry['keyword_occurrences']
    
    print(f"\n📊 Found {len(videos)} videos containing at least {len(results.keys())-2} out of {len(results.keys())} keywords:")
    
    # Prepare JSON output data
    json_output = {
        'search_criteria': f'at least {len(results.keys())-4} out of {len(results.keys())} keywords',
        'total_videos': len(videos),
        'total_keyword_matches': sum(len(entries) for entries in results.values()),
        'videos': []
    }
    
    for i, (video_path, video_data) in enumerate(videos.items(), 1):
        print(f"\n   {i:3d}. {video_path}")
        print(f"       Subject: {video_data['subject_id']} | Age: {video_data['age']} | Gender: {video_data['gender']} | Skin: {video_data['skin_type']} | Dark: {video_data['is_dark']}")
        print(f"       Keywords and expression timestamps:")
        
        for keyword, timestamps in video_data['keywords'].items():
            if timestamps:
                print(f"         • {keyword}: {timestamps}")
            else:
                print(f"         • {keyword}: []")
        
        # Print transcription with timestamps
        print(f"       Transcription:")
        print(f"         {video_data['transcription_timestamp']}")
        
        # Add to JSON output
        json_output['videos'].append({
            'video_path': video_path,
            'subject_id': video_data['subject_id'],
            'age': video_data['age'],
            'gender': video_data['gender'],
            'skin_type': video_data['skin_type'],
            'is_dark': video_data['is_dark'],
            'transcription_timestamp': video_data['transcription_timestamp'],
            'keywords': video_data['keywords']
        })
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total unique videos with at least {len(results.keys())-4} out of {len(results.keys())} keywords: {len(videos)}")
    print(f"   Total keyword matches: {sum(len(entries) for entries in results.values())}")
    print("="*80)
    
    # Save to JSON if output path specified
    if output_json_path:
        try:
            import json
            with open(output_json_path, 'w') as f:
                json.dump(json_output, f, indent=2)
            print(f"\n💾 Results saved to: {output_json_path}")
        except Exception as e:
            print(f"\n❌ Failed to save JSON: {e}")


def main():
    """Main function to search for keywords in transcriptions"""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Search for keywords in transcription data')
    parser.add_argument('--json_path', type=str, 
                       default='/Users/ozgewhiting/Documents/EQLabs/datasets/CasualConversations/CC_annotations/CasualConversations_transcriptions.json',
                       help='Path to the transcriptions JSON file')
    parser.add_argument('--keywords', nargs='+', 
                       default=['smile', 'natural', 'neutral', 'sad', 'surprised', 'expressions', 'happy', 'angry'],
                       help='Keywords to search for (default: smile natural neutral sad)')
    parser.add_argument('--output_json', type=str, 
                       default='out.json',
                       help='Path to save results as JSON (default: out.json)')
    
    args = parser.parse_args()
    
    # Keywords to search for
    keywords = args.keywords
    
    print("🔍 TRANSCRIPTION KEYWORD SEARCH")
    print("="*50)
    print(f"📝 Searching for keywords: {', '.join(keywords)}")
    print(f"📊 Criteria: Videos must contain at least {len(keywords)-4} out of {len(keywords)} keywords")
    print(f"📁 JSON file: {args.json_path}")
    print(f"💾 Output JSON: {args.output_json}")
    print("="*50)
    
    # Load transcriptions
    transcriptions = load_transcriptions(args.json_path)
    if not transcriptions:
        print("❌ Failed to load transcriptions")
        return
    
    print(f"✅ Loaded {len(transcriptions)} video transcriptions")
    
    # Search for keywords
    results = search_keywords_in_transcriptions(transcriptions, keywords)
    
    # Print results and save to JSON
    print_results(results, args.output_json)


if __name__ == "__main__":
    main() 