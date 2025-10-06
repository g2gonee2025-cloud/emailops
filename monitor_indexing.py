#!/usr/bin/env python3
"""Real-time monitoring script for indexing progress"""
import os
from datetime import datetime, timedelta
from pathlib import Path

def monitor_worker2():
    log_file = Path(r"C:\Users\ASUS\Desktop\Outlook\_index\embedder_2_20251005_104603.log")
    
    if not log_file.exists():
        print("❌ Log file not found!")
        return
    
    # Read log file
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Count successful API calls
    success_calls = [l for l in lines if '200 OK' in l]
    error_lines = [l for l in lines if 'ERROR' in l]
    
    print("=" * 70)
    print(f"WORKER 2 INDEXING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Parse timing
    if success_calls:
        try:
            first_time = datetime.strptime(success_calls[0].split(' - ')[0], '%Y-%m-%d %H:%M:%S,%f')
            last_time = datetime.strptime(success_calls[-1].split(' - ')[0], '%Y-%m-%d %H:%M:%S,%f')
            
            elapsed_seconds = (last_time - first_time).total_seconds()
            elapsed_hours = elapsed_seconds / 3600
            
            rate_per_hour = len(success_calls) / elapsed_hours if elapsed_hours > 0 else 0
            
            print(f"\n📊 ACTIVITY STATUS:")
            print(f"   Started: {first_time.strftime('%H:%M:%S')}")
            print(f"   Latest:  {last_time.strftime('%H:%M:%S')}")
            print(f"   Running: {elapsed_hours:.1f} hours")
            
            # Check if still active (last call within 2 minutes)
            time_since_last = (datetime.now() - last_time).total_seconds()
            if time_since_last < 120:
                print(f"   Status:  🟢 ACTIVE (last call {int(time_since_last)}s ago)")
            else:
                print(f"   Status:  🔴 STOPPED (last call {int(time_since_last/60)} min ago)")
            
            print(f"\n📈 PROGRESS:")
            print(f"   API Calls Made: {len(success_calls):,}")
            print(f"   Processing Rate: {rate_per_hour:.1f} calls/hour")
            print(f"   Errors: {len(error_lines)}")
            
            # Estimate completion
            # Based on 12,924 total chunks and ~4,554 already done = 8,370 remaining
            # But we don't know exact chunk count per API call
            print(f"\n⏱️  ESTIMATES:")
            print(f"   If 1 call = 1 chunk:")
            remaining = 8370
            hours_left = remaining / rate_per_hour if rate_per_hour > 0 else 0
            completion = datetime.now() + timedelta(hours=hours_left)
            print(f"      Remaining: ~{remaining:,} chunks")
            print(f"      Time left: ~{hours_left:.1f} hours ({hours_left/24:.1f} days)")
            print(f"      Complete by: {completion.strftime('%Y-%m-%d %H:%M')}")
            
        except Exception as e:
            print(f"Error parsing log: {e}")
    else:
        print("❌ No successful API calls found in log")
    
    # Show recent activity
    print(f"\n📝 LAST 3 LOG ENTRIES:")
    for line in lines[-3:]:
        print(f"   {line.strip()[:100]}")
    
    print("=" * 70)

if __name__ == "__main__":
    monitor_worker2()