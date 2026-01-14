#!/usr/bin/env python3
"""
Verify difference images and generate quality report.

Usage:
    python scripts/verify_difference_images.py --input-dir output/difference_images
"""

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits


def verify_fits_file(filepath: Path) -> dict:
    """Verify a FITS file and return basic stats."""
    try:
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            
            if data is None:
                return {"status": "error", "message": "No data in FITS"}
            
            finite_mask = np.isfinite(data)
            finite_data = data[finite_mask]
            
            if len(finite_data) == 0:
                return {"status": "error", "message": "No finite values"}
            
            return {
                "status": "ok",
                "shape": data.shape,
                "finite_fraction": float(np.sum(finite_mask) / data.size),
                "min": float(np.min(finite_data)),
                "max": float(np.max(finite_data)),
                "mean": float(np.mean(finite_data)),
                "std": float(np.std(finite_data)),
                "size_mb": filepath.stat().st_size / 1024 / 1024,
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Verify difference images")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/difference_images"),
        help="Directory with difference images",
    )
    args = parser.parse_args()

    # Load summary
    summary_file = args.input_dir / "processing_summary.json"
    if not summary_file.exists():
        print(f"❌ Summary file not found: {summary_file}")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    print("=" * 70)
    print("DIFFERENCE IMAGE VERIFICATION")
    print("=" * 70)
    print(f"\nTotal SNe processed: {summary['n_processed']}")
    print(f"Input directory: {args.input_dir}")

    # Verify each result
    issues = []
    by_mission = {}
    
    for result in summary["results"]:
        sn_name = result["sn_name"]
        mission = result["mission_name"]
        
        if mission not in by_mission:
            by_mission[mission] = {"count": 0, "total_size_mb": 0, "issues": []}
        
        by_mission[mission]["count"] += 1
        
        # Check files exist
        diff_file = args.input_dir / result["difference_file"]
        sig_file = args.input_dir / result["significance_file"]
        mask_file = args.input_dir / result["mask_file"]
        
        for ftype, fpath in [("diff", diff_file), ("sig", sig_file), ("mask", mask_file)]:
            if not fpath.exists():
                issue = f"{sn_name}: Missing {ftype} file"
                issues.append(issue)
                by_mission[mission]["issues"].append(issue)
            else:
                by_mission[mission]["total_size_mb"] += fpath.stat().st_size / 1024 / 1024
        
        # Check quality metrics
        if result["overlap_fraction"] < 50:
            issue = f"{sn_name}: Low overlap ({result['overlap_fraction']:.1f}%)"
            issues.append(issue)
            by_mission[mission]["issues"].append(issue)
        
        if abs(result["sig_max"]) > 1e9:
            issue = f"{sn_name}: Extreme significance ({result['sig_max']:.1e}σ) - possible artifact"
            issues.append(issue)
            by_mission[mission]["issues"].append(issue)

    # Print summary by mission
    print("\n" + "=" * 70)
    print("BY MISSION")
    print("=" * 70)
    
    for mission in sorted(by_mission.keys()):
        info = by_mission[mission]
        print(f"\n{mission}:")
        print(f"  SNe processed: {info['count']}")
        print(f"  Total size: {info['total_size_mb']:.1f} MB")
        print(f"  Avg size per SN: {info['total_size_mb']/info['count']:.1f} MB")
        
        if info["issues"]:
            print(f"  ⚠️  Issues: {len(info['issues'])}")
            for issue in info["issues"][:3]:
                print(f"    - {issue}")
            if len(info["issues"]) > 3:
                print(f"    ... and {len(info['issues'])-3} more")
        else:
            print("  ✅ No issues detected")

    # Print overall issues
    print("\n" + "=" * 70)
    print("QUALITY SUMMARY")
    print("=" * 70)
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} potential issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues)-10} more")
    else:
        print("\n✅ All files verified successfully!")

    # Sample a few files for detailed verification
    print("\n" + "=" * 70)
    print("SAMPLE FILE VERIFICATION")
    print("=" * 70)
    
    sample_sne = summary["results"][:3]  # First 3
    for result in sample_sne:
        sn_name = result["sn_name"]
        diff_file = args.input_dir / result["difference_file"]
        
        print(f"\n{sn_name} ({result['mission_name']} {result['filter_name']}):")
        print(f"  Overlap: {result['overlap_fraction']:.1f}%")
        print(f"  Significance range: [{result['sig_min']:.1f}, {result['sig_max']:.1f}]σ")
        print(f"  Detections: {result['n_detections']}")
        
        if diff_file.exists():
            stats = verify_fits_file(diff_file)
            if stats["status"] == "ok":
                print(f"  File: {stats['size_mb']:.1f} MB, shape={stats['shape']}")
                print(f"  Data: {stats['finite_fraction']*100:.1f}% finite")
                print(f"  Range: [{stats['min']:.2e}, {stats['max']:.2e}]")
            else:
                print(f"  ❌ Error: {stats['message']}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    
    if not issues:
        print("\n✅ All difference images are ready for training!")
    else:
        print(f"\n⚠️  Review {len(issues)} issues before proceeding to training")


if __name__ == "__main__":
    main()
