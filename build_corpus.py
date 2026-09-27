"""
build_corpus.py

Reconstructs a corpus from the dataset variants (Original, Detoxed, Detoxed+Antispam) and saves it as a single JSON lines file.

FOLDER LAYOUT EXPECTED
-----------------------
    <root>/
        Original data/
            [7812378231].txt
            [7812378231] [part 2].txt   (multi-part files are fine)
            stats.json
        Detoxed data/
            ...
            stats.json
        Detoxed and antispam data/
            ...
            stats.json


OUTPUT
------
One JSONL file with each line like:
    {"channel_id": "775774467321233509", "messages": ["User: text", ...]}


USAGE
-----
    python3 build_corpus.py \\
        --original "Datafolder/Original data" \\
        --detoxed "Datafolder/Detoxed data" \\
        --detox-antispam "Datafolder/Detoxed and antispam data" \\
        --output corpus_spam_removed.jsonl

Add --limit-channels N to do a quick test run on just the first N channels.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

CHANNEL_ID_RE = re.compile(r"\[(\d+)\]")


def channel_id_from_filename(filename: str):
    match = CHANNEL_ID_RE.search(filename)
    return match.group(1) if match else None


def map_channels_to_files(folder: Path):

    mapping = defaultdict(list)
    for path in folder.glob("*.txt"):
        channel_id = channel_id_from_filename(path.name)
        if channel_id is not None:
            mapping[channel_id].append(path)
    for channel_id in mapping:
        mapping[channel_id].sort()  # keep part files in a stable, predictable order
    return mapping

#Get one conversation (list of message strings) per non-empty line."""
def read_conversations(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line.split("\t")

#Concat conversations across all part-files for one channel
def read_channel_conversations(paths):
    conversations = []
    for path in paths:
        conversations.extend(read_conversations(path))
    return conversations

#Build text corpus: Original data minus antispam
def build_spam_removed_corpus(original_convs, detoxed_convs, detox_antispam_convs):
    # How many times each exact message text appears in Detoxed vs. Detoxed+Antispam. 
    # The positive difference = occurrences removed by the antispam step specifically.
    detoxed_counts = Counter(msg for conv in detoxed_convs for msg in conv)
    detox_antispam_counts = Counter(msg for conv in detox_antispam_convs for msg in conv)

    spam_removed = detoxed_counts - detox_antispam_counts  # Counter subtraction clamps at 0

    result = []
    for conv in original_convs:
        new_conv = []
        for msg in conv:
            if spam_removed.get(msg, 0) > 0:
                # This occurrence matches a message that was cut by antispam
                # (not detox) -- drop it, and use up one unit of the budget
                # so a later duplicate occurrence of the same text is still
                # correctly kept if fewer copies were actually removed.
                spam_removed[msg] -= 1
            else:
                new_conv.append(msg)
        if new_conv:  # a conversation that lost every message is dropped
            result.append(new_conv)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original", required=True, help="Path to the 'Original data' folder")
    parser.add_argument("--detoxed", required=True, help="Path to the 'Detoxed data' folder")
    parser.add_argument("--detox-antispam", required=True, help="Path to the 'Detoxed and antispam data' folder")
    parser.add_argument("--output", required=True, help="Output .jsonl path")
    parser.add_argument("--limit-channels", type=int, default=None, help="Only process the first N channels (for a quick test run)")
    args = parser.parse_args()

    original_folder = Path(args.original)
    detoxed_folder = Path(args.detoxed)
    detox_antispam_folder = Path(args.detox_antispam)
    output_path = Path(args.output)

    print("Indexing files by channel ID...")
    original_map = map_channels_to_files(original_folder)
    detoxed_map = map_channels_to_files(detoxed_folder)
    detox_antispam_map = map_channels_to_files(detox_antispam_folder)

    # Only process channels present in all three variants -- warn about any
    # mismatch rather than silently skipping or crashing.
    channel_ids = sorted(set(original_map) & set(detoxed_map) & set(detox_antispam_map))
    missing = (set(original_map) | set(detoxed_map) | set(detox_antispam_map)) - set(channel_ids)
    if missing:
        print(f"Warning: {len(missing)} channel ID(s) not present in all three folders, skipping: "f"{sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}")

    if args.limit_channels:
        channel_ids = channel_ids[: args.limit_channels]

    print(f"Processing {len(channel_ids)} channels...")

    total_convs_out = 0
    total_msgs_out = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for i, cid in enumerate(channel_ids, start=1):
            original_convs = read_channel_conversations(original_map[cid])
            detoxed_convs = read_channel_conversations(detoxed_map[cid])
            detox_antispam_convs = read_channel_conversations(detox_antispam_map[cid])

            spam_removed_convs = build_spam_removed_corpus(
                original_convs, detoxed_convs, detox_antispam_convs
            )

            for conv in spam_removed_convs:
                out_f.write(json.dumps({"channel_id": cid, "messages": conv}, ensure_ascii=False) + "\n")
                total_convs_out += 1
                total_msgs_out += len(conv)

            if i % 10 == 0 or i == len(channel_ids):
                print(f"  [{i}/{len(channel_ids)}] channels done "
                      f"({total_convs_out:,} conversations, {total_msgs_out:,} messages written so far)")

    print(f"\nDone. Wrote {total_convs_out:,} conversations / {total_msgs_out:,} messages to {output_path}")


if __name__ == "__main__":
    main()
