"""
Example:
# Generate sqlite files from nsys-rep files
# Each sqlite file contains the data from 1 server.
nsys export --type sqlite --output <sqlite file> <nsys-rep file>

# Put all the sqlite files in the same directory `SQLITE_DIR`
SQLITE_DIR=1f1bv
# Select the k-th iteration to plot.
ITERATION_TO_PLOT=2
GRAPH_WIDTH=1000
python tools/load_nsys_events.py -c -d "${SQLITE_DIR}" -o nvtx.json
python tools/viz_nsys_events.py -i "${SQLITE_DIR}/nvtx.json" -o 1f1bv.svg -n ${ITERATION_TO_PLOT} -w ${GRAPH_WIDTH}
"""
import argparse
import itertools
import math
import os
import re
import bisect
import sqlite3
import json
import time
from collections import defaultdict


def query_string_map(cur):
    q = '''SELECT id, value FROM StringIds'''
    res = cur.execute(q)
    strings = res.fetchall()
    return {r[0]: r[1] for r in strings}


def query_kernel_events(cur, string_map):
    q = '''
        SELECT runtime.correlationId, kernel.start, kernel.end, runtime.start, runtime.end,
            kernel.deviceId, kernel.shortName, runtime.nameId, runtime.globalTid
        FROM CUPTI_ACTIVITY_KIND_KERNEL as kernel, CUPTI_ACTIVITY_KIND_RUNTIME as runtime
        WHERE runtime.correlationId = kernel.correlationId
            and kernel.globalPid / 0x1000000 % 0x1000000 = runtime.globalTid / 0x1000000 % 0x1000000
            and runtime.globalTid in (
                SELECT distinct globalTid FROM NVTX_EVENTS
                    WHERE length(text) > 0 and length(text) < 20 and
                        (text GLOB 'F[0-9]*' or text GLOB 'B[0-9]*' or text GLOB 'W[0-9]*' or text = 'Optimizer')
                    );
        '''
    res = cur.execute(q)
    records = res.fetchall()
    events = [{
        "cid": r[0],
        "kernel_start": r[1],
        "kernel_end": r[2],
        "runtime_start": r[3],
        "runtime_end": r[4],
        "device_id": r[5],
        "kernel_name": string_map[r[6]],
        "runtime_name": string_map[r[7]],
        "tid": r[8],
    } for r in records]
    return events


FBW_PATTERN = re.compile("^[F|B|W].*$")


def remove_fbw_number(text):
    if FBW_PATTERN.match(text):
        return text[:1]
    return text


def is_fbwo(text):
    return FBW_PATTERN.match(text) or text == "Optimizer"


def filter_nvtx(text, include_communication):
    if is_fbwo(text):
        return True
    return include_communication and COMM_PATTERN.match(text)


COMM_PATTERN = re.compile(r'(SEND_FORWARD|RECV_FORWARD|SEND_BACKWARD|RECV_BACKWARD|SEND_POST_VALIDATION|RECV_POST_VALIDATION)')


def remove_comm_number(text):
    """
    Since communication op can be fused, the name could be like:
        SEND_BACKWARD.1.18_RECV_BACKWARD.0.16_RECV_FORWARD.1.22_RECV_FORWARD.0.23
    """
    return COMM_PATTERN.findall(text)


def create_nvtx_event_set(nvtx_events):
    event_texts = set(e["text"] for e in nvtx_events)
    s = set()
    fbwo = {"F", "B", "W", "Optimizer"}
    for text in event_texts:
        if text in fbwo:
            s.add(text)
            continue
        assert COMM_PATTERN.match(text)
        s.update(text.split(","))
    return s


def cleanup_nvtx_event_name(text):
    if FBW_PATTERN.match(text):
        return remove_fbw_number(text)
    if COMM_PATTERN.match(text):
        return ",".join(remove_comm_number(text))
    return text


FBW_FIELD_PATTERN = re.compile(r"[FBW][a-z_]*(\d*)\.?(\d+)\.(\d+)")
COMM_FIELD_PATTERN = re.compile(r"[A-Z]+\.(\d+)\.(\d+)\.(\d+)")


def extract_event_fields_from_text(text):
    matcher = []
    if FBW_PATTERN.match(text):
        # F{mb}.{chunk}.{seq}
        # W_clear.{chunk}.{seq}
        matcher = FBW_FIELD_PATTERN
    if COMM_PATTERN.match(text):
        # RECV_BACKWARD.8.0.0_SEND_BACKWARD.7.0.0_RECV_FORWARD.28.0.0
        matcher = COMM_FIELD_PATTERN
    if not matcher:
        return []
    matches = matcher.findall(text)
    l = []
    for match in matches:
        mb = int(match[0]) if len(match[0]) else None
        chunk = int(match[1])
        seq = int(match[2])
        l.append({
            "microbatch": mb,
            "chunk": chunk,
            "sequence": seq,
        })
    return l


def get_nvtx_push_pop_range_id(cur):
    # 59 is NvtxPushPopRange
    q = "SELECT id from ENUM_NSYS_EVENT_TYPE WHERE name = 'NvtxPushPopRange'"
    res = cur.execute(q)
    records = res.fetchall()
    assert len(records) == 1
    return records[0][0]


def query_nvtx_events(cur, include_communication, limit=None):
    event_type_id = get_nvtx_push_pop_range_id(cur)
    nvtx_q = f"""SELECT start, end, globalTid, text FROM NVTX_EVENTS
        WHERE eventType = {event_type_id} and length(text) < 10 and
            (text GLOB 'F[0-9]*' or text GLOB 'B[0-9]*' or text GLOB 'W[0-9]*' or text GLOB 'W_clear.[0-9]*' or text = 'Optimizer')
        """
    if include_communication:
        nvtx_q = f"""SELECT start, end, globalTid, text FROM NVTX_EVENTS
        WHERE eventType = {event_type_id} and
            (text GLOB 'F[0-9]*' or text GLOB 'B[0-9]*' or text GLOB 'W[0-9]*' or text GLOB 'W_clear.[0-9]*' or text = 'Optimizer'
                or text GLOB 'SEND_*' or text GLOB 'RECV_*')
        """
    if limit:
        nvtx_q += f" LIMIT {limit}"
    res = cur.execute(nvtx_q)
    records = res.fetchall()
    nvtx_evs = [{
        "start": r[0],
        "end": r[1],
        "tid": r[2],
        "text": cleanup_nvtx_event_name(r[3]),
        "fields": extract_event_fields_from_text(r[3]),
    } for r in records if filter_nvtx(r[3], include_communication)]
    print("Found nvtx events: ", create_nvtx_event_set(nvtx_evs))
    return nvtx_evs


COMPUTE_NVTX_NAME = re.compile(r"^[F|B|W][0-9]+\.[0-9]+\.[0-9]+$")


def extract_nvtx_meta(nvtx_text):
    if not COMPUTE_NVTX_NAME.match(nvtx_text):
        return None
    meta = extract_event_fields_from_text(nvtx_text)[0]
    return meta["microbatch"], meta["chunk"], meta["sequence"]


def get_nvtx_meta_key(nvtx_event):
    meta = extract_nvtx_meta(nvtx_event["text"])
    if meta is None:
        return None
    return nvtx_event["tid"], *meta


def create_nvtx_events_map(sqlite_file, include_communication):
    """local device id => nvtx events (with kernel events inside)"""
    con = sqlite3.connect(sqlite_file)
    cur = con.cursor()

    timer_start = time.time()
    print(f"start to query from sqlite")
    string_map = query_string_map(cur)
    kernel_events = query_kernel_events(cur, string_map)
    nvtx_events = query_nvtx_events(cur, include_communication, limit=None)
    print(f"query from sqlite done using {time.time() - timer_start} seconds")

    # tid => nvtx events
    nvtx_event_map = defaultdict(list)

    def nvtx_range_func(e):
        return e["start"], e["end"]

    def kernel_range_func(e):
        return e["kernel_start"], e["kernel_end"]

    for ev in nvtx_events:
        ev["kernels"] = []
        r = bisect.bisect_right(nvtx_event_map[ev["tid"]], nvtx_range_func(ev), key=nvtx_range_func)
        nvtx_event_map[ev["tid"]].insert(r, ev)

    for evs in nvtx_event_map.values():
        rgs = [(e["start"], e["end"]) for e in evs]
        assert rgs == sorted(rgs)

    nvtx_event_sort_by_starts = defaultdict(list)
    nvtx_event_sort_by_ends = defaultdict(list)
    for tid, evs in nvtx_event_map.items():
        sts = nvtx_event_sort_by_starts[tid]
        eds = nvtx_event_sort_by_ends[tid]
        for i, ev in enumerate(evs):
            sts.append((ev["start"], i))
            eds.append((ev["end"], i))
        sts.sort(key=lambda t: t[0])
        eds.sort(key=lambda t: t[0])

    max_end_sort_by_starts = defaultdict(list)
    min_start_sort_by_ends = defaultdict(list)
    for tid, start_idx in nvtx_event_sort_by_starts.items():
        n = int(math.sqrt(len(start_idx)))
        ends = [nvtx_event_map[tid][idx]["end"] for _, idx in start_idx]
        for i in range(0, len(start_idx), n):
            max_end = max(ends[i:i+n])
            max_end_sort_by_starts[tid].append((max_end, i))
    for tid, end_idx in nvtx_event_sort_by_ends.items():
        n = int(math.sqrt(len(end_idx)))
        starts = [nvtx_event_map[tid][idx]["start"] for _, idx in end_idx]
        for i in range(0, len(end_idx), n):
            min_start = min(starts[i:i+n])
            min_start_sort_by_ends[tid].append((min_start, min(i+n, len(end_idx))))

    timer_start = time.time()
    print(f"start to match nvtx kernels")

    for ev in kernel_events:
        nvtx_thread_events = nvtx_event_map[ev["tid"]]

        if not include_communication:
            if ev["kernel_name"].startswith("nccl"):
                continue
            if ev["kernel_name"] == "Kernel" or ev["kernel_name"].startswith("kernel1") or ev["kernel_name"].startswith("kernel2"):
                continue

        # While nvtx_event["end"] sorted by nvtx_event["start"] are not strictly sorted and vice versa,
        # they should be roughly sorted. We employ this property to optimize the range when constructing
        # the start_set and end_set using max_end_sort_by_starts and min_start_sort_by_ends.
        r = bisect.bisect_right(nvtx_event_sort_by_starts[ev["tid"]], ev["runtime_start"], key=lambda t: t[0])
        start_idx = next((idx for max_end, idx in max_end_sort_by_starts[ev["tid"]] if max_end >= ev["runtime_start"]), None)
        # start_idx = 0
        if start_idx is None:
            start_set = set()
        else:
            start_set = set(t[1] for t in nvtx_event_sort_by_starts[ev["tid"]][start_idx:r])

        l = bisect.bisect_left(nvtx_event_sort_by_ends[ev["tid"]], ev["runtime_end"], 0, r, key=lambda t: t[0])
        end_idx = next((idx for min_start, idx in reversed(min_start_sort_by_ends[ev["tid"]]) if min_start <= ev["runtime_end"]),
                       None)
        # end_idx = len(nvtx_event_sort_by_ends[ev["tid"]])
        if len(start_set) == 0 or end_idx is None:
            end_set = set()
        else:
            end_set = set(t[1] for t in nvtx_event_sort_by_ends[ev["tid"]][l:end_idx])

        nvtx_indices = start_set & end_set

        for q in nvtx_indices:
            nvtx_ev = nvtx_thread_events[q]
            assert nvtx_ev['start'] <= ev['runtime_start'] and ev['runtime_end'] <= nvtx_ev['end']
            nvtx_evs = nvtx_ev["kernels"]
            k = bisect.bisect_right(nvtx_evs, kernel_range_func(ev), key=kernel_range_func)
            nvtx_evs.insert(k, ev)

    print(f"matching nvtx kernels done using {time.time() - timer_start} seconds")

    for tid, nvtx_evs in nvtx_event_map.items():
        for nvtx_ev in nvtx_evs:
            if not nvtx_ev["kernels"]:
                name = nvtx_ev['text']
                if name not in ("forward_step_func", "get_batch", "model"):
                    if name == 'W':
                        print(f"kernel not found for nvtx event W. This may happen when some layer does not contain any trainable parameter.")
                    else:
                        fields = nvtx_ev['fields']
                        print(f"kernel not found for nvtx event: {name} {tid} {fields} {nvtx_ev['start']} {nvtx_ev['end']}")
                # Assign a reasonable start, end time below.
                nvtx_ev["device_id"] = None
                nvtx_ev["kernel_start"] = None
                nvtx_ev["kernel_end"] = None
                continue
            nvtx_ev["kernels"].sort(key=lambda e: e["kernel_start"])
            nvtx_ev["kernel_start"] = nvtx_ev["kernels"][0]["kernel_start"]
            nvtx_ev["kernel_end"] = max(k["kernel_end"] for k in nvtx_ev["kernels"])
            all_devs = set(e["device_id"] for e in nvtx_ev["kernels"])
            assert len(all_devs) == 1
            nvtx_ev["device_id"] = all_devs.pop()

    # Used to guess the duration for those nvtx event without any kernel captured.
    # (stage, microbatch, chunk, seq) => time
    duration_map = defaultdict(list)
    for nvtx_evs in nvtx_event_map.values():
        for nvtx_ev in nvtx_evs:
            if nvtx_ev.get("device_id") is None:
                continue
            key = get_nvtx_meta_key(nvtx_ev)
            if key is None:
                continue
            duration_map[key].append(nvtx_ev["kernel_end"] - nvtx_ev["kernel_start"])
    aver_duration_map = {k: sum(v) / len(v) for k, v in duration_map.items()}

    device_nvtx_event_map = {}
    for nvtx_evs in nvtx_event_map.values():
        dev = next(filter(lambda d: d is not None,
                          (e.get("device_id") for e in nvtx_evs)), None)
        assert dev is not None
        prev_ev = None
        for nvtx_ev, next_ev in itertools.zip_longest(nvtx_evs, nvtx_evs[1:]):
            if nvtx_ev.get("device_id") is None:
                # For the nvtx event where no kernel is found.
                assert nvtx_ev.get("kernel_start") is None and nvtx_ev.get("kernel_end") is None
                nvtx_ev["device_id"] = dev
                # This time is inaccurate
                nvtx_ev["kernel_start"] = prev_ev["kernel_end"] if prev_ev else 0
                key = get_nvtx_meta_key(nvtx_ev)
                if key and aver_duration_map.get(key):
                    d = aver_duration_map[key]
                    nvtx_ev["kernel_end"] = nvtx_ev["kernel_start"] + d
                else:
                    nvtx_ev["kernel_end"] = next_ev["kernel_start"] if next_ev and next_ev["kernel_start"] \
                                                                else nvtx_ev["kernel_start"] + 1000
                nvtx_ev["vague_time"] = True
            else:
                assert nvtx_ev["device_id"] == dev
            prev_ev = nvtx_ev
        device_nvtx_event_map[dev] = nvtx_evs

    return device_nvtx_event_map


def shift_device_and_time(nvtx_event_map, dev_start, last_f_end, last_b_start):
    """Align the time for different servers."""
    device_nvtx_event_map = {}
    for dev, nvtx_evs in nvtx_event_map.items():
        for nvtx_ev in nvtx_evs:
            nvtx_ev["device_id"] += dev_start
        device_nvtx_event_map[dev + dev_start] = nvtx_evs

    if last_b_start is not None:
        # Align the start time of each server
        first_dev = min(device_nvtx_event_map.keys())
        first_b_end = None
        for e in device_nvtx_event_map[first_dev]:
            if e["text"] == "B":
                first_b_end = e["kernel_end"]
                break
        assert first_b_end is not None
        # Make first_b_end == last_b_start + 1
        d = first_b_end - last_b_start + 1
        for nvtx_evs in device_nvtx_event_map.values():
            for nvtx_ev in nvtx_evs:
                nvtx_ev["kernel_start"] -= d
                nvtx_ev["kernel_end"] -= d
    elif last_f_end is not None:
        # Align the start time of each server
        first_dev = min(device_nvtx_event_map.keys())
        first_f_start = device_nvtx_event_map[first_dev][0]["kernel_start"]
        # -1 to make it strictly after
        d = first_f_start - last_f_end - 1
        for nvtx_evs in device_nvtx_event_map.values():
            for nvtx_ev in nvtx_evs:
                nvtx_ev["kernel_start"] -= d
                nvtx_ev["kernel_end"] -= d
    return device_nvtx_event_map


def server_sort_key(nvtx_event_map):
    f_num = first_dev_f_num(nvtx_event_map)
    forward_times = first_forward_time(nvtx_event_map)
    first_dev = 0
    first_dev_f_start_time = forward_times[first_dev][0]
    # negate for more F to be earlier stages
    return -f_num, first_dev_f_start_time


def first_forward_time(nvtx_event_map):
    forward_times = {}
    for dev_id, nvtx_events in nvtx_event_map.items():
        for e in nvtx_events:
            if e["text"] == "F":
                forward_times[dev_id] = (e["kernel_start"], e["kernel_end"])
                break
    return forward_times


def first_backward_time(nvtx_event_map):
    forward_times = {}
    for nvtx_events in nvtx_event_map.values():
        for e in nvtx_events:
            if e["text"] == "B":
                dev_id = e["kernels"][0]["device_id"]
                forward_times[dev_id] = (e["kernel_start"], e["kernel_end"])
                break
    return forward_times


def first_dev_f_num(nvtx_event_map):
    f_count = 0
    for e in nvtx_event_map[0]:
        if e["text"] == "F":
            f_count += 1
        else:
            break
    return f_count


def create_event_json(nvtx_kernels_map):
    # This will determine the color.
    events = defaultdict(list)
    for dev, nvtx_events in nvtx_kernels_map.items():
        for e in nvtx_events:
            name = e["text"]
            start = e["kernel_start"]
            end = e["kernel_end"]
            evt = {
                "type": name,
                "fields": e["fields"],
                "start_time": start,
                "end_time": end,
            }
            if e.get("vague_time"):
                evt["vague_time"] = e["vague_time"]
            events[dev].append(evt)
    min_dev = min(events.keys())
    max_dev = max(events.keys())
    return [events[d] for d in range(min_dev, max_dev + 1)]


def create_nvtx_kernels_map(sqlite_files, include_communication=False, err_shift=0):
    nvtx_kernels_map = {}
    server_events = []
    for sqlite_file in sqlite_files:
        m = create_nvtx_events_map(sqlite_file, include_communication)
        server_events.append((server_sort_key(m), m))
    server_events.sort(key=lambda t: t[0])

    dev_start = 0
    last_f_end = None
    first_b_start = None
    for _, m in server_events:
        # last_f_end = None
        first_b_start = None  # Not working for V schedule. Disable first.
        dev_m = shift_device_and_time(m, dev_start, last_f_end, first_b_start)
        dev_start += len(dev_m)

        fw_time = first_forward_time(dev_m)
        # The time of first F in dev 0 could be wrong.
        # Shift the second dev back a bit.
        shf = err_shift if last_f_end is None else 0
        last_f_end = fw_time[max(fw_time.keys())][1] + shf

        bw_time = first_backward_time(dev_m)
        first_b_start = bw_time[max(bw_time.keys())][0]

        nvtx_kernels_map.update(dev_m)
    return nvtx_kernels_map


def main(sqlite_dir, output_json, include_communication=False):
    sqlite_files = [os.path.join(sqlite_dir, f) for f in os.listdir(sqlite_dir) if f.endswith(".sqlite")]
    print(f"sqlite files {sqlite_files}")

    nvtx_kernels_map = create_nvtx_kernels_map(sqlite_files, include_communication, 0)

    data = create_event_json(nvtx_kernels_map)
    with open(os.path.join(sqlite_dir, output_json), "w", encoding='utf-8') as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export nvtx data to json from sqlite.')
    parser.add_argument('-d', '--nvtx_sqlite_dir', type=str, help='Path to the nvtx sqlite directory')
    parser.add_argument('-o', '--output_json', type=str, help='Path to the output json file')
    parser.add_argument('-c', '--include_communication', action='store_true',
                        help='Whether include the communication nvtx events')
    args = parser.parse_args()

    main(args.nvtx_sqlite_dir, args.output_json, args.include_communication)
