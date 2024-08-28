import os
import re
import sys
import bisect
import copy
import sqlite3
import json
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
                    WHERE length(text) > 0 and length(text) < 20 and text not like 'NCCL%'
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


def query_nvtx_events(cur, ignore_comm):
    # 59 is NvtxPushPopRange|NvtxPushPopRange
    nvtx_q = """SELECT start, end, globalTid, text FROM NVTX_EVENTS
    WHERE eventType = 59 and length(text) > 0 and length(text) < 20 and text != 'iter' and text not like 'NCCL%'
    """
    if ignore_comm:
        nvtx_q = """SELECT start, end, globalTid, text FROM NVTX_EVENTS
            WHERE eventType = 59 and length(text) < 10 and
                (text like "F%" or text like "B%" or text like "W%" or text = "Optimizer")
            """
    res = cur.execute(nvtx_q)
    records = res.fetchall()
    nvtx_evs = [{
        "start": r[0],
        "end": r[1],
        "tid": r[2],
        "text": remove_fbw_number(r[3]),
    } for r in records if is_fbwo(r[3])]
    print(set(e["text"] for e in nvtx_evs))
    return nvtx_evs


def create_nvtx_events_map(sqlite_file, ignore_comm):
    """tid => nvtx events (with kernel events inside)"""
    con = sqlite3.connect(sqlite_file)
    cur = con.cursor()

    string_map = query_string_map(cur)
    kernel_events = query_kernel_events(cur, string_map)
    nvtx_events = query_nvtx_events(cur, ignore_comm)

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

    for ev in kernel_events:
        nvtx_thread_events = nvtx_event_map[ev["tid"]]

        if ignore_comm:
            if ev["kernel_name"].startswith("nccl"):
                continue
            if ev["kernel_name"] == "Kernel" or ev["kernel_name"].startswith("kernel1") or ev["kernel_name"].startswith("kernel2"):
                continue
            # This implementation is a bit faster if we can assume nvtx events are in order and don't overlap.
            i = bisect.bisect_right(nvtx_event_map[ev["tid"]], ev["runtime_start"], key=lambda e: e["start"])
            # Some vectorized_elementwise_kernel do not belong to F, B, W
            if i == 0:
                if ev["kernel_name"] != "vectorized_elementwise_kernel":
                    print(f'no nvtx event found for kernel {ev["kernel_name"]}')
                continue
            j = bisect.bisect_left(nvtx_event_map[ev["tid"]], ev["runtime_end"], 0, i, key=lambda e: e["end"])
            # Some vectorized_elementwise_kernel do not belong to F, B, W
            if j >= len(nvtx_event_map[ev["tid"]]):
                if ev["kernel_name"] != "vectorized_elementwise_kernel":
                    print(f'no nvtx event found for kernel {ev["kernel_name"]}')
                continue

            for q in range(j, i):
                nvtx_ev = nvtx_thread_events[q]
                if nvtx_ev['start'] > ev['runtime_start'] or ev['runtime_end'] > nvtx_ev['end']:
                    continue
                # Sometimes F may contain incorrect kernel
                # Excluding "vectorized_elementwise_kernel" may result in incorrect time should be fine.
                if nvtx_ev["text"] == "F" and ev["kernel_name"] == "vectorized_elementwise_kernel":
                    continue
                nvtx_evs = nvtx_thread_events[q]["kernels"]
                k = bisect.bisect_right(nvtx_evs, kernel_range_func(ev), key=kernel_range_func)
                nvtx_evs.insert(k, ev)
            continue

        r = bisect.bisect_right(nvtx_event_sort_by_starts[ev["tid"]], ev["runtime_start"], key=lambda t: t[0])
        start_set = set(t[1] for t in nvtx_event_sort_by_starts[ev["tid"]][:r])

        l = bisect.bisect_left(nvtx_event_sort_by_ends[ev["tid"]], ev["runtime_end"], 0, r, key=lambda t: t[0])
        end_set = set(t[1] for t in nvtx_event_sort_by_ends[ev["tid"]][l:])

        nvtx_indices = start_set & end_set

        for q in nvtx_indices:
            nvtx_ev = nvtx_thread_events[q]
            assert nvtx_ev['start'] <= ev['runtime_start'] and ev['runtime_end'] <= nvtx_ev['end']
            nvtx_evs = nvtx_thread_events[q]["kernels"]
            k = bisect.bisect_right(nvtx_evs, kernel_range_func(ev), key=kernel_range_func)
            nvtx_evs.insert(k, ev)

    for nvtx_evs in nvtx_event_map.values():
        for nvtx_ev in nvtx_evs:
            if not nvtx_ev["kernels"]:
                name = nvtx_ev['text']
                if name not in ("send_forward", "send_backward", "recv_backward", "recv_forward", "forward_step_func", "get_batch", "model"):
                    print(f"kernel not found for nvtx event: {name} {nvtx_ev['start']} {nvtx_ev['end']}")
                nvtx_ev["device_id"] = 0
                nvtx_ev["kernel_start"] = 100000000000
                nvtx_ev["kernel_end"] = 100000000000
                continue
            nvtx_ev["kernels"].sort(key=lambda e: e["kernel_start"])
            nvtx_ev["kernel_start"] = nvtx_ev["kernels"][0]["kernel_start"]
            nvtx_ev["kernel_end"] = nvtx_ev["kernels"][-1]["kernel_end"]
            all_devs = set(e["device_id"] for e in nvtx_ev["kernels"])
            assert len(all_devs) == 1
            nvtx_ev["device_id"] = nvtx_ev["kernels"][0]["device_id"]

    return nvtx_event_map


def to_device_nvtx_event_map(nvtx_event_map, start, last_f_end, last_b_start):
    device_nvtx_event_map = {}
    for nvtx_evs in nvtx_event_map.values():
        for nvtx_ev in nvtx_evs:
            nvtx_ev["device_id"] += start
        dev = nvtx_evs[0]["device_id"]
        device_nvtx_event_map[dev] = nvtx_evs

    if last_b_start is not None:
        # Align the start time of each server
        first_dev = min(device_nvtx_event_map.keys())
        first_b_end = None
        for e in device_nvtx_event_map[first_dev]:
            if e["text"] == "B":
                first_b_end = e["kernel_end"]
                break
        assert first_b_end is not None
        d = first_b_end - last_b_start + 1
        for nvtx_evs in device_nvtx_event_map.values():
            for nvtx_ev in nvtx_evs:
                nvtx_ev["kernel_start"] -= d
                nvtx_ev["kernel_end"] -= d

    if last_b_start is None and last_f_end is not None:
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
    for nvtx_events in nvtx_event_map.values():
        for e in nvtx_events:
            if e["text"] == "F":
                dev_id = e["kernels"][0]["device_id"]
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
    for nvtx_events in nvtx_event_map.values():
        f_count = 0
        dev_id = None
        names = set(e["text"] for e in nvtx_events)
        for e in nvtx_events:
            if e["text"] == "F":
                f_count += 1
            elif e["text"] == "B":
                dev_id = e["kernels"][0]["device_id"]
                break
        assert dev_id is not None
        if dev_id == 0:
            return f_count


cname_map = {
    "iter": "good",
    "F": "bad",
    "B": "terrible",
    "W": "olive",
    "Optimizer": "vsync_highlight_color",
    "send_forward": "rail_animation",
    "recv_forward": "black",
    "send_backward": "rail_animation",
    "recv_backward": "black",
}


def create_event_json(nvtx_kernels_map):
    compute_names = {"F", "B", "W", "Optimizer"}
    # This will determine the color.
    events = defaultdict(list)
    for dev, nvtx_events in nvtx_kernels_map.items():
        for e in nvtx_events:
            name = e["text"]
            if name not in compute_names:
                continue
            if "kernel_start" not in e:
                continue
            start = e["kernel_start"]
            end = e["kernel_end"]
            events[dev].append({
                "type": name,
                "start_time": start,
                "completion_time": end,
            })
    min_dev = min(events.keys())
    max_dev = max(events.keys())
    return [events[d] for d in range(min_dev, max_dev + 1)]


def create_gte(nvtx_events, ignore_comm=False, remove_name=False, tid_prefix=0):
    events = []
    compute_names = {"F", "B", "W", "Optimizer"}
    # This will determine the color.
    spaces = {
        "F": " " * 1,
        "B": " " * 3,
        "W": " " * 16,
        "Optimizer": " " * 11,
    }
    for e in nvtx_events:
        name = e["text"]
        if ignore_comm and name not in compute_names:
            continue
        if "kernel_start" not in e:
            continue
        tid = e["tid"]
        start = e["kernel_start"] / 1000
        end = e["kernel_end"] / 1000
        # pid = tid // 0x1000000 % 0x1000000
        be = {
            "name": name if not remove_name else spaces[name],
            "cat": name,
            "ph": "B",
            "ts": start,
            "pid": e["device_id"] + tid_prefix,
            "tid": e["device_id"] + tid_prefix,
            "cname": cname_map.get(name) or "grey",
        }
        events.append(be)
        ee = copy.deepcopy(be)
        ee["ph"] = "E"
        ee["ts"] = end
        events.append(ee)
    return events


def nvtx_kernel_types(nvtx_kernels_map):
    kernels = defaultdict(set)
    for nvtx_evs in nvtx_kernels_map.values():
        for e in nvtx_evs:
            for ke in e["kernels"]:
                kernels[e["text"]].add(ke["kernel_name"])
    return kernels


def create_nvtx_kernels_map(sqlite_files, ignore_comm=False, err_shift=0):
    nvtx_kernels_map = {}
    server_events = []
    for i, sqlite_file in enumerate(sqlite_files):
        m = create_nvtx_events_map(sqlite_file, ignore_comm)
        server_events.append((server_sort_key(m), m))
    server_events.sort(key=lambda t: t[0])

    dev_start = 0
    last_f_end = None
    first_b_start = None
    for _, m in server_events:
        # last_f_end = None
        first_b_start = None
        dev_m = to_device_nvtx_event_map(m, dev_start, last_f_end, first_b_start)
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


def main(sqlite_dir, sqlite_dir_other, ignore_comm=False, remove_name=False):
    sqlite_files = [os.path.join(sqlite_dir, f) for f in os.listdir(sqlite_dir) if f.endswith(".sqlite")]
    if sqlite_dir_other:
        sqlite_files_other = [os.path.join(sqlite_dir_other, f) for f in os.listdir(sqlite_dir_other) if f.endswith(".sqlite")]
    else:
        sqlite_files_other = []
    print(f"sqlite files {sqlite_files}")
    print(f"sqlite other files {sqlite_files_other}")

    nvtx_kernels_map = create_nvtx_kernels_map(sqlite_files, ignore_comm, 0)

    events = []
    for nvtx_events in nvtx_kernels_map.values():
        evs = create_gte(nvtx_events, ignore_comm, remove_name)
        events.extend(evs)

    data = create_event_json(nvtx_kernels_map)
    with open(os.path.join(sqlite_dir, "zero-events.json"), "w") as f:
        json.dump(data, f)

    if sqlite_files_other:
        nvtx_kernels_map_other = create_nvtx_kernels_map(sqlite_files_other, ignore_comm)
        for nvtx_events in nvtx_kernels_map_other.values():
            evs = create_gte(nvtx_events, ignore_comm, remove_name, tid_prefix=100)
            events.extend(evs)

        data = create_event_json(nvtx_kernels_map_other)
        with open(os.path.join(sqlite_dir_other, "1f1b-events.json"), "w") as f:
            json.dump(data, f)

    gte_dict = {
      "traceEvents": events,
      "displayTimeUnit": "ns",
    }

    with open('nvtx.json', 'w') as f:
        json.dump(gte_dict, f)


if __name__ == "__main__":
    sqlite_dir = sys.argv[1]
    sqlite_dir_other = sys.argv[2] if len(sys.argv) == 3 else None

    main(sqlite_dir, sqlite_dir_other, True, True)