import argparse
import bisect
import dataclasses
import itertools
import json
import re

import numpy as np
import drawsvg as draw
import colorsys


@dataclasses.dataclass(eq=True, frozen=True)
class EventData:
    events: list
    duration: int


FBW_PATTERN = re.compile("^[F|B|W].*$")


def is_fbwo(text):
    return FBW_PATTERN.match(text) or text == "Optimizer"


def load_kth_iteration(filename, k, enable_comm=True, exclude_previous_iteration=True):
    with open(filename) as f:
        data = json.loads(f.read())

    if k == 0:
        start_time = 0
    else:
        if exclude_previous_iteration:
            start_time = min(find_kth_optimizer(dev_evs, k - 1)["end_time"] for dev_evs in data)
        else:
            start_time = min(find_kth_optimizer(dev_evs, k - 1)["start_time"] for dev_evs in data)
    end_time = max(find_kth_optimizer(dev_evs, k)["end_time"] for dev_evs in data)

    events = []
    for stage_evs in data:
        # Include previous optimizer
        l = bisect.bisect_left(stage_evs, start_time, key=lambda e: e["start_time"])
        r = bisect.bisect_right(stage_evs, end_time, key=lambda e: e["end_time"])
        evs = [{
            "type": e["type"],
            "fields": e["fields"],
            "start_time": int(max(e["start_time"] - start_time, 0)),
            "end_time": int(e["end_time"] - start_time),
        } for e in stage_evs[l:r] if enable_comm or is_fbwo(e["type"])]
        events.append(evs)

    duration = end_time - start_time
    if exclude_previous_iteration:
        iter_start_time = min(e["start_time"] for e in sum(events, []))
        for stage_evs in events:
            for e in stage_evs:
                e["start_time"] -= iter_start_time
                e["end_time"] -= iter_start_time

    nvtx_names = set()
    for e in sum(events, []):
        nvtx_names.update(get_nvtx_names(e["type"]))
    print(f"{filename}: nvtx names {nvtx_names}")
    return EventData(
        events=events,
        duration=duration,
    )


def find_kth_optimizer(stage_events, k):
    it = filter(lambda e: e["type"] == "Optimizer", stage_events)
    event = next(e for i, e in enumerate(it) if i == k)
    return event


def to_color_fmt(c):
    # c = to_greyscale(c)
    return f"#{hex(c[0])[2:]}{hex(c[1])[2:]}{hex(c[2])[2:]}"


GREYSCALE_WEIGHTS = np.array([0.299, 0.587, 0.114])


def to_greyscale(color):
    c = np.dot(GREYSCALE_WEIGHTS, color[:3].astype(float)).astype(int)
    return np.array([c, c, c, 255])


def change_color_sat(c, percentage):
    c = c.astype(float) / 255.0
    (h, s, v) = colorsys.rgb_to_hsv(c[0], c[1], c[2])
    s *= percentage
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    c = np.array([r, g, b]) * 255
    return c.astype(int)


SEND_RECV_COLOR = np.array([255, 211, 91])
SEND_COLOR = change_color_sat(SEND_RECV_COLOR, 0.6)
RECV_COLOR = change_color_sat(SEND_RECV_COLOR, 0.3)
COLOR_VALUE_MAP = {
    "F": np.array([57, 122, 242]),
    "B": np.array([68, 211, 218]),
    "W": np.array([224, 240, 231]),
    "Optimizer": np.array([200, 83, 8]),
}
BLACK = to_color_fmt(np.array([0, 0, 0, 255]))
WARNING_COLOR = np.array([227, 66, 52])
FBWO_PATTERN = re.compile(r'(F|B|W|Optimizer)')
COMM_PATTERN = re.compile(r'(SEND_FORWARD|RECV_FORWARD|SEND_BACKWARD|RECV_BACKWARD|SEND_POST_VALIDATION|RECV_POST_VALIDATION)')


def get_color_value(nvtx_event):
    nvtx_name = nvtx_event["type"]
    vague_time = bool(nvtx_event.get("vague_time"))
    color_value = get_color_value_by_name(nvtx_name)
    if vague_time:
        # The kernel time range is guessed by previous and next event.
        color_value = WARNING_COLOR
    return color_value


def get_color_value_by_name(nvtx_name):
    if nvtx_name in COLOR_VALUE_MAP:
        return COLOR_VALUE_MAP[nvtx_name]
    assert COMM_PATTERN.match(nvtx_name)
    names = COMM_PATTERN.findall(nvtx_name)
    assert len(names) > 0
    if len(names) == 1:
        name = names[0]
        if "SEND" in name:
            return SEND_COLOR
        assert "RECV" in name
        return RECV_COLOR
    return SEND_RECV_COLOR


def get_color(nvtx_event):
    v = get_color_value(nvtx_event)
    return to_color_fmt(v)


def get_color_by_name(nvtx_name):
    v = get_color_value_by_name(nvtx_name)
    return to_color_fmt(v)


def get_nvtx_names(event_type: str):
    if FBWO_PATTERN.match(event_type):
        return [event_type]
    assert COMM_PATTERN.match(event_type)
    return COMM_PATTERN.findall(event_type)


@dataclasses.dataclass(eq=True, frozen=True)
class PlotSetting:
    enable_border: bool
    enable_batch_id: bool
    enable_type: bool
    enable_all_fields: bool
    enable_edge_blur: bool
    unit_size: int
    time_per_unit: int
    graph_width: int

    @property
    def border_size(self):
        # return self.unit_size
        return 1

    @property
    def font_size(self):
        return self.unit_size * 10

    @property
    def span_height(self):
        return self.unit_size * 10

    @property
    def center_title_height(self):
        return self.span_height * 6

    @property
    def title_width(self):
        return self.unit_size * 60


class DrawCtx:
    def __init__(self, setting: PlotSetting, d, oy, ox):
        assert not isinstance(d, DrawCtx)
        self.setting = setting
        self.d = d
        self.oy = oy
        self.ox = ox

    @classmethod
    def from_base_ctx(cls, base_ctx, oy, ox):
        assert isinstance(base_ctx, DrawCtx)
        return cls(base_ctx.setting, base_ctx.d, base_ctx.oy + oy, base_ctx.ox + ox)

    def width(self):
        return self.d.width

    def height(self):
        return self.d.height

    def line(self, sy, sx, ey, ex, width=None):
        self.d.append(draw.Line(
            self.ox + sx,
            self.oy + sy,
            self.ox + ex,
            self.oy + ey,
            stroke='black',
            stroke_width=width or self.setting.border_size,
        ))

    def rect(self, sy, sx, h, w, color):
        self.d.append(draw.Rectangle(
            self.ox + sx,
            self.oy + sy,
            w, h,
            fill=color,
            shape_rendering="geometricPrecision",
        ))

    def rect_frame(self, sy, sx, h, w):
        self.d.append(draw.Rectangle(
            self.ox + sx,
            self.oy + sy,
            w, h,
            fill="none",
            stroke=BLACK,
            stroke_width=self.setting.border_size,
        ))

    def text(self, y, x, text, anchor="middle", font_scale=1):
        font_size = self.setting.font_size * font_scale
        tl = len(text) * font_size // 2
        self.d.append(draw.Text(
            text, font_size,
            self.ox + x,
            # Magic 3 to make it vertical center
            self.oy + y + font_size - 3,
            textLength=tl, lengthAdjust='spacing',
            text_anchor=anchor,
            font_family="Times New Roman",
            # font_style="oblique",
            # font_family="Computer Modern Roman",
        ))


def draw_events(setting: PlotSetting, file_event_data, output_filename, include_w=True, include_o=True, tail=50):
    canvas_info_list = [
        CanvasInfo(setting, d.events, tail, center_title_height=0, enable_info=True) for d in file_event_data
    ]
    span_height = setting.span_height
    height_sum = sum(info.get_canvas_size()[0] + span_height for info in canvas_info_list)
    width_max = max(info.get_canvas_size()[1] for info in canvas_info_list)

    d = draw.Drawing(width_max, height_sum, origin="top-left")
    ctx = DrawCtx(setting, d, 0, 0)
    sub_ctx = ctx

    i = 1
    for event_data, canvas_info in zip(file_event_data, canvas_info_list):
        plot_events(sub_ctx, event_data.events, "", canvas_info, include_w, include_o)
        shift_height = (canvas_info.get_canvas_size()[0] + span_height) * i
        sub_ctx = DrawCtx.from_base_ctx(ctx, shift_height, 0)
        i += 1

    d.save_svg(output_filename)


class CanvasInfo:
    def __init__(self, setting: PlotSetting, events, tail, center_title_height, enable_info=True):
        self.setting = setting
        time_per_unit = setting.time_per_unit

        last_time = max(max([e["end_time"] for e in dev_evs]) for dev_evs in events)
        self.max_len = (last_time + time_per_unit - 1) // time_per_unit + tail

        border_size = setting.border_size
        span_height = setting.span_height
        comp_comm_events = split_comm_events_if_exists(events)
        self.height = span_height * len(comp_comm_events) + border_size * (len(comp_comm_events) + 1)
        color_text_row_height = int(span_height * 1.6)
        self.color_text_height = color_text_row_height + border_size
        self.info_height = span_height + color_text_row_height + 3 * border_size
        if not enable_info:
            self.info_height /= 2
        self.center_title_height = center_title_height

    def get_canvas_size(self):
        # height, width
        return self.height + self.info_height + self.center_title_height, self.max_len + self.setting.title_width


def split_comm_events_if_exists(events):
    new_events = []
    comm_found = False
    for stage, evs in enumerate(events):
        comp_evs = []
        comm_evs = []
        for e in evs:
            if FBWO_PATTERN.match(e["type"]):
                comp_evs.append(e)
                continue
            assert COMM_PATTERN.match(e["type"])
            comm_evs.append(e)
            comm_found = True
        new_events.append(comp_evs)
        new_events.append(comm_evs)
    if not comm_found:
        return events
    return new_events


def plot_events(ctx, events, title_text: str, canvas_info: CanvasInfo, include_w=True, include_o=True, include_info=True):
    max_len = canvas_info.max_len
    height = canvas_info.height
    color_text_height = canvas_info.color_text_height

    setting = ctx.setting
    data_ctx = DrawCtx.from_base_ctx(ctx, 0, setting.title_width)

    border_size = setting.border_size
    span_height = setting.span_height
    time_per_unit = setting.time_per_unit
    enable_border = setting.enable_border

    comp_comm_events = split_comm_events_if_exists(events)
    enable_comm = len(comp_comm_events) > len(events)

    for i, evs in enumerate(comp_comm_events):
        h = i * span_height + (i + 1) * border_size
        for e in evs:
            start = border_size + e["start_time"] // time_per_unit
            end = border_size + e["end_time"] // time_per_unit
            if start == end or not setting.enable_edge_blur:
                plot_span(data_ctx, start, end, h, get_color(e))
            else:
                plot_span(data_ctx, start + 1, end - 1, h, get_color(e))
                c = change_color_sat(
                    get_color_value(e),
                    (e["start_time"] / time_per_unit) % 1.0)
                plot_span(data_ctx, start, start + 1, h, to_color_fmt(c))
                c = change_color_sat(
                    get_color_value(e),
                    (e["end_time"] / time_per_unit) % 1.0)
                plot_span(data_ctx, end - 1, end, h, to_color_fmt(c))

            if e["type"] == "Optimizer":
                continue

            if setting.enable_all_fields:
                def fmt_fields(fields):
                    mb = fields['microbatch'] if fields["microbatch"] is not None else "*"
                    return f"{mb}.{fields['chunk']}.{fields['sequence']}"
                fs = [fmt_fields(f) for f in e["fields"]]
                types = e["type"].split(",")
                assert len(fs) == len(types), f"{fs}, {types}"
                txt = ",".join([f"{typ}{f}" for typ, f in itertools.zip_longest(types, fs)])
                center = (start + end) // 2
                data_ctx.text(h, center, txt)
            elif setting.enable_batch_id:
                bs = [str(f["microbatch"]) for f in e["fields"] if f["microbatch"] is not None]
                microbatch = ",".join(bs)
                center = (start + end) // 2
                data_ctx.text(h, center, microbatch)
            elif setting.enable_type:
                center = (start + end) // 2
                data_ctx.text(h, center, e["type"])
        if enable_border:
            data_ctx.line(h+span_height, 0, h+span_height+border_size, max_len - 1)

    if enable_border:
        data_ctx.line(0, 0, 0, max_len - 1)
        data_ctx.line(0, 0, height, 0)
        data_ctx.line(0, max_len - 1, height, max_len - 1)

    dev_title_ctx = DrawCtx.from_base_ctx(ctx, 0, 0)
    ndev = len(comp_comm_events)
    if enable_comm:
        devs = sum([[i, i] for i in range(len(events))], [])
    else:
        devs = list(range(len(events)))
    add_devices(dev_title_ctx, devs)

    if not include_info:
        return

    info_height = ndev * span_height + (ndev + 1) * border_size
    info_ctx = DrawCtx.from_base_ctx(ctx, info_height, 0)
    add_info(info_ctx, color_text_height, include_w, include_o)

    if title_text:
        center_title_ctx = DrawCtx.from_base_ctx(info_ctx, canvas_info.info_height, 0)
        add_center_title(center_title_ctx, title_text)


def plot_span(ctx, start, end, h, color):
    setting = ctx.setting
    border_size = setting.border_size
    span_height = setting.span_height
    enable_border = setting.enable_border
    ctx.rect(h, start, span_height, end - start, color)
    if enable_border:
        ctx.rect_frame(h-border_size, start, span_height + border_size, end - start)


def add_devices(ctx, devs):
    setting = ctx.setting
    border_size = setting.border_size
    span_height = setting.span_height
    unit_size = setting.unit_size
    for i, dev in enumerate(devs):
        h = i * span_height + (i + 1) * border_size
        ctx.text(h, 6 * unit_size, "Device {}".format(dev), "left")


def add_info(ctx, color_text_height, include_w=True, include_o=True):
    div = 4 + int(include_w) + int(include_o)
    f_start = ctx.width() // div
    b_start = ctx.width() // div * 2
    w_start = ctx.width() // div * 3
    o_start = ctx.width() // div * 4

    setting = ctx.setting
    border_size = setting.border_size
    span_height = setting.span_height
    unit_size = setting.unit_size

    block_w = 25 * unit_size
    plot_span(ctx, f_start, f_start+block_w, color_text_height + border_size, get_color_by_name("F"))
    plot_span(ctx, b_start, b_start+block_w, color_text_height + border_size, get_color_by_name("B"))
    if include_w:
        plot_span(ctx, w_start, w_start+block_w, color_text_height + border_size, get_color_by_name("W"))
    if include_o:
        plot_span(ctx, o_start, o_start+block_w, color_text_height + border_size, get_color_by_name("Optimizer"))

    ctx.text(0, 6 * unit_size, "Time", "left")
    draw_arrow(ctx, span_height // 2 + border_size + 1, 65 * unit_size, 50 * unit_size)

    block_w = 30 * unit_size
    ctx.text(color_text_height, f_start + block_w, "F", "left")
    ctx.text(color_text_height, b_start + block_w,
             "B", "left")
    if include_w:
        ctx.text(color_text_height, w_start + block_w, "W", "left")
    if include_o:
        ctx.text(color_text_height, o_start + block_w, "Optimizer Step", "left")


def add_center_title(ctx: DrawCtx, text):
    center_title_height = ctx.setting.center_title_height
    ctx.text(center_title_height / 4, ctx.width() / 2,
             text, "middle", 2)


def draw_arrow(ctx: DrawCtx, start_y, start_x, width, thickness=2):
    unit_size = ctx.setting.unit_size
    b = thickness * (unit_size // 2)
    ctx.line(start_y, start_x, start_y, start_x + width, b)
    ctx.line(start_y, start_x + width, start_y - 3*b, start_x + width - 3*b)
    ctx.line(start_y, start_x + width, start_y + 3*b, start_x + width - 3*b)


def render_svg_graph(args):
    input_json_files = args.input_json.split(',')

    file_event_data = [load_kth_iteration(input_json, args.iteration, args.enable_comm) for input_json in input_json_files]
    first_event_data = file_event_data[0]
    time_per_unit = first_event_data.duration / args.graph_width
    setting = PlotSetting(
        enable_border=True,
        enable_batch_id=args.plot_microbatch,
        enable_type=args.plot_type,
        enable_all_fields=args.plot_all,
        enable_edge_blur=False,
        unit_size=2,
        time_per_unit=time_per_unit,
        graph_width=args.graph_width,
    )
    draw_events(setting, file_event_data, args.output_svg, include_w=True, include_o=False, tail=50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the pipeline nvtx json data to svg.')
    parser.add_argument('-i', '--input_json', type=str, required=True,
                        help='Path to the input nvtx json file. Use "1.json,2.json" for multiple inputs.')
    parser.add_argument('-o', '--output_svg', type=str, required=True, help='Path to the output svg file')
    parser.add_argument('-n', '--iteration', type=int, required=True, help='Which iteration to plot.')
    parser.add_argument('-w', '--graph_width', type=int, required=True, help='Width of the graph part.')
    parser.add_argument('-c', '--enable_comm', action='store_true', help='Plot communication.')
    parser.add_argument('-t', '--plot_type', action='store_true', help='Plot function type.')
    parser.add_argument('-m', '--plot_microbatch', action='store_true', help='Plot microbatch index.')
    parser.add_argument('-a', '--plot_all', action='store_true', help='Plot all fields.')
    args = parser.parse_args()
    render_svg_graph(args)
