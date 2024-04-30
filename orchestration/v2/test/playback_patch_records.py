import curses
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from v2.api.api import RagRecord
from v2.api.util import merge_records, record_to_dict
from v2.test.simple_test_client import read_patch_records


def convert_lists_to_dicts(dict_: dict) -> dict:
    new_dict = {}
    for key, value in dict_.items():
        if isinstance(value, dict):
            new_value = convert_lists_to_dicts(value)
        elif isinstance(value, list):
            new_value = {f"[{i}]": value[i] for i in range(len(value))}
        else:
            new_value = value
        new_dict[key] = new_value
    return new_dict


DeltaKey = Tuple[str, bool]
DeltaPrim = int | float | bool
DeltaValue = Union[
    "DeltaDict",
    Tuple[str, str],
    Tuple[DeltaPrim, bool],
]
DeltaDict = Dict[DeltaKey, DeltaValue]

# TODO(Guy): If same key is used at different paths,
# then resulting order may be buggy.
delta_dict_key_ordinals: Dict[str, int] = {}


def sorted_delta_dict_items(delta_dict: DeltaDict) -> List[Tuple[DeltaKey, DeltaValue]]:

    def sort_key(item: Tuple[DeltaKey, DeltaValue]) -> int:
        return delta_dict_key_ordinals[item[0][0]]

    return sorted(delta_dict.items(), key=sort_key)


def create_delta_dict_helper(old_dict: dict, new_dict: dict) -> DeltaDict:
    delta_dict: DeltaDict = {}
    for new_key, new_value in new_dict.items():
        if new_key not in delta_dict_key_ordinals:
            delta_dict_key_ordinals[new_key] = len(delta_dict_key_ordinals)
        if new_key in old_dict:
            delta_key = (new_key, False)
        else:
            delta_key = (new_key, True)
        old_value = old_dict.get(new_key)
        delta_value: DeltaValue
        if isinstance(new_value, dict):
            if old_value is None:
                old_value = {}
                assert delta_key[1]
            else:
                assert isinstance(old_value, dict)
                assert not delta_key[1]
            delta_value = create_delta_dict_helper(old_value, new_value)
        elif isinstance(new_value, str):
            if old_value is None:
                old_value = ""
                assert delta_key[1]
            else:
                assert isinstance(old_value, str)
                assert not delta_key[1]
            delta_value = (old_value, new_value[len(old_value) :])
        else:
            delta_value = (new_value, old_value != new_value)
        delta_dict[delta_key] = delta_value
    return delta_dict


def create_delta_dict(old_record: RagRecord, new_record: RagRecord) -> DeltaDict:
    old_dict = convert_lists_to_dicts(record_to_dict(old_record))
    new_dict = convert_lists_to_dicts(record_to_dict(new_record))
    return create_delta_dict_helper(old_dict, new_dict)


@dataclass
class Colors:
    def __init__(self, window: curses.window) -> None:
        curses.start_color()
        curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK)
        self.old_key = curses.color_pair(1)
        self.new_key = curses.color_pair(2)
        self.old_value = curses.color_pair(3)
        self.new_value = curses.color_pair(4)
        self.scroll_message = curses.color_pair(5)


DrawInst = Tuple[int, str, int]
DrawLine = List[DrawInst]


def create_draw_lines_helper(
    delta_dict: DeltaDict, colors: Colors, indent: int, draw_lines: List[DrawLine]
):
    delta_dict_items = sorted_delta_dict_items(delta_dict)
    for delta_key, delta_value in delta_dict_items:
        draw_line = []
        key_str, key_is_new = delta_key
        key_color = colors.new_key if key_is_new else colors.old_key
        draw_line.append((indent, f"{key_str}:", key_color))
        if isinstance(delta_value, dict):
            draw_lines.append(draw_line)
            create_draw_lines_helper(delta_value, colors, indent + 2, draw_lines)
        else:
            value_indent = indent + len(key_str) + 2
            if isinstance(delta_value[0], str):
                old_str, new_str = delta_value
                draw_line.append((value_indent, old_str, colors.old_value))
                new_str_indent = value_indent + len(old_str)
                draw_line.append((new_str_indent, new_str, colors.new_value))
            else:
                value, value_is_new = delta_value
                value_color = colors.new_value if value_is_new else colors.old_value
                draw_line.append((value_indent, f"{value}", value_color))
            draw_lines.append(draw_line)


def create_draw_lines(
    old_record: RagRecord, new_record: RagRecord, colors: Colors
) -> List[DrawLine]:
    delta_dict = create_delta_dict(old_record, new_record)
    draw_lines: List[DrawLine] = []
    create_draw_lines_helper(delta_dict, colors, 0, draw_lines)
    return draw_lines


def playback_patch_records(window: curses.window, patch_records: List[RagRecord]):
    curses.curs_set(0)
    colors = Colors(window)
    row_count, column_count = window.getmaxyx()

    patch_records = patch_records + [RagRecord()]
    merged_records = patch_records.copy()
    for i in range(1, len(merged_records)):
        merged_records[i] = merge_records(merged_records[i - 1], merged_records[i])
    merged_records = [merged_records[0]] + merged_records  # Create empty initial delta.

    record_index = 0
    line_offset = 0

    while True:
        record_index = max(0, min(record_index, len(merged_records) - 2))
        old_record = merged_records[record_index]
        new_record = merged_records[record_index + 1]

        draw_lines = create_draw_lines(old_record, new_record, colors)
        line_count = len(draw_lines)
        line_offset = max(0, min(line_offset, line_count - row_count))

        if line_count > row_count:
            draw_lines = draw_lines[line_offset : line_offset + row_count]
            if line_offset > 0:
                message = "=====  Up arrow for lines above.  ====="
                draw_lines[0] = [(0, message, colors.scroll_message)]
            if line_count - line_offset > row_count:
                message = "=====  Down arrow for lines below.  ====="
                draw_lines[-1] = [(0, message, colors.scroll_message)]

        window.clear()
        for line_index, draw_line in enumerate(draw_lines):
            for indent, text, color in draw_line:
                if indent >= column_count:
                    continue
                column_count_hack = (
                    column_count if line_index < row_count - 1 else column_count - 1
                )  # Hack to compensate for some curses weirdness.
                if indent + len(text) > column_count_hack:
                    text = text[: column_count_hack - indent]
                window.addstr(line_index, indent, text, color)
        window.refresh()

        ch = window.getch()
        if ch == curses.KEY_LEFT:
            record_index -= 1
        elif ch == curses.KEY_RIGHT:
            record_index += 1
        elif ch == curses.KEY_UP:
            line_offset -= 1
        elif ch == curses.KEY_DOWN:
            line_offset += 1
        else:
            break


usage_message = """
To control playback:
* Right arrow applies the next patch record.
* Left arrow reverts the last applied patch record.
* Down arrow scrolls down if text overflows below.
* Up arrow scrolls up if the text overflows above.
* Any other key quits.
"""


def main():
    _, patch_records_file = sys.argv
    patch_records = read_patch_records(patch_records_file)
    print(usage_message)
    input("Press ENTER to begin.")
    curses.wrapper(playback_patch_records, patch_records)


if __name__ == "__main__":
    main()
