import json
import os

from PIL.PngImagePlugin import PngInfo

import args_manager
import modules.config

from PIL import Image
from modules.util import generate_temp_filename


log_cache = {}


def get_current_html_path():
    date_string, local_temp_filename, only_name = generate_temp_filename(folder=modules.config.path_outputs)
    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')
    return html_name


def get_current_html_batch_path():
    date_string, local_temp_filename, only_name = generate_temp_filename(folder=modules.config.path_outputs)
    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log_batch.html')
    return html_name


def log(img, dic, single_line_number=3, save_file_folder=None, save_file_name=None, save_file_format: str = 'PNG', save_metadata: bool = False):
    if img is None or args_manager.args.disable_image_log:
        return

    date_string, local_temp_filename, only_name = generate_temp_filename(folder=modules.config.path_outputs, extension=save_file_format.lower())
    os.makedirs(os.path.dirname(local_temp_filename), exist_ok=True)

    if save_metadata or save_file_format != 'PNG':
        pnginfo = PngInfo()
        pnginfo.add_text("Comment", json.dumps(dic, ensure_ascii=False))
    else:
        pnginfo = None

    Image.fromarray(img).save(local_temp_filename, pnginfo=pnginfo)
    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')

    css_styles = (
        "<style>"
        "body { background-color: #121212; color: #E0E0E0; } "
        "a { color: #BB86FC; } "
        ".metadata { border-collapse: collapse; width: 100%; } "
        ".metadata .key { width: 15%; } "
        ".metadata .value { width: 85%; font-weight: bold; } "
        ".metadata th, .metadata td { border: 1px solid #4d4d4d; padding: 4px; } "
        ".image-container img { height: auto; max-width: 512px; display: block; padding-right:10px; } "
        ".image-container div { text-align: center; padding: 4px; } "
        "hr { border-color: gray; } "
        "</style>"
    )

    begin_part = f"<html><head><title>Fooocus Log {date_string}</title>{css_styles}</head><body><p>Fooocus Log {date_string} (private)</p>\n<p>All images are clean, without any hidden data/meta, and safe to share with others.</p><!--fooocus-log-split-->\n\n"
    end_part = f'\n<!--fooocus-log-split--></body></html>'

    middle_part = log_cache.get(html_name, "")

    if middle_part == "":
        if os.path.exists(html_name):
            existing_split = open(html_name, 'r', encoding='utf-8').read().split('<!--fooocus-log-split-->')
            if len(existing_split) == 3:
                middle_part = existing_split[1]
            else:
                middle_part = existing_split[0]

    div_name = only_name.replace('.', '_')
    item = f"<div id=\"{div_name}\" class=\"image-container\"><hr><table><tr>\n"
    item += f"<td><a href=\"{only_name}\" target=\"_blank\"><img src='{only_name}' onerror=\"this.closest('.image-container').style.display='none';\" loading='lazy'></img></a><div>{only_name}</div></td>"
    item += "<td><table class='metadata'>"
    for key, value in dic:
        item += f"<tr><td class='key'>{key}</td><td class='value'>{value}</td></tr>\n"
    item += "</table>"
    item += "</td>"
    item += "</tr></table></div>\n\n"

    middle_part = item + middle_part

    with open(html_name, 'w', encoding='utf-8') as f:
        f.write(begin_part + middle_part + end_part)

    print(f'Image generated with private log at: {html_name}')

    log_cache[html_name] = middle_part

    return only_name


def log_batch(filenames, dic, single_line_number=3):
    if not filenames:
        return

    date_string, local_temp_filename, _ = generate_temp_filename(folder=modules.config.path_outputs)
    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log_batch.html')
    existing_log = log_cache.get(html_name, None)

    if existing_log is None:
        if os.path.exists(html_name):
            existing_log = open(html_name, encoding='utf-8').read()
        else:
            existing_log = f'<p>Fooocus Log {date_string} (private)</p>\n<p>All images do not contain any hidden data.</p>'

    div_name = filenames[0].replace('.', '_')
    item = f'<div id="{div_name}">\n'
    item += f"<p>{filenames[0]}</p>\n"
    for i, (k, v) in enumerate(dic):
        if i < single_line_number:
            item += f"<p>{k}: <b>{v}</b> </p>\n"
        else:
            if (i - single_line_number) % 2 == 0:
                item += f"<p>{k}: <b>{v}</b>, "
            else:
                item += f"{k}: <b>{v}</b></p>\n"

    imgs = "".join([
                       f"<img src=\"{name}\" width=auto height=100% loading=lazy style=\"height:auto;max-width:512px\" onerror=\"document.getElementById('{div_name}').style.display = 'none';\"></img>"
                       for name in filenames])
    item += f"<p></p>{imgs}<hr></div>\n"
    existing_log = item + existing_log

    with open(html_name, 'w', encoding='utf-8') as f:
        f.write(existing_log)

    print(f'Image generated with private log at: {html_name}')

    log_cache[html_name] = existing_log

    return
