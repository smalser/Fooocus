import dataclasses
import typing as tp
import uuid

import modules.flags as flags


@dataclasses.dataclass
class TaskArgs:
    prompt: str
    negative_prompt: str
    style_selections: list[str]
    performance_selection: str
    aspect_ratios_selection: str
    image_number: int
    image_seed: int
    sharpness: float
    guidance_scale: float
    custom_steps: int
    base_model_name: str
    refiner_model_name: str
    refiner_switch: float
    loras: list[tuple[str, float]]
    input_image_checkbox: bool
    current_tab: str
    uov_method: str
    uov_input_image: tp.Any
    outpaint_selections: str
    inpaint_input_image: tp.Any
    inpaint_additional_prompt: str
    inpaint_mask_image: tp.Any
    save_file_folder: str
    save_file_name: str
    save_file_format: str
    save_metadata: bool
    image_prompts: list[dict[str, tp.Any]]


class DynamicArgs:
    ...


class AsyncTask:
    def __init__(self, **kwargs):
        self.uuid = str(uuid.uuid4())
        self.args = TaskArgs(**kwargs)
        self.name = f"[{self.uuid}] {kwargs.get('prompt')}"
        #
        self.tasks = []
        self.yields = []
        self.results = []
        self.finished = False
        #
        self.store = DynamicArgs()

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<AsyncTask: "{self.name}">'

    def get_control_net_tasks(self):
        self.store.cn_tasks = {x: [] for x in flags.ip_list}
        for ip in self.args.image_prompts:
            if ip['img'] is not None:
                self.store.cn_tasks[ip['type']].append([ip['img'], ip['stop'], ip['weight']])

        return self.store.cn_tasks

