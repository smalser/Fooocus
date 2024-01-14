import numpy as np
import re
import typing as tp
import copy
import dataclasses
import threading
import uuid

from modules.private_logger import log_batch
from modules.async_task import AsyncTask, TaskArgs


async_tasks: list[AsyncTask] = []
running_tasks: list[AsyncTask] = []
events: list[tuple[str, tp.Any]] = []
states = {
    'progress_bar': (0, '...'),
    'preview': None,
    'running_task': None,
    'tasks_list': [],
}


def _update_task_states():
    running_task = None
    try:
        running_task = running_tasks[0]
    except IndexError:
        pass
    finally:
        states['running_task'] = running_task.name if running_tasks else None

    states['tasks_list'] = [x.name for x in async_tasks]
    return states['running_task'], states['tasks_list']


def _clear_events():
    global events

    try:
        last_event = [x for x in events if x[0] == 'Finish'][-1]
    except IndexError:
        last_event = None

    events.clear()
    if last_event:
        events.append(last_event)


def add_tasks(*tasks):
    async_tasks.extend(tasks)
    return _update_task_states()


def remove_task(selected):
    rer = re.compile(r'\[([\w-]+)]')
    to_stop = {re.findall(rer, x)[0] for x in selected}
    for x in list(async_tasks):
        if x.uuid in to_stop:
            async_tasks.remove(x)

    return _update_task_states()


def clear_tasks():
    async_tasks.clear()
    return _update_task_states()


def worker():
    import traceback
    import math
    import numpy as np
    import torch
    import time
    import shared
    import random
    import copy
    import modules.default_pipeline as pipeline
    import modules.core as core
    import modules.flags as flags
    import modules.config
    import modules.patch
    import ldm_patched.modules.model_management
    import extras.preprocessors as preprocessors
    import modules.inpaint_worker as inpaint_worker
    import modules.constants as constants
    import modules.advanced_parameters as advanced_parameters
    import extras.ip_adapter as ip_adapter
    import extras.face_crop
    import fooocus_version

    from modules.sdxl_styles import apply_style, apply_wildcards, fooocus_expansion
    from modules.private_logger import log
    from extras.expansion import safe_str
    from modules.util import remove_empty_str, HWC3, resize_image, \
        get_image_shape_ceil, set_image_shape_ceil, get_shape_ceil, resample_image, erode_or_dilate
    from modules.upscaler import perform_upscale

    try:
        async_gradio_app = shared.gradio_root
        flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''
        if async_gradio_app.share:
            flag += f''' or {async_gradio_app.share_url}'''
        print(flag)
    except Exception as e:
        print(e)

    def progressbar(async_task, number, text):
        print(f'[Fooocus] {text}')
        states['progress_bar'] = (number, text)
        events.append(('Preview', (number, text, None)))

    def yield_preview(async_task, number, text, img):
        states['progress_bar'] = (number, text)
        states['preview'] = img
        events.append(('Preview', (number, text, img)))

    def yield_prompt(async_task: AsyncTask):
        last_task = async_task.tasks[-1]

        async_task.positive_prompt = last_task['positive'][-1] or ''
        async_task.negative_prompt = last_task['negative'][-1] or ''
        events.append(('Prompts', (async_task.positive_prompt, async_task.negative_prompt)))

    def yield_result(async_task, imgs, do_not_show_finished_images=False):
        if not isinstance(imgs, list):
            imgs = [imgs]

        async_task.results = async_task.results + imgs

        if do_not_show_finished_images:
            return

        events.append(('Results', async_task.results))

    def yield_finish(async_task):
        states['preview'] = None
        events.append(('Finish', async_task.results))

    def build_image_wall(async_task):
        if not advanced_parameters.generate_image_grid:
            return

        results = async_task.results

        if len(results) < 2:
            return

        for img in results:
            if not isinstance(img, np.ndarray):
                return
            if img.ndim != 3:
                return

        H, W, C = results[0].shape

        for img in results:
            Hn, Wn, Cn = img.shape
            if H != Hn:
                return
            if W != Wn:
                return
            if C != Cn:
                return

        cols = float(len(results)) ** 0.5
        cols = int(math.ceil(cols))
        rows = float(len(results)) / float(cols)
        rows = int(math.ceil(rows))

        wall = np.zeros(shape=(H * rows, W * cols, C), dtype=np.uint8)

        for y in range(rows):
            for x in range(cols):
                if y * cols + x < len(results):
                    img = results[y * cols + x]
                    wall[y * H:y * H + H, x * W:x * W + W, :] = img

        # must use deep copy otherwise gradio is super laggy. Do not use list.append() .
        async_task.results = async_task.results + [wall]
        return wall

    def _preprocess(async_task: AsyncTask):
        args, store = async_task.args, async_task.store

        async_task.tasks = []
        store.save_args = {x: getattr(async_task.args, x) for x in TaskArgs.__match_args__ if x.startswith('save_')}
        store.result_filenames = []
        store.cn_tasks = async_task.get_control_net_tasks()
        store.outpaint_selections = [o.lower() for o in args.outpaint_selections]
        store.base_model_additional_loras = []
        store.raw_style_selections = copy.deepcopy(args.style_selections)
        store.sampler_name = advanced_parameters.sampler_name
        store.scheduler_name = advanced_parameters.scheduler_name
        store.goals = []

        # Set styles
        if fooocus_expansion in args.style_selections:
            store.use_expansion = True
            args.style_selections.remove(fooocus_expansion)
        else:
            store.use_expansion = False

        store.use_style = len(args.style_selections) > 0

        store.initial_latent = None
        store.denoising_strength = 1.0
        store.tiled = False

        width, height = args.aspect_ratios_selection.replace('×', ' ').split(' ')[:2]
        store.width, store.height = int(width), int(height)

        store.skip_prompt_processing = False
        store.refiner_swap_method = advanced_parameters.refiner_swap_method

        # Inpaint
        inpaint_worker.current_task = None
        store.inpaint_parameterized = advanced_parameters.inpaint_engine != 'None'
        store.inpaint_image = None
        store.inpaint_mask = None
        store.inpaint_head_model_path = None

        store.use_synthetic_refiner = False

        # Set steps/switch
        steps_map = {
            'Speed': 30,
            'Quality': 60,
            'Extreme Speed': 8,
        }
        store.steps = steps_map.get(args.performance_selection, args.custom_steps)

    def _set_control_nets(async_task: AsyncTask):
        args, store = async_task.args, async_task.store

        store.controlnet_canny_path = None
        store.controlnet_cpds_path = None
        store.clip_vision_path, store.ip_negative_path, store.ip_adapter_path, store.ip_adapter_face_path = None, None, None, None

        store.cn_tasks = {x: [] for x in flags.ip_list}
        for ip in args.image_prompts:
            if ip['img'] is not None:
                store.cn_tasks[ip['type']].append([ip['img'], ip['stop'], ip['weight']])

        if args.input_image_checkbox:
            if (args.current_tab == 'uov' or (
                    args.current_tab == 'ip' and advanced_parameters.mixing_image_prompt_and_vary_upscale)) \
                    and args.uov_method != flags.disabled and args.uov_input_image is not None:
                args.uov_input_image = HWC3(args.uov_input_image)
                if 'vary' in args.uov_method.lower():
                    store.goals.append('vary')
                elif 'upscale' in args.uov_method.lower():
                    store.goals.append('upscale')
                    if 'fast' in args.uov_method.lower():
                        store.skip_prompt_processing = True
                    else:
                        store.steps = {
                            'Speed': 18,
                            'Quality': 36,
                            'Extreme Speed': 8,
                        }.get(args.performance_selection, 18)

                    progressbar(async_task, 1, 'Downloading upscale models ...')
                    modules.config.downloading_upscale_model()
            if (args.current_tab == 'inpaint' or (
                    args.current_tab == 'ip' and advanced_parameters.mixing_image_prompt_and_inpaint)) \
                    and isinstance(args.inpaint_input_image, dict):
                args.inpaint_image = args.inpaint_input_image['image']
                args.inpaint_mask = args.inpaint_input_image['mask'][:, :, 0]

                if advanced_parameters.inpaint_mask_upload_checkbox:
                    if isinstance(args.inpaint_mask_image_upload, np.ndarray):
                        if args.inpaint_mask_image_upload.ndim == 3:
                            H, W, C = args.inpaint_image.shape
                            args.inpaint_mask_image_upload = resample_image(args.inpaint_mask_image_upload, width=W, height=H)
                            args.inpaint_mask_image_upload = np.mean(args.inpaint_mask_image_upload, axis=2)
                            args.inpaint_mask_image_upload = (args.inpaint_mask_image_upload > 127).astype(np.uint8) * 255
                            args.inpaint_mask = np.maximum(args.inpaint_mask, args.inpaint_mask_image_upload)

                if int(advanced_parameters.inpaint_erode_or_dilate) != 0:
                    args.inpaint_mask = erode_or_dilate(inpaint_mask, advanced_parameters.inpaint_erode_or_dilate)

                if advanced_parameters.invert_mask_checkbox:
                    args.inpaint_mask = 255 - args.inpaint_mask

                args.inpaint_image = HWC3(inpaint_image)
                if isinstance(args.inpaint_image, np.ndarray) and isinstance(args.inpaint_mask, np.ndarray) \
                        and (np.any(inpaint_mask > 127) or len(args.outpaint_selections) > 0):
                    progressbar(async_task, 1, 'Downloading upscale models ...')
                    modules.config.downloading_upscale_model()
                    if inpaint_parameterized:
                        progressbar(async_task, 1, 'Downloading inpainter ...')
                        store.inpaint_head_model_path, store.inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                            advanced_parameters.inpaint_engine)
                        store.base_model_additional_loras += [(store.inpaint_patch_model_path, 1.0)]
                        print(f'[Inpaint] Current inpaint model is {store.inpaint_patch_model_path}')
                        if args.refiner_model_name == 'None':
                            store.use_synthetic_refiner = True
                            store.refiner_switch = 0.5
                    else:
                        store.inpaint_head_model_path, store.inpaint_patch_model_path = None, None
                        print(f'[Inpaint] Parameterized inpaint is disabled.')
                    if args.inpaint_additional_prompt != '':
                        if args.prompt == '':
                            args.prompt = args.inpaint_additional_prompt
                        else:
                            args.prompt = args.inpaint_additional_prompt + '\n' + args.prompt
                    store.goals.append('inpaint')
            if args.current_tab == 'ip' or \
                    advanced_parameters.mixing_image_prompt_and_inpaint or \
                    advanced_parameters.mixing_image_prompt_and_vary_upscale:
                store.goals.append('cn')
                progressbar(async_task, 1, 'Downloading control models ...')
                if len(store.cn_tasks[flags.cn_canny]) > 0:
                    store.controlnet_canny_path = modules.config.downloading_controlnet_canny()
                if len(store.cn_tasks[flags.cn_cpds]) > 0:
                    store.controlnet_cpds_path = modules.config.downloading_controlnet_cpds()
                if len(store.cn_tasks[flags.cn_ip]) > 0:
                    store.clip_vision_path, store.ip_negative_path, store.ip_adapter_path = modules.config.downloading_ip_adapters('ip')
                if len(store.cn_tasks[flags.cn_ip_face]) > 0:
                    store.clip_vision_path, store.ip_negative_path, store.ip_adapter_face_path = modules.config.downloading_ip_adapters(
                        'face')
                progressbar(async_task, 1, 'Loading control models ...')

        # Load or unload CNs
        pipeline.refresh_controlnets([store.controlnet_canny_path, store.controlnet_cpds_path])
        ip_adapter.load_ip_adapter(store.clip_vision_path, store.ip_negative_path, store.ip_adapter_path)
        ip_adapter.load_ip_adapter(store.clip_vision_path, store.ip_negative_path, store.ip_adapter_face_path)

    def _set_extreme_speed(async_task: AsyncTask):
        args, store = async_task.args, async_task.store

        print('Enter LCM mode.')
        if args.refiner_model_name != 'None':
            print(f'Refiner disabled in LCM mode.')
            args.refiner_model_name = 'None'

        progressbar(async_task, 1, 'Downloading LCM components ...')
        args.loras += [(modules.config.downloading_sdxl_lcm_lora(), 1.0)]
        args.refiner_switch = 1.0

        store.sampler_name = advanced_parameters.sampler_name = 'lcm'
        store.scheduler_name = advanced_parameters.scheduler_name = 'lcm'
        store.cfg_scale = args.guidance_scale = 1.0

        args.sharpness = 0.0
        advanced_parameters.adaptive_cfg = 1.0
        advanced_parameters.adm_scaler_positive = 1.0
        advanced_parameters.adm_scaler_negative = 1.0
        advanced_parameters.adm_scaler_end = 0.0

    def _prompt_processing(async_task: AsyncTask):
        args, store = async_task.args, async_task.store

        prompts = remove_empty_str([safe_str(p) for p in args.prompt.splitlines()], default='')
        negative_prompts = remove_empty_str([safe_str(p) for p in args.negative_prompt.splitlines()], default='')

        prompt = prompts[0]
        negative_prompt = negative_prompts[0]

        extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
        extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

        progressbar(async_task, 3, 'Loading models ...')
        pipeline.refresh_everything(
            refiner_model_name=args.refiner_model_name,
            base_model_name=args.base_model_name,
            loras=args.loras,
            base_model_additional_loras=store.base_model_additional_loras,
            use_synthetic_refiner=store.use_synthetic_refiner
        )

        progressbar(async_task, 3, 'Processing prompts ...')
        for i in range(args.image_number):
            task_seed = (args.image_seed + i) % (constants.MAX_SEED + 1)  # randint is inclusive, % is not
            task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future

            task_prompt = apply_wildcards(prompt, task_rng)
            task_negative_prompt = apply_wildcards(negative_prompt, task_rng)
            task_extra_positive_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_positive_prompts]
            task_extra_negative_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_negative_prompts]

            positive_basic_workloads = []
            negative_basic_workloads = []

            if store.use_style:
                for s in args.style_selections:
                    p, n = apply_style(s, positive=task_prompt)
                    positive_basic_workloads += p
                    negative_basic_workloads += n
            else:
                positive_basic_workloads.append(task_prompt)

            negative_basic_workloads.append(task_negative_prompt)  # Always use independent workload for negative.

            positive_basic_workloads = remove_empty_str(positive_basic_workloads + task_extra_positive_prompts, default=task_prompt)
            negative_basic_workloads = remove_empty_str(negative_basic_workloads + task_extra_negative_prompts, default=task_negative_prompt)

            async_task.tasks.append(dict(
                task_seed=task_seed,
                task_prompt=task_prompt,
                task_negative_prompt=task_negative_prompt,
                positive=positive_basic_workloads,
                negative=negative_basic_workloads,
                expansion='',
                c=None,
                uc=None,
                positive_top_k=len(positive_basic_workloads),
                negative_top_k=len(negative_basic_workloads),
                log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
                log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
            ))

        # disable expansion when empty since it is not meaningful and influences image prompt
        if store.use_expansion and prompt != '':
            for i, t in enumerate(async_task.tasks):
                progressbar(async_task, 5, f'Preparing Fooocus text #{i + 1} ...')
                expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
                print(f'[Prompt Expansion] {expansion}')
                t['expansion'] = expansion
                t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.

        yield_prompt(async_task)

        for i, t in enumerate(async_task.tasks):
            progressbar(async_task, 7, f'Encoding positive #{i + 1} ...')
            t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])

            if abs(float(store.cfg_scale) - 1.0) < 1e-4:
                t['uc'] = pipeline.clone_cond(t['c'])
            else:
                progressbar(async_task, 10, f'Encoding negative #{i + 1} ...')
                t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])

    def _process_goals_vary(async_task: AsyncTask):
        args, store = async_task.args, async_task.store

        if 'subtle' in store.uov_method:
            store.denoising_strength = 0.5
        if 'strong' in store.uov_method:
            store.denoising_strength = 0.85
        if advanced_parameters.overwrite_vary_strength > 0:
            store.denoising_strength = advanced_parameters.overwrite_vary_strength

        shape_ceil = get_image_shape_ceil(args.uov_input_image)
        if shape_ceil < 1024:
            print(f'[Vary] Image is resized because it is too small.')
            shape_ceil = 1024
        elif shape_ceil > 2048:
            print(f'[Vary] Image is resized because it is too big.')
            shape_ceil = 2048

        args.uov_input_image = set_image_shape_ceil(args.uov_input_image, shape_ceil)

        store.initial_pixels = core.numpy_to_pytorch(args.uov_input_image)
        progressbar(async_task, 13, 'VAE encoding ...')

        candidate_vae, _ = pipeline.get_candidate_vae(
            steps=store.steps,
            switch=store.switch,
            denoise=store.denoising_strength,
            refiner_swap_method=store.refiner_swap_method
        )

        store.initial_latent = core.encode_vae(vae=candidate_vae, pixels=store.initial_pixels)
        B, C, H, W = store.initial_latent['samples'].shape
        width = W * 8
        height = H * 8
        print(f'Final resolution is {str((height, width))}.')

    def _process_goals_upscale(async_task: AsyncTask):
        args, store = async_task.args, async_task.store

        H, W, C = args.uov_input_image.shape
        progressbar(async_task, 13, f'Upscaling image from {str((H, W))} ...')
        args.uov_input_image = perform_upscale(args.uov_input_image)
        print(f'Image upscaled.')

        if '1.5x' in args.uov_method:
            f = 1.5
        elif '2x' in args.uov_method:
            f = 2.0
        else:
            f = 1.0

        store.shape_ceil = get_shape_ceil(H * f, W * f)

        if store.shape_ceil < 1024:
            print(f'[Upscale] Image is resized because it is too small.')
            args.uov_input_image = set_image_shape_ceil(args.uov_input_image, 1024)
            store.shape_ceil = 1024
        else:
            args.uov_input_image = resample_image(args.uov_input_image, width=W * f, height=H * f)

        image_is_super_large = store.shape_ceil > 2800

        if 'fast' in store.uov_method:
            direct_return = True
        elif image_is_super_large:
            print('Image is too large. Directly returned the SR image. '
                  'Usually directly return SR image at 4K resolution '
                  'yields better results than SDXL diffusion.')
            direct_return = True
        else:
            direct_return = False

        if direct_return:
            d = [('Upscale (Fast)', '2x')]
            log(args.uov_input_image, d)
            yield_result(async_task, args.uov_input_image, do_not_show_finished_images=True)
            return

        store.tiled = True
        store.denoising_strength = 0.382

        if advanced_parameters.overwrite_upscale_strength > 0:
            store.denoising_strength = advanced_parameters.overwrite_upscale_strength

        initial_pixels = core.numpy_to_pytorch(args.uov_input_image)
        progressbar(async_task, 13, 'VAE encoding ...')

        candidate_vae, _ = pipeline.get_candidate_vae(
            steps=store.steps,
            switch=store.switch,
            denoise=store.denoising_strength,
            refiner_swap_method=store.refiner_swap_method
        )

        initial_latent = core.encode_vae(
            vae=candidate_vae,
            pixels=initial_pixels, tiled=store.tiled)
        B, C, H, W = initial_latent['samples'].shape
        width = W * 8
        height = H * 8
        print(f'Final resolution is {str((height, width))}.')

    def _process_goals_inpaint(async_task: AsyncTask):
        args, store = async_task.args, async_task.store

        inpaint_image = store.inpaint_image
        inpaint_mask = store.inpaint_mask

        if len(store.outpaint_selections) > 0:
            H, W, C = inpaint_image.shape
            if 'top' in store.outpaint_selections:
                inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant',
                                      constant_values=255)
            if 'bottom' in store.outpaint_selections:
                inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant',
                                      constant_values=255)

            H, W, C = inpaint_image.shape
            if 'left' in store.outpaint_selections:
                inpaint_image = np.pad(inpaint_image, [[0, 0], [int(H * 0.3), 0], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(H * 0.3), 0]], mode='constant',
                                      constant_values=255)
            if 'right' in store.outpaint_selections:
                inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(H * 0.3)], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(H * 0.3)]], mode='constant',
                                      constant_values=255)

            inpaint_image = np.ascontiguousarray(inpaint_image.copy())
            inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())
            advanced_parameters.inpaint_strength = 1.0
            advanced_parameters.inpaint_respective_field = 1.0

        store.denoising_strength = advanced_parameters.inpaint_strength

        inpaint_worker.current_task = inpaint_worker.InpaintWorker(
            image=inpaint_image,
            mask=inpaint_mask,
            use_fill=store.denoising_strength > 0.99,
            k=advanced_parameters.inpaint_respective_field
        )

        if advanced_parameters.debugging_inpaint_preprocessor:
            yield_result(async_task, inpaint_worker.current_task.visualize_mask_processing(),
                         do_not_show_finished_images=True)
            return

        progressbar(async_task, 13, 'VAE Inpaint encoding ...')

        inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
        inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
        inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)

        candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
            steps=store.steps,
            switch=store.switch,
            denoise=store.denoising_strength,
            refiner_swap_method=store.refiner_swap_method
        )

        latent_inpaint, latent_mask = core.encode_vae_inpaint(
            mask=inpaint_pixel_mask,
            vae=candidate_vae,
            pixels=inpaint_pixel_image)

        latent_swap = None
        if candidate_vae_swap is not None:
            progressbar(async_task, 13, 'VAE SD15 encoding ...')
            latent_swap = core.encode_vae(
                vae=candidate_vae_swap,
                pixels=inpaint_pixel_fill)['samples']

        progressbar(async_task, 13, 'VAE encoding ...')
        latent_fill = core.encode_vae(
            vae=candidate_vae,
            pixels=inpaint_pixel_fill)['samples']

        inpaint_worker.current_task.load_latent(
            latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)

        if store.inpaint_parameterized:
            pipeline.final_unet = inpaint_worker.current_task.patch(
                inpaint_head_model_path=store.inpaint_head_model_path,
                inpaint_latent=latent_inpaint,
                inpaint_latent_mask=latent_mask,
                model=pipeline.final_unet
            )

        if not advanced_parameters.inpaint_disable_initial_latent:
            store.initial_latent = {'samples': latent_fill}

        B, C, H, W = latent_fill.shape
        store.height, store.width = H * 8, W * 8
        final_height, final_width = inpaint_worker.current_task.image.shape[:2]
        print(f'Final resolution is {str((final_height, final_width))}, latent is {str((store.height, store.width))}.')

    def _process_goals_cn(async_task: AsyncTask):
        args, store = async_task.args, async_task.store

        for task in store.cn_tasks[flags.cn_canny]:
            cn_img, cn_stop, cn_weight = task
            cn_img = resize_image(HWC3(cn_img), width=store.width, height=store.height)

            if not advanced_parameters.skipping_cn_preprocessor:
                cn_img = preprocessors.canny_pyramid(cn_img)

            cn_img = HWC3(cn_img)
            task[0] = core.numpy_to_pytorch(cn_img)
            if advanced_parameters.debugging_cn_preprocessor:
                yield_result(async_task, cn_img, do_not_show_finished_images=True)
                return

        for task in store.cn_tasks[flags.cn_cpds]:
            cn_img, cn_stop, cn_weight = task
            cn_img = resize_image(HWC3(cn_img), width=store.width, height=store.height)

            if not advanced_parameters.skipping_cn_preprocessor:
                cn_img = preprocessors.cpds(cn_img)

            cn_img = HWC3(cn_img)
            task[0] = core.numpy_to_pytorch(cn_img)
            if advanced_parameters.debugging_cn_preprocessor:
                yield_result(async_task, cn_img, do_not_show_finished_images=True)
                return

        for task in store.cn_tasks[flags.cn_ip]:
            cn_img, cn_stop, cn_weight = task
            cn_img = HWC3(cn_img)

            # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
            cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

            task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=store.ip_adapter_path)
            if advanced_parameters.debugging_cn_preprocessor:
                yield_result(async_task, cn_img, do_not_show_finished_images=True)
                return

        for task in store.cn_tasks[flags.cn_ip_face]:
            cn_img, cn_stop, cn_weight = task
            cn_img = HWC3(cn_img)

            if not advanced_parameters.skipping_cn_preprocessor:
                cn_img = extras.face_crop.crop_image(cn_img)

            # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
            cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

            task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=store.ip_adapter_face_path)
            if advanced_parameters.debugging_cn_preprocessor:
                yield_result(async_task, cn_img, do_not_show_finished_images=True)
                return

        all_ip_tasks = store.cn_tasks[flags.cn_ip] + store.cn_tasks[flags.cn_ip_face]

        if len(all_ip_tasks) > 0:
            pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)

    @torch.no_grad()
    @torch.inference_mode()
    def handler(async_task: AsyncTask):
        execution_start_time = time.perf_counter()
        _preprocess(async_task)

        args, store = async_task.args, async_task.store

        assert args.performance_selection in ['Speed', 'Quality', 'Extreme Speed', 'Custom']
        if args.base_model_name == args.refiner_model_name:
            print(f'Refiner disabled because base model and refiner are same.')
            args.refiner_model_name = 'None'

        # Extreme speed
        if args.performance_selection == 'Extreme Speed':
            _set_extreme_speed(async_task)

        modules.patch.adaptive_cfg = advanced_parameters.adaptive_cfg
        print(f'[Parameters] Adaptive CFG = {modules.patch.adaptive_cfg}')

        modules.patch.sharpness = args.sharpness
        print(f'[Parameters] Sharpness = {modules.patch.sharpness}')

        modules.patch.positive_adm_scale = advanced_parameters.adm_scaler_positive
        modules.patch.negative_adm_scale = advanced_parameters.adm_scaler_negative
        modules.patch.adm_scaler_end = advanced_parameters.adm_scaler_end
        print(f'[Parameters] ADM Scale = '
              f'{modules.patch.positive_adm_scale} : '
              f'{modules.patch.negative_adm_scale} : '
              f'{modules.patch.adm_scaler_end}')

        store.cfg_scale = float(args.guidance_scale)
        print(f'[Parameters] CFG = {store.cfg_scale}')
        print(f'[Parameters] Seed = {args.image_seed}')

        # Control nets
        _set_control_nets(async_task)

        # Steps/Switch
        store.switch = int(round(store.steps * args.refiner_switch))

        # Overwrites
        if advanced_parameters.overwrite_step > 0:
            store.steps = advanced_parameters.overwrite_step
        if advanced_parameters.overwrite_switch > 0:
            store.switch = advanced_parameters.overwrite_switch
        if advanced_parameters.overwrite_width > 0:
            store.width = advanced_parameters.overwrite_width
        if advanced_parameters.overwrite_height > 0:
            store.height = advanced_parameters.overwrite_height

        print(f'[Parameters] Sampler = {store.sampler_name} - {store.scheduler_name}')
        print(f'[Parameters] Steps = {store.steps} - {store.switch}')

        progressbar(async_task, 1, 'Initializing ...')

        # Prompt processing
        if not store.skip_prompt_processing:
            _prompt_processing(async_task)

        if len(store.goals) > 0:
            progressbar(async_task, 13, 'Image processing ...')

        if 'vary' in store.goals:
            _process_goals_vary(async_task)

        if 'upscale' in store.goals:
            _process_goals_upscale(async_task)

        if 'inpaint' in store.goals:
            _process_goals_inpaint(async_task)

        if 'cn' in store.goals:
            _process_goals_cn(async_task)

        # FreeU
        if advanced_parameters.freeu_enabled:
            print(f'FreeU is enabled!')
            pipeline.final_unet = core.apply_freeu(
                pipeline.final_unet,
                advanced_parameters.freeu_b1,
                advanced_parameters.freeu_b2,
                advanced_parameters.freeu_s1,
                advanced_parameters.freeu_s2
            )

        all_steps = store.steps * args.image_number

        print(f'[Parameters] Denoising Strength = {store.denoising_strength}')

        if isinstance(store.initial_latent, dict) and 'samples' in store.initial_latent:
            store.log_shape = store.initial_latent['samples'].shape
        else:
            store.log_shape = f'Image Space {(store.height, store.width)}'

        print(f'[Parameters] Initial Latent shape: {store.log_shape}')

        preparation_time = time.perf_counter() - execution_start_time
        print(f'Preparation time: {preparation_time:.2f} seconds')

        final_sampler_name = store.sampler_name
        final_scheduler_name = store.scheduler_name

        if store.scheduler_name == 'lcm':
            store.final_scheduler_name = 'sgm_uniform'
            if pipeline.final_unet is not None:
                pipeline.final_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_refiner_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            print('Using lcm scheduler.')

        progressbar(async_task, 13, 'Moving model to GPU ...')

        def callback(step, x0, x, total_steps, y):
            done_steps = current_task_id * store.steps + step
            yield_preview(
                async_task,
                int(15.0 + 85.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {current_task_id + 1}-th Sampling',
                y,
            )

        for current_task_id, task in enumerate(async_task.tasks):
            execution_start_time = time.perf_counter()

            try:
                positive_cond, negative_cond = task['c'], task['uc']

                if 'cn' in store.goals:
                    for cn_flag, cn_path in [
                        (flags.cn_canny, store.controlnet_canny_path),
                        (flags.cn_cpds, store.controlnet_cpds_path)
                    ]:
                        for cn_img, cn_stop, cn_weight in store.cn_tasks[cn_flag]:
                            positive_cond, negative_cond = core.apply_controlnet(
                                positive_cond, negative_cond,
                                pipeline.loaded_ControlNets[cn_path], cn_img, cn_weight, 0, cn_stop)

                imgs = pipeline.process_diffusion(
                    positive_cond=positive_cond,
                    negative_cond=negative_cond,
                    steps=store.steps,
                    switch=store.switch,
                    width=store.width,
                    height=store.height,
                    image_seed=task['task_seed'],
                    callback=callback,
                    sampler_name=final_sampler_name,
                    scheduler_name=final_scheduler_name,
                    latent=store.initial_latent,
                    denoise=store.denoising_strength,
                    tiled=store.tiled,
                    cfg_scale=store.cfg_scale,
                    refiner_swap_method=store.refiner_swap_method
                )

                del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

                if inpaint_worker.current_task is not None:
                    imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

                for x in imgs:
                    d = [
                        ('Prompt', task['log_positive_prompt']),
                        ('Negative Prompt', task['log_negative_prompt']),
                        ('Fooocus V2 Expansion', task['expansion']),
                        ('Styles', str(store.raw_style_selections)),
                        ('Performance', args.performance_selection),
                        ('Resolution', str((store.width, store.height))),
                        ('Sharpness', args.sharpness),
                        ('Guidance Scale', args.guidance_scale),
                        ('ADM Guidance', str((
                            modules.patch.positive_adm_scale,
                            modules.patch.negative_adm_scale,
                            modules.patch.adm_scaler_end))),
                        ('Base Model', args.base_model_name),
                        ('Refiner Model', args.refiner_model_name),
                        ('Refiner Switch', args.refiner_switch),
                        ('Sampler', store.sampler_name),
                        ('Scheduler', store.scheduler_name),
                        ('Seed', task['task_seed']),
                    ]
                    for li, (n, w) in enumerate(args.loras):
                        if n != 'None':
                            d.append((f'LoRA {li + 1}', f'{n} : {w}'))
                    d.append(('Version', 'v' + fooocus_version.version))
                    filename = log(x, d, single_line_number=3, **store.save_args)
                    store.result_filenames.append(filename)

                yield_result(async_task, imgs, do_not_show_finished_images=len(async_task.tasks) == 1)
            except ldm_patched.modules.model_management.InterruptProcessingException as e:
                if shared.last_stop == 'skip':
                    print('User skipped')
                    continue
                else:
                    print('User stopped')
                    break

            execution_time = time.perf_counter() - execution_start_time
            print(f'Generating and saving time: {execution_time:.2f} seconds')

        if advanced_parameters.generate_image_grid:
            wall = build_image_wall(async_task)
            store.result_filenames = [log(wall, locals().get('d', {}), **store.save_args)] or store.result_filenames

        log_batch(store.result_filenames, locals().get('d', {}))

        return

    while True:
        time.sleep(0.01)
        if len(async_tasks) > 0:
            task = async_tasks.pop(0)
            running_tasks.append(task)
            _update_task_states()
            _clear_events()
            try:
                handler(task)
            except:
                traceback.print_exc()
            finally:
                yield_finish(task)
                running_tasks.remove(task)
                _update_task_states()
                pipeline.prepare_text_encoder(async_call=True)
    pass


threading.Thread(target=worker, daemon=True).start()
