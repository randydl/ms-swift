# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import transformers
from dataclasses import dataclass, field
from packaging import version
from typing import Any, Dict, List, Literal, Optional

from swift.utils import get_env_args
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall
from ..vision_utils import load_video_llava
from .llama import Llama3TemplateMeta
from .qwen import QwenTemplateMeta
from .utils import ChatmlTemplateMeta


class LlavaHfTemplate(Template):
    placeholder_tokens = ['<image>']

    @property
    def image_token_index(self):
        if not hasattr(self, '_image_token_index'):
            self._image_token_index = self.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        return self._image_token_index

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return ['<image>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        if images:
            image_processor = self.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model_info.torch_dtype)
            encoded['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                encoded['image_sizes'] = image_inputs['image_sizes']
            if version.parse(transformers.__version__) >= version.parse('4.47'):
                input_ids = encoded['input_ids']
                labels = encoded['labels']
                idx_list = findall(input_ids, self.image_token_index)  # <image>
                height, width = image_inputs['pixel_values'][0].shape[-2:]
                added_tokens_len = 0
                for i, idx in enumerate(idx_list):
                    if 'image_sizes' in image_inputs:
                        orig_height, orig_width = image_inputs['image_sizes'][i].tolist()
                        num_image_tokens = self.processor._get_number_of_features(orig_height, orig_width, height,
                                                                                  width)
                    else:
                        num_image_tokens = (height // self.processor.patch_size) * (
                            width // self.processor.patch_size) + self.processor.num_additional_image_tokens
                    if self.processor.vision_feature_select_strategy == 'default':
                        num_image_tokens -= 1
                    input_ids = input_ids[:added_tokens_len + idx] + [self.image_token_index] * num_image_tokens \
                        + input_ids[added_tokens_len + idx + 1:]
                    if labels is not None:
                        labels = labels[:added_tokens_len + idx] + [-100] * num_image_tokens \
                            + labels[added_tokens_len + idx + 1:]
                    added_tokens_len += num_image_tokens - 1
                encoded['input_ids'] = input_ids
                encoded['labels'] = labels
        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.llava1_5_hf,
        prefix=['<s>'],
        prompt=['USER: {{QUERY}}\nASSISTANT:'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        system_prefix=['<s>{{SYSTEM}}\n'],
        template_cls=LlavaHfTemplate,
    ))


class LlavaVideoHfTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<image>\n']
        assert media_type == 'video'
        media_file = inputs.videos[index]
        if media_file.rsplit('.', 1)[-1] in {'jpg', 'png'}:
            return ['<image>\n']
        else:
            inputs.videos[index] = load_video_llava(inputs.videos[index])
            return ['<video>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images or []
        videos = inputs.videos or []
        if len(videos) > 0:
            video_processor = self.processor.video_processor
            video_inputs = video_processor(videos, return_tensors='pt').to(self.model_info.torch_dtype)
            encoded['pixel_values_videos'] = video_inputs['pixel_values_videos']
        if len(images) > 0:
            image_processor = self.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model_info.torch_dtype)
            encoded['pixel_values'] = image_inputs['pixel_values']
            encoded['image_sizes'] = image_inputs['image_sizes']
        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.llava_next_video_hf,
        prefix=['{{SYSTEM}} '],
        prompt=['USER: {{QUERY}} ASSISTANT:'],
        chat_sep=[' '],
        suffix=[['eos_token_id']],
        template_cls=LlavaVideoHfTemplate,
        auto_add_bos=True,
    ))


class Llava1_6HfTemplate(LlavaHfTemplate):

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        for b in batch:
            pixel_values = b.get('pixel_values')
            if pixel_values is not None:
                b['pixel_values'] = pixel_values.squeeze(0)  # 5d -> 4d
        res = super()._data_collator(batch, padding_to=padding_to)
        return res


@dataclass
class LlavaMistralTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<s>[INST] '])
    prompt: Prompt = field(default_factory=lambda: ['{{QUERY}} [/INST]'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['</s>[INST] '])
    suffix: Prompt = field(default_factory=lambda: ['</s>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<<SYS>>\n{{system}}\n<</SYS>>\n\n'])


register_template(LlavaMistralTemplateMeta(MLLMTemplateType.llava1_6_mistral_hf, template_cls=Llava1_6HfTemplate))

register_template(
    TemplateMeta(
        MLLMTemplateType.llava1_6_vicuna_hf,
        prefix=['<s>'],
        prompt=['USER: {{QUERY}} ASSISTANT:'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        default_system=('A chat between a curious human and an artificial intelligence assistant. '
                        "The assistant gives helpful, detailed, and polite answers to the human's questions."),
        system_prefix=['<s>{{SYSTEM}} '],
        template_cls=Llava1_6HfTemplate))


class LLava1_6YiHfTemplate(Llava1_6HfTemplate):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode == 'vllm':
            return [[64000], '\n']
        else:
            return super().replace_tag(media_type, index, inputs)


register_template(ChatmlTemplateMeta(
    MLLMTemplateType.llava1_6_yi_hf,
    template_cls=LLava1_6YiHfTemplate,
))

register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.llama3_llava_next_hf,
        template_cls=Llava1_6HfTemplate,
        agent_template=None,
    ))

register_template(
    QwenTemplateMeta(MLLMTemplateType.llava_next_qwen_hf, template_cls=Llava1_6HfTemplate, agent_template=None))


class LlavaOneVisionHfTemplate(Llava1_6HfTemplate):

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, 151646)  # <image>
        processor = self.processor
        if images:
            image_processor = processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model_info.torch_dtype)
            height, width = image_inputs['pixel_values'][0].shape[-2:]
            added_tokens_len = 0
            for idx, pixel_v, image_size in zip(idx_list, image_inputs['pixel_values'], image_inputs['image_sizes']):
                if isinstance(image_size, torch.Tensor):
                    image_size = image_size.tolist()
                orig_height, orig_width = image_size
                num_image_tokens = processor._get_number_of_features(orig_height, orig_width, height, width)
                input_ids = input_ids[:added_tokens_len
                                      + idx] + [151646] * num_image_tokens + input_ids[added_tokens_len + idx + 1:]
                if labels is not None:
                    labels = labels[:added_tokens_len + idx] + [-100] * num_image_tokens + labels[added_tokens_len + idx
                                                                                                  + 1:]
                added_tokens_len += num_image_tokens - 1
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
            encoded['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                encoded['image_sizes'] = image_inputs['image_sizes']
        return encoded


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.llava_onevision_hf,
        default_system=None,
        template_cls=LlavaOneVisionHfTemplate,
        agent_template=None,
    ))


class LlavaLlama3_1HfTemplate(LlavaHfTemplate):
    # DaozeZhang
    system = ('You are a helpful language and vision assistant. '
              'You are able to understand the visual content that the user provides, '
              'and assist the user with a variety of tasks using natural language.')

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if len(encoded['pixel_values'].shape) == 5:  # (1, num_patch, 3, H/W, W/H)
            encoded['pixel_values'] = torch.squeeze(encoded['pixel_values'], dim=0)  # (num_patch, 3, H/W, W/H)
        return encoded


register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.llava_llama3_1_hf,
        default_system=LlavaLlama3_1HfTemplate.system,
        template_cls=LlavaLlama3_1HfTemplate,
        agent_template=None,
    ))


class LLavaLlama3HfTemplate(Template):
    # xtuner
    image_placeholder = ['<image>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        raw_image = inputs.images
        if raw_image:
            pixel_values = self.processor.image_processor(raw_image, return_tensors='pt')['pixel_values']
            encoded['pixel_values'] = pixel_values.to(self.model_info.torch_dtype)
        return encoded


register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.llava_llama3_hf,
        template_cls=LLavaLlama3HfTemplate,
        agent_template=None,
    ))


class LLavaTemplate(Template):
    skip_prompt = False
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images or []
        image_sizes = [x.size for x in images]
        from llava.mm_utils import process_images
        model = self.model.model
        if not hasattr(model, 'vision_tower'):
            model = model.model
        image_processor = model.vision_tower.image_processor
        if images:
            images_tensor = process_images(images, image_processor, model.config)
            encoded['images'] = images_tensor.to(model.dtype).squeeze(0)
            encoded['image_sizes'] = image_sizes
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = images
            res['image_sizes'] = sum([b['image_sizes'] for b in batch if 'image_sizes' in b], start=[])
        return res


register_template(LlavaMistralTemplateMeta(MLLMTemplateType.llava1_6_mistral, template_cls=LLavaTemplate))

register_template(ChatmlTemplateMeta(MLLMTemplateType.llava1_6_yi, template_cls=LLavaTemplate))

register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.llama3_llava_next,
        template_cls=LLavaTemplate,
        default_system=('You are a helpful language and vision assistant. '
                        'You are able to understand the visual content that the user provides, '
                        'and assist the user with a variety of tasks using natural language.'),
        agent_template=None,
    ))

register_template(QwenTemplateMeta(MLLMTemplateType.llava_next_qwen, template_cls=LLavaTemplate, agent_template=None))


class LLavaOneVision1_5Template(Template):
    image_token_id = 151655
    video_token_id = 151656
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    use_model = True
    support_padding_free = True

    def init_env_args(self):
        super().init_env_args()
        self.bbox_format = get_env_args('QWENVL_BBOX_FORMAT', str, 'legacy')

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            video = inputs.videos[index]
            video, video_kwargs = fetch_video({'video': video}, return_video_sample_fps=True)
            inputs.mm_processor_kwargs.setdefault('fps', []).append(video_kwargs)
            tokens = ['<|vision_start|><|video_pad|><|vision_end|>']
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return tokens

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        if self.bbox_format == 'legacy':
            return [f'<|object_ref_start|>{ref}<|object_ref_end|>']
        else:
            return [ref]

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        if self.bbox_format == 'legacy':
            return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']
        else:
            return [str(bbox)]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            if mm_data:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(images=mm_data, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    kwargs = {}
                    if hasattr(processor, 'video_processor'):
                        processor_func = processor.video_processor
                    else:
                        processor_func = processor.image_processor
                        kwargs['images'] = None
                    media_inputs = processor_func(videos=mm_data, return_tensors='pt', do_resize=False, **kwargs)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    return [media_token] * token_len

                input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                    _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        base_model = self.get_base_model(model)
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
        inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.visual, self.processor, model.config)
        return {'inputs_embeds': inputs_embeds}


register_template(
    QwenTemplateMeta(MLLMTemplateType.llava_onevision1_5, template_cls=LLavaOneVision1_5Template, agent_template=None))


class LLavaOneVision2Template(Template):
    image_token_id = 151655
    video_token_id = 151656
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    use_model = True
    support_padding_free = True

    def init_env_args(self):
        super().init_env_args()
        self.bbox_format = get_env_args('QWENVL_BBOX_FORMAT', str, 'legacy')

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            video = inputs.videos[index]
            # fetch_video expects a str path or list/tuple of frames.
            # Convert raw tensor to list of PIL images for compatibility.
            if isinstance(video, torch.Tensor):
                import numpy as np
                from PIL import Image as PILImage
                if video.dim() == 4:
                    # (T, C, H, W) -> list of PIL images
                    frames = [PILImage.fromarray(v.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                              for v in video]
                elif video.dim() == 3:
                    # (C, H, W) -> single frame
                    frames = [PILImage.fromarray(video.permute(1, 2, 0).cpu().numpy().astype(np.uint8))]
                else:
                    frames = video
                video, video_kwargs = fetch_video({'video': frames}, return_video_sample_fps=True)
            else:
                video, video_kwargs = fetch_video({'video': video}, return_video_sample_fps=True)
            inputs.mm_processor_kwargs.setdefault('fps', []).append(video_kwargs)
            tokens = ['<|vision_start|><|video_pad|><|vision_end|>']
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return tokens

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        if self.bbox_format == 'legacy':
            return [f'<|object_ref_start|>{ref}<|object_ref_end|>']
        else:
            return [ref]

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        if self.bbox_format == 'legacy':
            return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']
        else:
            return [str(bbox)]

    @staticmethod
    def _convert_positions_to_block_layout(positions, t, h, w, spatial_merge_size=2):
        """Reorder [t*h*w, 3] row-major positions to 2x2 block layout."""
        sms = spatial_merge_size
        if sms == 1:
            return positions
        total = t * h * w
        indices = torch.arange(total).view(t, h, w)
        h_m, w_m = h // sms, w // sms
        indices = indices.view(t, h_m, sms, w_m, sms).permute(0, 1, 3, 2, 4).contiguous().view(total)
        return positions[indices]

    @staticmethod
    def _build_patch_positions(grid_thw, spatial_merge_size=2, frame_indices=None):
        """Build block-layout [t,h,w] patch positions for images/videos.

        Args:
            grid_thw: [num_samples, 3] LongTensor with [T, H, W] per sample.
            spatial_merge_size: vision tower spatial-merge size (default 2).
            frame_indices: optional list of 1-D LongTensors for the t-coordinate.
                Each entry has length T for that sample. When provided, the t-axis
                encodes actual frame positions (training convention). Pass None for
                an entry to fall back to dense arange(T).
        Returns:
            [sum(T*H*W), 3] Int64 tensor in block layout.
        """
        out = []
        for sample_idx, row in enumerate(grid_thw):
            t_v, h_v, w_v = int(row[0]), int(row[1]), int(row[2])
            h_coords = torch.arange(h_v, dtype=torch.int64).repeat_interleave(w_v).repeat(t_v)
            w_coords = torch.arange(w_v, dtype=torch.int64).repeat(h_v).repeat(t_v)
            sample_frame_idx = None
            if frame_indices is not None and sample_idx < len(frame_indices):
                sample_frame_idx = frame_indices[sample_idx]
            if sample_frame_idx is not None:
                fi = torch.as_tensor(sample_frame_idx, dtype=torch.int64)
                if fi.numel() != t_v:
                    raise ValueError(
                        f'frame_indices[{sample_idx}] has length {fi.numel()} but '
                        f'grid_thw[{sample_idx}, 0] = {t_v}')
                t_coords = fi.repeat_interleave(h_v * w_v)
            else:
                t_coords = torch.arange(t_v, dtype=torch.int64).repeat_interleave(h_v * w_v)
            pp = torch.stack([t_coords, h_coords, w_coords], dim=1)
            pp = LLavaOneVision2Template._convert_positions_to_block_layout(pp, t_v, h_v, w_v, spatial_merge_size)
            out.append(pp)
        return torch.cat(out, dim=0)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        all_patch_positions_parts = []
        merge_size = processor.image_processor.merge_size

        # Process images
        mm_data = inputs.images
        if mm_data:
            media_inputs = processor.image_processor(images=mm_data, return_tensors='pt', do_resize=False)
            image_grid_thw = media_inputs['image_grid_thw']
            idx_list = findall(input_ids, self.image_token_id)
            merge_length = merge_size**2

            def _get_new_tokens(i):
                token_len = (image_grid_thw[i].prod() // merge_length)
                return [self.image_token_id] * token_len

            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_tokens)
            encoded.update(media_inputs)
            # Build patch_positions for images
            image_patch_positions = self._build_patch_positions(image_grid_thw, merge_size, frame_indices=None)
            all_patch_positions_parts.append(image_patch_positions)

        # Process videos
        mm_data = inputs.videos
        if mm_data:
            # Convert tensor videos to list[np.ndarray] for the OV 2.0 video processor
            # which accepts list[PIL.Image] or list[np.ndarray] but not raw tensors
            import numpy as np
            converted_videos = []
            for video in mm_data:
                if isinstance(video, torch.Tensor):
                    # video: (T, C, H, W) uint8 tensor -> list of (H, W, C) np.ndarray
                    if video.dim() == 4:
                        converted_videos.append([v.permute(1, 2, 0).numpy() for v in video])
                    elif video.dim() == 3:
                        # Single frame: (C, H, W) -> [(H, W, C)]
                        converted_videos.append([video.permute(1, 2, 0).numpy()])
                    else:
                        converted_videos.append(video)
                else:
                    converted_videos.append(video)

            # Use video_processor if available, otherwise fall back to image_processor
            if hasattr(processor, 'video_processor') and processor.video_processor is not None:
                video_inputs = processor.video_processor(videos=converted_videos, return_tensors='pt')
            else:
                video_inputs = processor.image_processor(videos=converted_videos, return_tensors='pt',
                                                         images=None)

            video_grid_thw = video_inputs['video_grid_thw']
            video_patch_positions = video_inputs.get('patch_positions')
            frame_timestamps = video_inputs.get('frame_timestamps')

            sms = merge_size
            merge_length = sms**2

            # For frames backend: expand video_grid_thw[T,H,W] into T rows of [1,H,W]
            # and rewrite <|video_pad|> placeholders into per-frame blocks
            expanded_grid_thw_rows = []
            video_idx = 0

            # Find all <|vision_start|><|video_pad|><|vision_end|> patterns in input_ids
            vision_start_id = 151652
            vision_end_id = 151653
            video_pad_id = self.video_token_id

            added_offset = 0
            for v_row_idx in range(video_grid_thw.shape[0]):
                T_v = int(video_grid_thw[v_row_idx, 0])
                H_v = int(video_grid_thw[v_row_idx, 1])
                W_v = int(video_grid_thw[v_row_idx, 2])
                n_per_frame = (H_v * W_v) // (sms * sms)

                # Get frame timestamps for this video
                if frame_timestamps is not None and video_idx < len(frame_timestamps):
                    seconds_seq = frame_timestamps[video_idx]
                    if len(seconds_seq) < T_v:
                        seconds_seq = list(seconds_seq) + [seconds_seq[-1] if seconds_seq else 0.0] * (
                            T_v - len(seconds_seq))
                    else:
                        seconds_seq = list(seconds_seq[:T_v])
                else:
                    seconds_seq = [float(i) for i in range(T_v)]

                # Build expanded token sequence for this video
                # Each frame: <X.X seconds> <|vision_start|> <|image_pad|>*n <|vision_end|> \n
                expanded_tokens = []
                expanded_labels = []
                for frame_i in range(T_v):
                    # Timestamp text
                    sec = seconds_seq[frame_i]
                    timestamp_text = f'<{sec:.1f} seconds>'
                    ts_token_ids = self.tokenizer.encode(timestamp_text, add_special_tokens=False)
                    expanded_tokens.extend(ts_token_ids)
                    expanded_labels.extend([-100] * len(ts_token_ids))

                    # Vision tokens for this frame
                    expanded_tokens.append(vision_start_id)
                    expanded_labels.append(-100)
                    expanded_tokens.extend([self.image_token_id] * n_per_frame)
                    expanded_labels.extend([-100] * n_per_frame)
                    expanded_tokens.append(vision_end_id)
                    expanded_labels.append(-100)

                    # Newline between frames (except last frame)
                    if frame_i < T_v - 1:
                        newline_ids = self.tokenizer.encode('\n', add_special_tokens=False)
                        expanded_tokens.extend(newline_ids)
                        expanded_labels.extend([-100] * len(newline_ids))

                # Find the <|vision_start|><|video_pad|><|vision_end|> pattern in input_ids
                found = False
                for pos in range(len(input_ids) - 2):
                    actual_pos = pos + added_offset
                    if (input_ids[actual_pos] == vision_start_id
                            and input_ids[actual_pos + 1] == video_pad_id
                            and input_ids[actual_pos + 2] == vision_end_id):
                        # Replace the 3-token pattern with expanded tokens
                        input_ids = input_ids[:actual_pos] + expanded_tokens + input_ids[actual_pos + 3:]
                        if labels is not None:
                            labels = labels[:actual_pos] + expanded_labels + labels[actual_pos + 3:]
                        if loss_scale is not None:
                            scale_val = loss_scale[actual_pos + 1]  # use the video_pad scale
                            expanded_loss_scale = [scale_val] * len(expanded_tokens)
                            loss_scale = loss_scale[:actual_pos] + expanded_loss_scale + loss_scale[actual_pos + 3:]
                        added_offset += len(expanded_tokens) - 3
                        found = True
                        break
                if not found:
                    raise ValueError(
                        f'Could not find <|vision_start|><|video_pad|><|vision_end|> pattern '
                        f'in input_ids for video {video_idx}')

                # Expand grid_thw: [T, H, W] -> T rows of [1, H, W]
                for _ in range(T_v):
                    expanded_grid_thw_rows.append([1, H_v, W_v])

                video_idx += 1

            # Set up the aliased video-as-image data
            video_pixel_values = video_inputs.get('pixel_values_videos', video_inputs.get('pixel_values'))
            expanded_image_grid_thw = video_grid_thw.new_tensor(expanded_grid_thw_rows)

            # Merge with existing image data if both present
            if encoded.get('pixel_values') is not None:
                encoded['pixel_values'] = torch.cat([encoded['pixel_values'], video_pixel_values], dim=0)
                encoded['image_grid_thw'] = torch.cat([encoded['image_grid_thw'], expanded_image_grid_thw], dim=0)
            else:
                encoded['pixel_values'] = video_pixel_values
                encoded['image_grid_thw'] = expanded_image_grid_thw
            # Remove video-specific keys if present (we aliased to image path)
            encoded.pop('video_grid_thw', None)
            encoded.pop('pixel_values_videos', None)

            # Use patch_positions from video processor, or build if not available
            if video_patch_positions is not None:
                all_patch_positions_parts.append(video_patch_positions)
            else:
                # Build patch_positions for the expanded grid_thw
                video_pp = self._build_patch_positions(expanded_image_grid_thw, merge_size, frame_indices=None)
                all_patch_positions_parts.append(video_pp)

        # Combine patch_positions from all media types
        if all_patch_positions_parts:
            if len(all_patch_positions_parts) == 1:
                encoded['patch_positions'] = all_patch_positions_parts[0]
            else:
                encoded['patch_positions'] = torch.cat(all_patch_positions_parts, dim=0)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        base_model = self.get_base_model(model)
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
        # Build dummy patch_positions for plain-text samples (DeepSpeed ZeRO-3 compat)
        if inputs.get('patch_positions') is None:
            from PIL import Image
            images = [Image.new('RGB', (32, 32), (0, 0, 0))]
            media_inputs = self.processor.image_processor(images=images, return_tensors='pt')
            inputs['patch_positions'] = self._build_patch_positions(
                media_inputs['image_grid_thw'],
                self.processor.image_processor.merge_size,
                frame_indices=None)
        inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.visual, self.processor, model.config)
        return {'inputs_embeds': inputs_embeds}


register_template(
    QwenTemplateMeta(MLLMTemplateType.llava_onevision2, template_cls=LLavaOneVision2Template, agent_template=None))
