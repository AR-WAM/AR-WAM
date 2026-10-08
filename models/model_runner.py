import torch
import torch.nn as nn
import logging
import json
from contextlib import nullcontext
from transformers import AutoModel, AutoConfig

from .vla_model_fm import VLAModel, calc_flow_matching_loss


logger = logging.getLogger(__name__)


class ModelFactory:
    """Initializes the DINOv3 vision encoder and the action prediction model"""

    @staticmethod
    def create_vision_encoder(checkpoint_path, dtype=torch.bfloat16, device="cuda"):
        """
        Load a frozen DINOv3 ViT from a local path (offline).

        Returns:
            model: DINOv3 model (eval mode, frozen)
            hidden_size: int, model hidden dimension
            num_register_tokens: int, number of register tokens (usually 4)
            patch_size: int, ViT patch size (usually 16)
        """
        logger.info(f"Loading frozen DINOv3 vision encoder from {checkpoint_path}...")

        # Read config first for metadata
        config = AutoConfig.from_pretrained(checkpoint_path, local_files_only=True)

        model = AutoModel.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device)

        # Freeze
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        hidden_size = getattr(config, "hidden_size", None)
        num_register_tokens = getattr(config, "num_register_tokens", 4)
        patch_size = getattr(config, "patch_size", 16)

        if hidden_size is None:
            # fallback: infer with a dummy forward
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=device, dtype=dtype)
                out = model(pixel_values=dummy)
                hidden_size = out.last_hidden_state.shape[-1]

        logger.info(
            f"DINOv3 loaded: hidden_size={hidden_size}, "
            f"num_register_tokens={num_register_tokens}, patch_size={patch_size}"
        )
        return model, hidden_size, num_register_tokens, patch_size

    @staticmethod
    def create_action_model(config, dino_hidden_size, num_dino_layers, patch_size=16):
        """Create the dense-concat AdaLN training model."""
        logger.info("Initializing VLAModel...")

        model_cfg = config.model
        ae_cfg = model_cfg.action_expert
        ve_cfg = model_cfg.vision_encoder
        ff_cfg = model_cfg.future_feat
        mask_cfg = model_cfg.mask_decoder
        condition_cfg = model_cfg.conditioning
        expected_channels = [
            "right_grasp", "right_carry", "right_place",
            "left_grasp", "left_carry", "left_place",
        ]
        if mask_cfg.channel_names != expected_channels:
            raise ValueError("mask channel_names must match the fixed right/left grasp/carry/place order")
        img_w, img_h = tuple(config.dataset.image_size)

        model = VLAModel(
            action_dim=config.common.action_dim,
            proprio_dim=config.common.state_dim,
            hidden_dim=ae_cfg.hidden_size,
            action_len=config.common.action_chunk_size,
            proprio_len=config.common.proprio_len,
            depth=ae_cfg.depth,
            num_heads=ae_cfg.num_heads,
            dino_feat_dims=tuple([dino_hidden_size] * num_dino_layers),
            concat_out_dim=ve_cfg.concat.out_dim,
            rla_tokens_per_frame=int(ff_cfg.rla_tokens_per_frame),
            rla_token_dim=int(ff_cfg.rla_token_dim),
            mask_decoder_dim=int(mask_cfg.hidden_dim),
            mask_decoder_layers=int(mask_cfg.num_layers),
            mask_decoder_heads=int(mask_cfg.num_heads),
            dino_patch_grid=(img_h // patch_size, img_w // patch_size),
            num_mask_channels=int(mask_cfg.num_channels),
            condition_embedding_init_std=float(condition_cfg.embedding_init_std),
            state_dropout_prob=float(condition_cfg.state_dropout_prob),
            future_mask_bottleneck_dim=int(ff_cfg.future_mask_bottleneck_dim),
        )
        for parameter in model.future_mask_teacher_proj.parameters():
            parameter.requires_grad_(False)
        if model.state_dropout_prob == 0.0:
            model.state_mask_token.requires_grad_(False)
        return model


class VLAWrapper(nn.Module):
    """
    VLA wrapper (DINOv3 version):
    1. Run DINOv3 on input images with output_hidden_states=True
    2. Extract dense multi-layer hidden states according to feat_layers
    3. Call action_model + flow matching loss
    """
    def __init__(self,
                 vision_encoder,
                 action_model,
                 time_sampler,
                 feat_layers,
                 include_cls_register,
                 num_register_tokens,
                 device,
                 dtype,
                 norm_stats_path,
                 train_config,
                 rla_work_dir=None,
                 rla_checkpoint_step=150000,
                 rla_adapter=None,
                 ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.action_model = action_model
        self.time_sampler = time_sampler
        self.feat_layers = list(feat_layers)
        self.include_cls_register = include_cls_register
        self.num_register_tokens = num_register_tokens
        if rla_adapter is None:
            if rla_work_dir is None:
                raise ValueError("rla_work_dir is required for policy training")
            from .rla_targets import FrozenRLAAdapter
            rla_adapter = FrozenRLAAdapter(
                work_dir=rla_work_dir,
                device=device,
                checkpoint_step=rla_checkpoint_step,
            )
        self.rla_adapter = rla_adapter
        self.device = device
        self.dtype = dtype
        # Keep the trainable DiT and wrapper feature tensors on one dtype. The
        # caller may still override this through an explicit model cast.
        if hasattr(self.action_model, "to"):
            self.action_model.to(device=self.device, dtype=self.dtype)

        self.time_mu = train_config['time_mu']
        self.time_sigma = train_config['time_sigma']
        self.lambda_action = train_config['lambda_action']
        self.lambda_rla = train_config['lambda_rla']
        self.lambda_mask = train_config['lambda_mask']
        self.lambda_mask_token = train_config['lambda_mask_token']
        self.lambda_condition_decode = train_config['lambda_condition_decode']
        self.lambda_run_endpoint_state = train_config['lambda_run_endpoint_state']

        logger.info(f"VLAWrapper initialized. feat_layers={self.feat_layers}, "
                    f"include_cls_register={self.include_cls_register}")

        # Load normalization stats
        self.load_norm_stats(norm_stats_path)

    def load_norm_stats(self, path):
        """Read JSON and load action / state min/max for normalization"""
        logger.info(f"Loading normalization stats from {path}...")
        with open(path, 'r') as f:
            data = json.load(f)

        stats = data['robotwin2']
        action_stats = stats['action']
        state_stats = stats['state']

        for name, values, width in (("action", action_stats, 14), ("state", state_stats, 16)):
            minimum = torch.tensor(values['min'], dtype=torch.float32)
            maximum = torch.tensor(values['max'], dtype=torch.float32)
            if minimum.shape != (width,) or maximum.shape != (width,):
                raise ValueError(f"{name} normalization min/max must each contain {width} values")
            if not torch.isfinite(minimum).all() or not torch.isfinite(maximum).all():
                raise ValueError(f"{name} normalization min/max must be finite")
            if (maximum < minimum).any():
                raise ValueError(f"{name} normalization max must be >= min")
            self.register_buffer(f'{name}_min', minimum)
            self.register_buffer(f'{name}_max', maximum)

    @torch.no_grad()
    def _run_vision_output(self, pixel_values):
        pixel_values = pixel_values.to(self.device, self.dtype)
        return self.vision_encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

    def _vision_features_from_hidden_states(self, hidden_states):
        """Select the configured dense DINO layers from one vision pass."""
        feats_list = []
        for layer_idx in self.feat_layers:
            h = hidden_states[layer_idx]   # (B, 1+R+P, D)
            if not self.include_cls_register:
                skip = 1 + self.num_register_tokens
                h = h[:, skip:, :]
            feats_list.append(h)
        return feats_list

    def _normalize_tensor(self, x, min_val, max_val):
        min_v = min_val.to(device=x.device, dtype=x.dtype)
        max_v = max_val.to(device=x.device, dtype=x.dtype)

        denominator = max_v - min_v
        denominator[denominator < 1e-6] = 1.0

        norm_x = 2 * (x - min_v) / denominator - 1
        return norm_x

    def normalize_action(self, action):
        return self._normalize_tensor(action, self.action_min, self.action_max)

    def normalize_state(self, state):
        return self._normalize_tensor(state, self.state_min, self.state_max)

    def forward(self, batch):
        """Forward pass and loss computation"""
        # 1. Vision features (multi-layer DINO).  Keep current and future as
        # separate B-sized forwards: measured H100 throughput is higher than a
        # single 2B forward despite its extra launch.
        pixel_values = batch['pixel_values']      # (B, 3, H, W)
        current_vision_output = self._run_vision_output(pixel_values)
        current_hidden_states = current_vision_output.hidden_states
        dino_features_list = self._vision_features_from_hidden_states(
            current_hidden_states
        )
        mask_decoder_dino = current_vision_output.last_hidden_state

        if batch.get('future_pixel_values') is None:
            raise ValueError("joint RLA training requires future_pixel_values")
        future_pixels = batch['future_pixel_values']
        if future_pixels.ndim != 5 or future_pixels.shape[1] != 1:
            raise ValueError("future_pixel_values must have shape (B,1,3,H,W)")
        future_vision_output = self._run_vision_output(future_pixels[:, 0])
        future_dino_features = self._vision_features_from_hidden_states(
            future_vision_output.hidden_states
        )
        patch_start = 1 + self.num_register_tokens
        current_patches = mask_decoder_dino[:, patch_start:]
        future_patches = future_vision_output.last_hidden_state[
            :, patch_start:
        ].unsqueeze(1)
        future_mask_valid = batch.get('future_mask')
        if future_mask_valid is None:
            raise ValueError("joint RLA training requires future_mask")
        future_mask_valid = torch.as_tensor(
            future_mask_valid, device=self.device, dtype=torch.bool
        )
        future_rla_target = self.rla_adapter.encode_targets(
            current_patches,
            future_patches,
            valid_frames=future_mask_valid,
        )

        # 2. Action / State preparation
        x1_raw = batch['action_sequence'].to(self.device, self.dtype)
        qpos_raw = batch['state'].to(self.device, self.dtype)

        if qpos_raw.dim() == 2:
            qpos_raw = qpos_raw.unsqueeze(1)

        # 3. Normalize
        x1 = self.normalize_action(x1_raw)
        qpos = self.normalize_state(qpos_raw)
        # Align state sequence with proprio_len (history_len=1 when state_indices=[0], qpos_history=qpos)
        qpos_history = qpos
        run_endpoint_state_target = None
        if batch.get('run_endpoint_state') is not None:
            run_endpoint_state_target = self.normalize_state(
                batch['run_endpoint_state'].to(self.device, self.dtype)
            ).detach()
        future_qpos = self.normalize_state(
            batch['future_state'].to(self.device, self.dtype)
        )

        def _future_one(key, dtype=None, long=False):
            value = batch.get(f"future_{key}")
            if value is None:
                return None
            value = torch.as_tensor(value, device=self.device)
            if value.ndim >= 2 and value.shape[1] == 1:
                value = value[:, 0]
            if long:
                return value.long()
            return value.to(dtype or self.dtype)

        future_condition_ids = {
            "skill": _future_one("skill_id", long=True),
            "participants": _future_one("participants_id", long=True),
            "style": _future_one("style_id", long=True),
        }
        if any(value is None for value in future_condition_ids.values()):
            raise ValueError("v4 future condition IDs are required")
        teacher_model = getattr(self.action_model, "module", self.action_model)
        with torch.no_grad():
            if torch.is_autocast_enabled("cuda"):
                # The teacher and student share FP32 weights. Running no_grad
                # first with the outer autocast cache would leave detached BF16
                # casts for the student to reuse.
                teacher_autocast = torch.autocast(
                    device_type="cuda",
                    dtype=torch.get_autocast_dtype("cuda"),
                    cache_enabled=False,
                )
            else:
                teacher_autocast = nullcontext()
            with teacher_autocast:
                future_prefix = teacher_model.prefill_prefix(
                    dino_features_list=future_dino_features,
                    qpos_history=future_qpos,
                    condition_ids=future_condition_ids,
                    bbox_l=_future_one("bbox_l"),
                    bbox_r=_future_one("bbox_r"),
                    valid_l=_future_one("valid_l"),
                    valid_r=_future_one("valid_r"),
                    kind_l=_future_one("kind_l", long=True),
                    kind_r=_future_one("kind_r", long=True),
                )
            future_mask_target = future_prefix["future_mask_bottleneck"].detach()

        if self.lambda_mask and batch.get('visual_mask') is None:
            raise ValueError("current mask supervision requires visual_mask in the batch")
        if self.lambda_mask and batch.get('visual_mask_valid') is None:
            raise ValueError("current mask supervision requires visual_mask_valid in the batch")

        def _maybe(key, dtype=None, long=False):
            v = batch.get(key)
            if v is None:
                return None
            if not torch.is_tensor(v):
                v = torch.as_tensor(v)
            v = v.to(self.device)
            if long:
                return v.long()
            if dtype is None:
                dtype = self.dtype
            return v.to(dtype)

        condition_ids = {
            key: _maybe(f"{key}_id", long=True)
            for key in ("skill", "participants", "style")
        }
        if any(value is None for value in condition_ids.values()):
            raise ValueError("v4 current condition IDs are required")

        # 5. Flow Matching Loss
        loss, info_dic = calc_flow_matching_loss(
            self.action_model,
            x1=x1,
            dino_features_list=dino_features_list,
            qpos_history=qpos_history,
            condition_ids=condition_ids,
            action_mask=_maybe('action_mask', long=False),
            time_sampler=self.time_sampler,
            time_mu=self.time_mu,
            time_sigma=self.time_sigma,
            mask_decoder_dino=mask_decoder_dino,
            visual_mask=_maybe('visual_mask'),
            visual_mask_valid=_maybe('visual_mask_valid', dtype=torch.bool),
            future_mask=future_mask_valid,
            future_rla_target=future_rla_target,
            future_mask_target=future_mask_target,
            lambda_action=self.lambda_action,
            lambda_rla=self.lambda_rla,
            lambda_mask=self.lambda_mask,
            lambda_mask_token=self.lambda_mask_token,
            lambda_condition_decode=self.lambda_condition_decode,
            lambda_run_endpoint_state=self.lambda_run_endpoint_state,
            run_endpoint_state_target=run_endpoint_state_target,
            condition_skill=condition_ids.get("skill"),
            condition_participants=condition_ids.get("participants"),
            condition_style=condition_ids.get("style"),
            condition_valid_l=(
                (_maybe('valid_l') > 0.5).long()
                if batch.get('valid_l') is not None else None
            ),
            condition_valid_r=(
                (_maybe('valid_r') > 0.5).long()
                if batch.get('valid_r') is not None else None
            ),
            condition_kind_l=_maybe('kind_l', long=True),
            condition_kind_r=_maybe('kind_r', long=True),
            bbox_l=_maybe('bbox_l'),
            bbox_r=_maybe('bbox_r'),
            valid_l=_maybe('valid_l'),
            valid_r=_maybe('valid_r'),
            kind_l=_maybe('kind_l', long=True),
            kind_r=_maybe('kind_r', long=True),
        )

        return loss, info_dic
