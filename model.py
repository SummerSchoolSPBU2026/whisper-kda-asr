import torch
from transformers import AutoProcessor, WhisperModel, GenerationConfig
from fla.models import KDAConfig
import torch.nn as nn
from fla.layers.kda import KimiDeltaAttention
from fla.modules import GatedMLP as KDAMLP
from fla.modules import RMSNorm
from fla.models.kda.modeling_kda import KDAPreTrainedModel
from fla.models.utils import Cache, FLAGenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput
from utils import shift_tokens_right, resize_audio_attention_mask

WHISPER_MODEL_NAME = "openai/whisper-tiny"

def create_kda_config(
    tokenizer,
    max_target_length=448,
    freeze_encoder=True,
):
    required_ids = {
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    missing = [
    name
    for name, token_id in required_ids.items()
    if token_id is None
    ]

    if missing:
        raise ValueError(
            f"У токенизатора отсутствуют специальные токены: {missing}"
        )

    if len(set(required_ids.values())) != len(required_ids):
        raise ValueError(
            "PAD, BOS и EOS должны иметь разные token ID"
        )

    return KDAConfig(
        vocab_size=len(tokenizer),
        hidden_size=384,
        num_hidden_layers=4,
        num_heads=6,
        head_dim=64,
        num_v_heads=6,
        expand_v=1.0,
        intermediate_size=1024,

        attn_mode="chunk",
        attn=None,
        use_short_conv=True,
        conv_size=4,
        norm_eps=1e-6,
        attnres_block_size=None,

        tie_word_embeddings=True,
        use_cache=False,
        fuse_norm=True,
        fuse_swiglu=True,
        fuse_cross_entropy=False,

        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,

        decoder_start_token_id=tokenizer.bos_token_id,
        decoder_prompt_ids=[],
        max_target_length=max_target_length,
        is_encoder_decoder=True,

        whisper_model_name=WHISPER_MODEL_NAME,
        freeze_encoder=freeze_encoder,
    )

class KDACrossAttentionBlock(nn.Module):
    def __init__(self, config: KDAConfig, layer_idx: int):
        super().__init__()

        if config.attnres_block_size is not None:
              raise ValueError(
                  "attnres_block_size=None"
              )

        self.config = config
        self.layer_idx = layer_idx

        Norm = RMSNorm if config.fuse_norm else nn.RMSNorm

        self.attn_norm = Norm(
            config.hidden_size,
            eps=config.norm_eps
        )
        
        self.attn = KimiDeltaAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            expand_v=config.expand_v,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_v_heads=config.num_v_heads,
            use_short_conv=config.use_short_conv,
            allow_neg_eigval=config.allow_neg_eigval,
            safe_gate=config.safe_gate,
            lower_bound=config.lower_bound,
            conv_size=config.conv_size,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx,
        )

        self.cross_attn_norm = Norm(
            config.hidden_size,
            eps=config.norm_eps
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.mlp_norm = Norm(
              config.hidden_size,
              eps=config.norm_eps,
        )
        self.mlp = KDAMLP(
              hidden_size=config.hidden_size,
              hidden_ratio=config.hidden_ratio,
              intermediate_size=config.intermediate_size,
              hidden_act=config.hidden_act,
              fuse_swiglu=config.fuse_swiglu,
        )

    def _add_residual_and_norm(
        self,
        hidden_states,
        residual,
        norm
    ):
        if self.config.fuse_norm:
            # Одновременно:
            # residual = residual + hidden_states
            # hidden_states = norm(residual)
            hidden_states, residual = norm(
                hidden_states,
                residual,
                True,  # prenorm
            )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = norm(hidden_states)

        return hidden_states, residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        if (encoder_hidden_states is not None and encoder_hidden_states.size(-1) != self.config.hidden_size):
            raise ValueError(
                "Размер encoder_hidden_states должен совпадать с "
                f"hidden_size={self.config.hidden_size}, получено "
                f"{encoder_hidden_states.size(-1)}"
        )

        # KDA

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )

        next_norm = (
            self.cross_attn_norm
            if encoder_hidden_states is not None
            else self.mlp_norm
        )

        hidden_states, residual = self._add_residual_and_norm(
            hidden_states,
            residual,
            next_norm,
        )

        # cross-attention
        
        if encoder_hidden_states is not None:
            encoder_padding_mask = None

            if encoder_attention_mask is not None:
                expected_shape = encoder_hidden_states.shape[:2]

                if encoder_attention_mask.shape != expected_shape:
                    raise ValueError(
                        "encoder_attention_mask должна иметь форму "
                        f"{expected_shape}, получено "
                        f"{encoder_attention_mask.shape}"
                    )

                encoder_padding_mask = ~encoder_attention_mask.bool()

            cross_output, _ = self.cross_attn(
                query=hidden_states,
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
            )

            hidden_states, residual = self._add_residual_and_norm(
                cross_output,
                residual,
                self.mlp_norm,
            )

        # Gated MLP / SwiGLU

        hidden_states = self.mlp(
            hidden_states,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        return (
            hidden_states,
            attentions,
            past_key_values,
            None, # AttnRes отключён
        )

class KDACrossAttentionBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )

        self.layers = nn.ModuleList([
            KDACrossAttentionBlock(
                config=config,
                layer_idx=layer_idx,
            )
            for layer_idx in range(config.num_hidden_layers)
        ])

        Norm = RMSNorm if config.fuse_norm else nn.RMSNorm

        self.norm = Norm(
            config.hidden_size,
            eps=config.norm_eps,
        )

        self.gradient_checkpointing = False

class KDACrossAttentionDecoder(KDAPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys= {
        "lm_head.weight": "model.embeddings.weight"
    }

    _no_split_modules = ["KDACrossAttentionBlock"]

    def __init__(self, config):
        # Создаёт embeddings, norm и LM head как в KDAForCausalLM.
        super().__init__(config)

        self.model = KDACrossAttentionBackbone(config)
        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # Инициализирует новые блоки по правилам KDA.
        self.post_init()

        # Связывает веса embedding и LM head.
        self.tie_weights()

    def get_input_embeddings(self):
          return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def forward(
        self,
        input_ids,
        decoder_attention_mask=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        return_dict=True,
        attention_mask=None,
        encoder_hidden_states=None,
        labels=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("Необходимо передать input_ids")

        use_cache = (
            self.config.use_cache
            if use_cache is None
            else use_cache
        )

        # Во время обучения cache не используется.
        if self.training:
            use_cache = False

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(
                past_key_values
            )

        hidden_states = self.model.embeddings(input_ids)

        for layer in self.model.layers:
            (
                hidden_states,
                _,
                past_key_values,
                _,
            ) = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attention_mask=decoder_attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=False,
                **kwargs,
            )

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.float().reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

        if not return_dict:
            return logits, past_key_values

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )

class WhisperKDAModel(KDAPreTrainedModel):
    config_class = KDAConfig
    base_model_prefix = ""
    main_input_name = "input_features"
    supports_gradient_checkpointing = False

    _no_split_modules = ["KDACrossAttentionBlock"]

    _tied_weights_keys = {
        "decoder.lm_head.weight":
        "decoder.model.embeddings.weight"
    }

    def __init__(
        self,
        config,
    ):
        super().__init__(config)

        whisper = WhisperModel.from_pretrained(
            config.whisper_model_name,
            attn_implementation="sdpa",
        )

        self.whisper_config = whisper.config
        self.encoder = whisper.encoder

        encoder_hidden_size = whisper.config.d_model
        decoder_hidden_size = config.hidden_size

        if encoder_hidden_size == decoder_hidden_size:
            self.encoder_projection = nn.Identity()
        else:
            self.encoder_projection = nn.Linear(
                encoder_hidden_size,
                decoder_hidden_size,
                bias=False,
            )

        self.decoder = KDACrossAttentionDecoder(config)

        self.encoder_is_frozen = False

        if config.freeze_encoder:
            self.freeze_encoder()
        
        self.generation_config = GenerationConfig(
            max_length=config.max_target_length,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            decoder_start_token_id=config.decoder_start_token_id,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
        )

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)
        self.encoder_is_frozen = True

    def unfreeze_encoder(self):
        self.encoder.requires_grad_(True)
        self.encoder_is_frozen = False

    def get_encoder(self):
      return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def encode(
        self,
        input_features,
        attention_mask=None,
    ):
        if self.encoder_is_frozen:
            self.encoder.eval()

            with torch.no_grad():
                encoder_outputs = self.encoder(
                    input_features=input_features,
                    return_dict=True,
                )
        else:
            encoder_outputs = self.encoder(
                input_features=input_features,
                return_dict=True,
            )

        encoder_hidden_states = self.encoder_projection(
            encoder_outputs.last_hidden_state
        )

        encoder_attention_mask = resize_audio_attention_mask(
            attention_mask=attention_mask,
            target_length=encoder_hidden_states.size(1),
        )

        return encoder_hidden_states, encoder_attention_mask

    def train(self, mode=True):
        super().train(mode)

        if self.encoder_is_frozen:
            self.encoder.eval()
        
        return self

    def forward(
        self,
        input_features=None,
        decoder_input_ids=None,
        attention_mask=None,
        decoder_attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        return_dict=True,
        **kwargs,
    ):
        if encoder_hidden_states is None:
            if input_features is None:
                raise ValueError(
                    "Нужно передать input_features или "
                    "encoder_hidden_states"
                )

            (
                encoder_hidden_states,
                computed_encoder_mask,
            ) = self.encode(
                input_features=input_features,
                attention_mask=attention_mask,
            )

            if encoder_attention_mask is None:
                encoder_attention_mask = computed_encoder_mask

        if decoder_input_ids is None:
            if labels is None:
                raise ValueError(
                    "Нужно передать decoder_input_ids или labels"
                )

            decoder_input_ids = shift_tokens_right(
                labels=labels,
                pad_token_id=self.decoder.config.pad_token_id,
                decoder_start_token_id=(
                    self.config.decoder_start_token_id
                ),
            )

        if decoder_attention_mask is None:
            decoder_attention_mask = decoder_input_ids.ne(
                self.decoder.config.pad_token_id
            ).long()

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

        logits = decoder_outputs.logits

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (
                logits,
                decoder_outputs.past_key_values,
                encoder_hidden_states,
            )

            return (loss,) + output if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            encoder_last_hidden_state=encoder_hidden_states,
        )
    
    def load_pretrained_decoder(
      self,
      checkpoint_path,
    ):
        pretrained_decoder = (
            KDACrossAttentionDecoder.from_pretrained(
                checkpoint_path
            )
        )

        architecture_fields = (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_heads",
            "head_dim",
            "intermediate_size",
            "num_v_heads",
            "expand_v",
            "use_short_conv",
            "conv_size",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
        )

        mismatches = []

        for field in architecture_fields:
            expected = getattr(self.config, field)
            actual = getattr(
                pretrained_decoder.config,
                field,
            )

            if expected != actual:
                mismatches.append(
                    f"{field}: ASR={expected}, checkpoint={actual}"
                )

        if mismatches:
            raise ValueError(
                "Decoder checkpoint несовместим:\n"
                + "\n".join(mismatches)
            )

        pretrained_state = {
            name: tensor
            for name, tensor
            in pretrained_decoder.state_dict().items()
            if "cross_attn" not in name
        }

        incompatible = self.decoder.load_state_dict(
            pretrained_state,
            strict=False,
        )

        unexpected = incompatible.unexpected_keys

        invalid_missing = [
            name
            for name in incompatible.missing_keys
            if "cross_attn" not in name
        ]

        if unexpected or invalid_missing:
            raise RuntimeError(
                "Ошибка загрузки decoder checkpoint: "
                f"missing={invalid_missing}, "
                f"unexpected={unexpected}"
            )

        self.decoder.tie_weights()

        if (
            self.decoder.lm_head.weight
            is not self.decoder.model.embeddings.weight
        ):
            raise RuntimeError(
                "После загрузки потеряна связь embeddings/lm_head"
            )

        del pretrained_decoder


    @torch.no_grad()
    def generate(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        generation_config=None,
        max_length=None,
        max_new_tokens=None,
        num_beams=None,
        do_sample=None,
        use_cache=None,
        **kwargs,
    ):
        if input_features is None:
            raise ValueError(
                "Для генерации необходимо передать input_features"
            )

        generation_config = (
            generation_config or self.generation_config
        )

        num_beams = (
            generation_config.num_beams
            if num_beams is None
            else num_beams
        )

        do_sample = (
            generation_config.do_sample
            if do_sample is None
            else do_sample
        )

        use_cache = (
            generation_config.use_cache
            if use_cache is None
            else use_cache
        )

        if num_beams != 1:
            raise NotImplementedError(
                "KDA cache пока поддерживается только "
                "для greedy decoding: num_beams=1"
            )

        if do_sample:
            raise NotImplementedError(
                "Sampling пока не реализован"
            )

        device = next(self.parameters()).device

        input_features = input_features.to(device)

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        was_training = self.training
        self.eval()

        try:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                (
                    encoder_hidden_states,
                    encoder_attention_mask,
                ) = self.encode(
                    input_features=input_features,
                    attention_mask=attention_mask,
                )

            batch_size = input_features.size(0)

            if decoder_input_ids is None:
                prompt_ids = [
                    self.config.decoder_start_token_id,
                    *self.config.decoder_prompt_ids,
                ]

                generated_ids = torch.tensor(
                    prompt_ids,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0).repeat(batch_size, 1)
            else:
                generated_ids = decoder_input_ids.to(device)

            if max_new_tokens is None:
                max_new_tokens = generation_config.max_new_tokens

            if max_length is None:
                max_length = generation_config.max_length

            if max_new_tokens is not None:
                stopping_length = (
                    generated_ids.size(1) + max_new_tokens
                )
            elif max_length is not None:
                stopping_length = max_length
            else:
                raise ValueError(
                    "Нужно установить max_new_tokens или max_length"
                )

            if stopping_length <= generated_ids.size(1):
                raise ValueError(
                    "Длина генерации должна быть больше длины prompt"
                )

            eos_token_id = self.config.eos_token_id
            if eos_token_id is None:
                raise ValueError("eos_token_id не установлен")

            pad_token_id = (
                self.config.pad_token_id
                if self.config.pad_token_id is not None
                else eos_token_id
            )

            finished = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=device,
            )

            past_key_values = None

            while generated_ids.size(1) < stopping_length:
                if use_cache and past_key_values is not None:
                    step_input_ids = generated_ids[:, -1:]
                else:
                    step_input_ids = generated_ids

                decoder_attention_mask = torch.ones_like(
                    step_input_ids,
                    dtype=torch.long,
                )

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    outputs = self.decoder(
                        input_ids=step_input_ids,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attention_mask=decoder_attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        return_dict=True,
                    )

                next_token = outputs.logits[:, -1].argmax(
                    dim=-1,
                    keepdim=True,
                )

                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )

                generated_ids = torch.cat(
                    [generated_ids, next_token],
                    dim=1,
                )

                finished |= next_token.squeeze(1).eq(
                    eos_token_id
                )

                past_key_values = outputs.past_key_values

                if finished.all():
                    break

            return generated_ids

        finally:
            self.train(was_training)
