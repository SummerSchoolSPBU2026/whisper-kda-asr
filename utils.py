import torch

def shift_tokens_right(
    labels,
    pad_token_id,
    decoder_start_token_id,
):
    shifted = labels.new_full(
        labels.shape,
        pad_token_id,
    )

    shifted[:, 0] = decoder_start_token_id
    shifted[:, 1:] = labels[:, :-1]

    shifted.masked_fill_(
        shifted.eq(-100),
        pad_token_id,
    )

    return shifted


def resize_audio_attention_mask(
    attention_mask,
    target_length,
):
    if attention_mask is None:
        return None

    if attention_mask.ndim != 2:
        raise ValueError(
            "Audio attention_mask должна иметь форму [batch, length]"
        )

    source_length = attention_mask.size(1)
    valid_lengths = attention_mask.long().sum(dim=1)

    target_lengths = (
        valid_lengths * target_length + source_length - 1
    ) // source_length

    target_lengths = target_lengths.clamp(
        min=0,
        max=target_length,
    )

    positions = torch.arange(
        target_length,
        device=attention_mask.device,
    )

    return (
        positions.unsqueeze(0)
        < target_lengths.unsqueeze(1)
    ).long()