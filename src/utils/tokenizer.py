import torch


def map_character_spans_to_token_ids(
    offset_mapping: torch.Tensor,
    char_spans: list[list[int]],
):
    token_ids = []
    for char_span in char_spans:
        start = char_span[0]
        end = char_span[1]

        start_token_id = torch.where(offset_mapping[0, :, 1] > start)[0][0].item()

        if end > offset_mapping.max():
            end_token_id = len(offset_mapping[0]) - 1
        else:
            end_token_id = torch.where(offset_mapping[0, :, 1] >= end)[0][0].item() + 1

        token_ids.extend(list(range(start_token_id, end_token_id)))

    return token_ids


def map_n_characters_to_n_tokens(
    offset_mapping: torch.Tensor,
    n_characters: int,
):
    return torch.where(offset_mapping[0, :, 0] >= n_characters)[0][0].item()
