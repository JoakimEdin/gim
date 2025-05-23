from pathlib import Path
import json
import re
from collections import namedtuple

import numpy as np
from datasets import load_dataset, Dataset, Value
import pandas as pd

dataset_paths = namedtuple(
    "dataset_paths",
    ["annotation_path", "corpus_path", "split_path", "subset_path"],
)
dataset_lookup = {
    "boolq": dataset_paths(
        Path("data/raw/boolq/test_comprehensive.jsonl"),
        Path("data/raw/boolq/docs"),
        None,
        None,
    ),
    "fever": dataset_paths(
        Path("data/raw/fever/test.jsonl"),
        Path("data/raw/fever/docs"),
        None,
        Path("data/raw/fever/subset.csv"),
    ),
    "movies": dataset_paths(
        Path("data/raw/movies/test.jsonl"),
        Path("data/raw/movies/docs"),
        None,
        None,
    ),
    "scifact": dataset_paths(
        Path("data/raw/scifact/claims_dev.jsonl"),
        Path("data/raw/scifact/corpus.jsonl"),
        None,
        None,
    ),
    "hatexplain": dataset_paths(
        Path("data/raw/hatexplain/hatexplain.json"),
        None,
        Path("data/raw/hatexplain/split.json"),
        Path("data/raw/hatexplain/subset.csv"),
    ),
    "twitter": dataset_paths(
        Path("data/raw/twitter/train.csv"),
        None,
        None,
        Path("data/raw/twitter/subset.csv"),
    ),  # only the training set has annotated rationals
}


def load_jsonl_dataset(path: Path) -> list[dict]:
    examples_list = []
    with open(path, "r") as file:
        for line in file:
            content = json.loads(line)
            examples_list.append(content)
    return examples_list


def retrieve_doc(doc_id: str, corpus_path: Path) -> str:
    with open(corpus_path / doc_id, "r") as file:
        return file.read()


def word_idx2char_idx(text: str, start: int, end: int) -> tuple[int, int]:
    text_split = text.split()
    # Add 1 to start_char_idx to account for the space
    start_char_idx = len(" ".join(text_split[:start]))
    end_char_idx = len(" ".join(text_split[:end]))

    while True:
        if text[start_char_idx] not in {" ", "\n"}:
            break
        start_char_idx += 1

    while True:
        if text[end_char_idx - 1] not in {" ", "\n"}:
            break
        end_char_idx -= 1

    return start_char_idx, end_char_idx


def replace_lrb_rrb(
    text: str,
) -> tuple[str, np.array]:
    """Replace -LRB- and -RRB- with ( and ) respectively. We need to keep track of the changes in the character indices. We therefore return the new text and the a mapping from old character indices to new character indices."""

    character_mapping = np.arange(len(text), dtype=int)

    cleaned_text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    locations_of_lrb_rrb = np.zeros(len(text), dtype=bool)
    lrb_rrb_mentions = re.finditer(r"-LRB-|-RRB-", text)
    for match in lrb_rrb_mentions:
        start, end = match.span()
        locations_of_lrb_rrb[start + 1 : end] = True

    character_mapping -= locations_of_lrb_rrb.cumsum()

    # set to True if the character is part of -LRB- or -RRB-

    return cleaned_text, character_mapping


def fix_double_quotation_marks(text: str) -> tuple[str, np.array]:
    """There are currently spaces before and after quotation marks. This function removes the spaces before and after the quotation marks.
    It returns the new text and the character mapping."""

    def clean_quotes(match):
        # If group 1 has a match (double quotes), use that, otherwise use group 2 (single quotes)
        return (
            f'"{match.group(1).strip()}"'
            if match.group(1) is not None
            else f"'{match.group(2).strip()}'"
        )

    character_mapping = np.arange(len(text), dtype=int)
    locations_of_spaces = np.zeros(len(text), dtype=bool)
    # Pattern for quotes
    # Pattern that matches quoted strings with possible spaces inside quotes
    pattern = r'"[\s](.*?)[\s]"'
    cleaned_text = re.sub(pattern, clean_quotes, text)
    citation_mentions = re.finditer(pattern, text)
    for match in citation_mentions:
        start, end = match.span()
        locations_of_spaces[start + 1] = True
        locations_of_spaces[end - 1] = True

    character_mapping -= locations_of_spaces.cumsum()
    return cleaned_text, character_mapping


def fix_single_quotation_marks(text: str) -> tuple[str, np.array]:
    """There are currently spaces before and after quotation marks. This function removes the spaces before and after the quotation marks.
    It returns the new text and the character mapping."""

    def clean_quotes(match):
        return f"'{match.group(1).strip()}'"

    character_mapping = np.arange(len(text), dtype=int)
    locations_of_spaces = np.zeros(len(text), dtype=bool)
    # Pattern for quotes
    # Pattern that matches quoted strings with possible spaces inside quotes
    pattern = r"'[\s](.*?)[\s]'"
    cleaned_text = re.sub(pattern, clean_quotes, text)
    citation_mentions = re.finditer(pattern, text)
    for match in citation_mentions:
        start, end = match.span()
        locations_of_spaces[start + 1] = True
        locations_of_spaces[end] = True

    character_mapping -= locations_of_spaces.cumsum()
    return cleaned_text, character_mapping


def fix_apostrophes(text: str) -> tuple[str, np.array]:
    """There are currently spaces before and after apostrophes. This function removes the spaces before and after the apostrophes.
    It returns the new text and the character mapping."""

    def clean_apostrophes(match):
        return match.group(0).replace(" ", "")

    character_mapping = np.arange(len(text), dtype=int)
    locations_of_spaces = np.zeros(len(text), dtype=bool)
    # Pattern for quotes
    # Pattern that matches quoted strings with possible spaces inside quotes
    pattern = r"\w\s\'\w"
    cleaned_text = re.sub(pattern, clean_apostrophes, text)
    citation_mentions = re.finditer(pattern, text)
    for match in citation_mentions:
        start, end = match.span()
        locations_of_spaces[start + 1] = True

    character_mapping -= locations_of_spaces.cumsum()
    return cleaned_text, character_mapping


def fix_lonely_apostrophe(text: str) -> tuple[str, np.array]:
    """fix apostrophes that are used to indicate the posession of plural nouns. For example, the text "the students ' books" should be "the students' books"."""

    def clean_apostrophes(match):
        return match.group(0).replace("s ", "s")

    character_mapping = np.arange(len(text), dtype=int)
    locations_of_spaces = np.zeros(len(text), dtype=bool)
    # Pattern for quotes
    # Pattern that matches quoted strings with possible spaces inside quotes
    pattern = r"s\s\'\s"
    cleaned_text = re.sub(pattern, clean_apostrophes, text)
    citation_mentions = re.finditer(pattern, text)
    for match in citation_mentions:
        start, end = match.span()
        locations_of_spaces[start + 1] = True

    character_mapping -= locations_of_spaces.cumsum()
    return cleaned_text, character_mapping


def remove_spaces(text: str) -> tuple[str, np.array]:
    """The text contains too many spaces. There are spaces before and after every punctuation mark and multiple spaces between words. This function removes spaces before a punctuation, and spaces withhin brackets and parantheses.
    We need to keep track of the changes in the character indices. We therefore return the new text and the a mapping from old character indices to new character indices."""
    num_char = len(text)
    character_mapping = np.arange(num_char, dtype=int)

    cleaned_text = ""
    removed_spaces = 0
    for idx, char in enumerate(text):
        if char == " ":
            # remove space if it is the last character
            if idx == num_char - 1:
                removed_spaces += 1
                character_mapping[idx] -= removed_spaces
                continue
            # remove space if the next character is one of the following characters
            if text[idx + 1] in {
                ".",
                ",",
                "!",
                "?",
                ":",
                ";",
                ")",
                "]",
                "}",
                "-",
                "/",
            }:
                character_mapping[idx] -= removed_spaces
                removed_spaces += 1
                continue
            # remove space if the previous character is one of the following characters
            if idx > 0 and (text[idx - 1] in ["(", "[", "{", "-", "/", "`"]):
                character_mapping[idx] -= removed_spaces
                removed_spaces += 1
                continue

        character_mapping[idx] -= removed_spaces
        cleaned_text += char

    return cleaned_text, character_mapping


def format_eraser_text(
    text: str, evidence_spans: list[list[int]]
) -> dict[str, str | list[list[int]]]:
    """This function removes spaces from the text and updates the evidence spans accordingly. It returns a dictionary with the new text and the updated evidence spans."""

    # format text while keeping track of the changes in the character indices
    cleaned_text, character_mapping = replace_lrb_rrb(text)
    cleaned_text, character_mapping_apostrophes = fix_apostrophes(cleaned_text)
    cleaned_text, character_mapping_double_quotes = fix_double_quotation_marks(
        cleaned_text
    )
    cleaned_text, character_mapping_single_quotes = fix_single_quotation_marks(
        cleaned_text
    )
    cleaned_text, character_mapping_lonely_apostrophes = fix_lonely_apostrophe(
        cleaned_text
    )
    cleaned_text, character_mapping_spaces = remove_spaces(cleaned_text)

    # Create one mapping from the original character indices to the new character indices
    for idx in range(len(character_mapping)):
        character_mapping[idx] = character_mapping_spaces[
            character_mapping_lonely_apostrophes[
                character_mapping_single_quotes[
                    character_mapping_double_quotes[
                        character_mapping_apostrophes[character_mapping[idx]]
                    ]
                ]
            ]
        ]

    for idx in range(len(evidence_spans)):
        start = evidence_spans[idx][0]
        end = evidence_spans[idx][1]

        if end > len(text) - 1:
            evidence_spans[idx] = [
                character_mapping[start],
                character_mapping[-1] + 1,
            ]
        else:
            evidence_spans[idx] = [
                character_mapping[start],
                character_mapping[end],
            ]
    return {"context": cleaned_text, "evidence_spans": evidence_spans}


def get_eraser_dataset(annoations_path: Path, corpus_path: Path) -> Dataset:
    if not corpus_path.is_dir():
        raise FileNotFoundError(f"The corpus pathdoes {corpus_path} not exist.")

    if not annoations_path.is_file():
        raise FileNotFoundError(
            f"The annotation path {annoations_path} does not exist."
        )

    dataset = load_dataset("json", data_files=str(annoations_path), split="train")

    # check that each evidence do not have multiple documents
    dataset = dataset.map(
        lambda example: {
            "doc_ids": {evidence[0]["docid"] for evidence in example["evidences"]},
        }
    )
    dataset = dataset.cast_column("annotation_id", Value("string"))

    dataset = dataset.map(
        lambda example: {
            "doc_id": example["evidences"][0][0]["docid"],
        }
    )
    dataset = dataset.map(
        lambda example: {"context": retrieve_doc(example["doc_id"], corpus_path)}
    )
    dataset = dataset.map(
        lambda example: {
            "evidence_spans": [
                word_idx2char_idx(
                    example["context"],
                    evidence["start_token"],
                    evidence["end_token"],
                )
                for evidence in example["evidences"][0]
            ]
        }
    )
    dataset = dataset.map(
        lambda example: format_eraser_text(
            example["context"], example["evidence_spans"]
        ),
    )

    dataset = dataset.rename_column("classification", "label")
    dataset = dataset.remove_columns(["evidences", "docids", "doc_ids", "query_type"])
    return dataset


def load_boolq_dataset(**kwargs) -> Dataset:
    dataset = get_eraser_dataset(
        dataset_lookup["boolq"].annotation_path, dataset_lookup["boolq"].corpus_path
    )
    dataset = dataset.map(lambda example: {"label": int(example["label"] == "True")})
    return dataset


def load_movies_dataset(**kwargs) -> Dataset:
    dataset = get_eraser_dataset(
        dataset_lookup["movies"].annotation_path, dataset_lookup["movies"].corpus_path
    )
    dataset = dataset.map(lambda example: {"label": int(example["label"] == "POS")})
    return dataset


def load_fever_dataset(use_subset: bool = True) -> Dataset:
    dataset = get_eraser_dataset(
        dataset_lookup["fever"].annotation_path, dataset_lookup["fever"].corpus_path
    )
    dataset = dataset.map(
        lambda example: {"label": int(example["label"] == "SUPPORTS")}
    )
    if use_subset:
        subset_ids = pd.read_csv(dataset_lookup["fever"].subset_path, dtype={"id": str})
        dataset = dataset.filter(
            lambda example: example["annotation_id"] in set(subset_ids["id"])
        )
    return dataset


def load_scifact_dataset(**kwargs) -> Dataset:
    """This function loads the SciFact dataset and returns a dataset with the context field containing the evidence. The text seems clean, so we do not need to clean it."""
    annotations = load_jsonl_dataset(dataset_lookup["scifact"].annotation_path)
    corpus = load_jsonl_dataset(dataset_lookup["scifact"].corpus_path)

    # make a dictionary with the docids as keys
    abstract_dict = {str(doc["doc_id"]): doc["abstract"] for doc in corpus}
    # title_dict = {str(doc["doc_id"]): doc["title"] for doc in corpus}

    # remove examples without evidence
    annotations = [example for example in annotations if example["evidence"]]

    # One document per example
    examples = []
    for example in annotations:
        for idx, (doc_id, evidences) in enumerate(example["evidence"].items()):
            label = evidences[0]["label"]  # This is the label for the evidence
            sentence_indices = [
                sentence for evidence in evidences for sentence in evidence["sentences"]
            ]
            abstract_sentence_list = abstract_dict[
                doc_id
            ]  # This is the abstract as a list of sentences
            abstract = " ".join(abstract_sentence_list)
            sentence_character_count = np.array(
                [0] + [len(sentence) + 1 for sentence in abstract_sentence_list]
            ).cumsum()

            abstract_sentence_list[2]
            abstract[sentence_character_count[2] : sentence_character_count[3]]

            evidence_spans = []
            for sentence_index in sentence_indices:
                start = sentence_character_count[sentence_index]
                end = sentence_character_count[sentence_index + 1]
                # Don't include the space at the end of the sentence
                if sentence_index < len(abstract_sentence_list):
                    end -= 1

                evidence_spans.append(
                    [
                        start,
                        end,
                    ]
                )

            examples.append(
                {
                    "claim_id": example["id"],
                    "annotation_id": str(
                        idx
                    ),  # we create a new annotation id for each evidence
                    "doc_id": doc_id,
                    "query": example["claim"],
                    "context": abstract,
                    "evidence_spans": evidence_spans,
                    "label": label,
                }
            )

    dataset = Dataset.from_pandas(pd.DataFrame(data=examples))
    dataset = dataset.map(lambda example: {"label": int(example["label"] == "SUPPORT")})
    return dataset


def load_hatexplain_dataset(use_subset: bool = True) -> Dataset:
    def majority_vote(labels: list[str]) -> int:
        """Returns the majority vote label. If there is no majority, return None."""
        assert len(labels) == 3
        if labels.count("normal") >= 2:
            return 0
        if labels.count("hatespeech") >= 2:
            return 1
        return -1

    def create_safe_2d_array(lists: list[list[int]]) -> np.array:
        """
        Copied from https://github.com/visual-ds/plausible-nlp-explanations/blob/main/experiments/datasets/hatexplain.py.
        Create 2d array from list of lists.

        Avoid the creation of the array when the lists are not of the same
        length.

        Args:
            lists (list of list): List of lists.

        Returns:
            np.array: NumPy array (None if the lists are not of the same
                length).
        """
        lens = {len(list_) for list_ in lists}
        if len(lens) == 1:
            return np.array(lists)
        else:
            return None

    def rationales_to_consesus(rationales: list[list[int]]) -> np.array:
        """Convert the rationales to a consesus rationale."""
        rationales_np = create_safe_2d_array(example["rationales"])
        if rationales_np is None:
            return None
        return rationales_np.mean(axis=0) > 0.5

    def boolean_rational_to_evidence_spans(
        boolean_rationales: np.array, character_count_cumulative: np.array
    ) -> list[list[int]]:
        """Convert boolean rationales to evidence spans."""
        evidence_word_indices = np.where(boolean_rationales)[0]
        evidence_spans = []
        start = None
        for evidence_word_index in evidence_word_indices:
            if start is None:
                start = character_count_cumulative[evidence_word_index]

            if (evidence_word_index + 1) not in evidence_word_indices:
                end = character_count_cumulative[evidence_word_index + 1]

                if evidence_word_index < len(character_count_cumulative) - 2:
                    end -= 1

                evidence_spans.append([start, end])
                start = None

        return evidence_spans

    with open(dataset_lookup["hatexplain"].annotation_path, "r") as file:
        dataset_dict = json.loads(file.read())

    with open(dataset_lookup["hatexplain"].split_path, "r") as file:
        split_dict = json.loads(file.read())

    example_list = []
    for example in dataset_dict.values():
        annotation_id = example["post_id"]
        word_list = example["post_tokens"]  # the text is split into words
        text = " ".join(word_list)

        character_count_cumulative = np.array(
            [0] + [len(word) + 1 for word in word_list]
        ).cumsum()
        labels = [annotator["label"] for annotator in example["annotators"]]
        label = majority_vote(labels)

        if label == -1:
            continue

        rationales = rationales_to_consesus(example["rationales"])

        if rationales is None:
            evidence_spans = None
        else:
            evidence_spans = boolean_rational_to_evidence_spans(
                rationales, character_count_cumulative
            )

        example_list.append(
            {
                "annotation_id": annotation_id,
                "context": text,
                "evidence_spans": evidence_spans,
                "label": label,
            }
        )
    df = pd.DataFrame(data=example_list)

    if use_subset:
        subset_ids = pd.read_csv(dataset_lookup["hatexplain"].subset_path)
        df = df[df["annotation_id"].isin(subset_ids["id"])]

    return Dataset.from_pandas(df[df["annotation_id"].isin(split_dict["test"])])


def load_twitter_dataset(use_subset: bool = True) -> Dataset:
    def get_span(text: str, selected_text: str) -> list[tuple[int, int]]:
        """Get the span of the selected text in the text."""
        match = re.search(re.escape(selected_text), text)
        if match is None:
            return []
        return [match.span()]

    df = pd.read_csv(dataset_lookup["twitter"].annotation_path)
    df = df.rename(
        columns={"text": "context", "sentiment": "label", "textID": "annotation_id"}
    )
    # remove all examples with neutral sentiment
    df = df[df["label"] != "neutral"]
    df["evidence_spans"] = df.apply(
        lambda row: get_span(row["context"], row["selected_text"]),
        axis=1,
    )
    df["label"] = df["label"].replace({"negative": 0, "positive": 1})
    if use_subset:
        subset_ids = pd.read_csv(dataset_lookup["twitter"].subset_path)
        df = df[df["annotation_id"].isin(subset_ids["id"])]
    return Dataset.from_pandas(df)
