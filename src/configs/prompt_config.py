from typing import List, Callable
from dataclasses import dataclass

# Type definitions for clarity
# Updated type definition - return both prompt and character position
PromptGenerator = Callable[..., dict[str, int | str]]
OptionsType = List[str]


@dataclass
class ModelDatasetConfig:
    """Configuration for a specific model-dataset pair"""

    prompt: PromptGenerator
    options: OptionsType


# ======================= HELPER FUNCTION =======================
def create_prompt(
    before_context: str, after_context: str, context: str
) -> dict[str, int | str]:
    """Create a prompt with the context inserted between before and after parts."""
    n_characters_before_context = len(before_context)
    n_characters_after_context = len(after_context)
    prompt = f"{before_context}{context}{after_context}"
    return {
        "prompt": prompt,
        "n_characters_before_context": n_characters_before_context,
        "n_characters_after_context": n_characters_after_context,
    }


# ======================= DEFAULT PROMPT TEMPLATES =======================
def default_hatexplain_prompt(context: str, **kwargs) -> dict[str, int | str]:
    return create_prompt(
        before_context="Classify the passage as hatespeech or not. Answer 'Yes' if the input contains hatespeech, and 'No' otherwise.\n\nPassage: ",
        after_context="\nHatespeech:",
        context=context,
    )


def default_twitter_prompt(context: str, **kwargs) -> dict[str, int | str]:
    return create_prompt(
        before_context="This is an overall sentiment classifier for twitter posts. Is the following sentiment positive? Yes or No?\n\nINPUT: ",
        after_context="\nSENTIMENT:",
        context=context,
    )


def default_movie_prompt(context: str, **kwargs) -> dict[str, int | str]:
    return create_prompt(
        before_context="",
        after_context=". Is the movie review positive? Yes or No? Answer:",
        context=context,
    )


def default_boolq_prompt(context: str, query: str, **kwargs) -> dict[str, int | str]:
    return create_prompt(
        before_context="",
        after_context=f"\nQuestion: {query}\nAnswer:",
        context=context,
    )


def default_fever_prompt(context: str, query: str, **kwargs) -> dict[str, int | str]:
    return create_prompt(
        before_context="",
        after_context=f"\n\n{query}\n\nIs this statement true? Yes or No?",
        context=context,
    )


def default_scifact_prompt(context: str, query: str, **kwargs) -> dict[str, int | str]:
    return create_prompt(
        before_context="",
        after_context=f"\n\n{query}\n\nIs this statement true? Yes or No?",
        context=context,
    )


def llama3_fact_verification_prompt(
    context: str, query: str, **kwargs
) -> dict[str, int | str]:
    return create_prompt(
        before_context="<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 21 Mar 2025\n\nYou are a fact verification chatbot. You answer Yes if a claim is true and No if the claim is false. You verify the claim based on the following text: ",
        after_context=f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        context=context,
    )


def llama3_boolq_prompt(context: str, query: str, **kwargs) -> dict[str, int | str]:
    return create_prompt(
        before_context="<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 21 Mar 2025\n\nYou are a question-answering chatbot. You answer Yes or No based on the following text: ",
        after_context=f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{query}?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        context=context,
    )


def llama3_hatespeech_prompt(context: str, **kwargs) -> dict[str, int | str]:
    return create_prompt(
        before_context="""<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 21 Mar 2025\n\nYou are a hatespeech detection model. You answer Yes if a passage contains hatespeech and No if it does not. It is a fine line between text being hatespeech or offensive. For it to be hatespeech it must attack an individual or group based on a protected attribute. Here are some samples:
        Text: we cannot continue calling ourselves feminists if the rights of all womxn arent addressed yes to a sexual offences public list but will a trans lesbian bisexual and queer womxn be able to enter their information on the reporting sheet gender forum. Answer: No
        Text: <user> i am bit confused coz chinese ppl can not access twitter thn how this ching chong using it i think he pakistani 🤔 🤔 🤔. Answer: Yes
        Text: and this is why i end up with nigger trainee doctors who can not speak properly lack basic knowledge of biology it truly scary if the public only knew. Answer: Yes
        Text: <user> <user> <user> <user> <user> he married a negress because white women did not want him but black females all over america love nobama. Answer: No
        <|eot_id|><|start_header_id|>user<|end_header_id|>\n\nText: """,
        after_context="Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        context=context,
    )


def llama3_sentiment_prompt(context: str, **kwargs) -> dict[str, int | str]:
    return create_prompt(
        before_context="<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 21 Mar 2025\n\nYou are a sentiment analysis model. You answer Yes if a passage has positive sentiment and No if negative. <|eot_id|><|start_header_id|>user<|end_header_id|>\n\nIs the sentiment positive?\n\n",
        after_context="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        context=context,
    )


def qwen2_fact_verification_prompt(
    context: str, query: str, **kwargs
) -> dict[str, int | str]:
    return create_prompt(
        before_context="<|im_start|>system\nYou are a fact verification chatbot. You answer Yes if a claim is true and No if the claim is false. You verify the claim based on the following text: ",
        after_context=f"<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        context=context,
    )


def qwen2_boolq_prompt(context: str, query: str, **kwargs) -> dict[str, int | str]:
    return create_prompt(
        before_context="<|im_start|>system\nYou are a question-answering chatbot. You answer Yes or No based on the following text: ",
        after_context=f"<|im_end|>\n<|im_start|>user\n{query}?<|im_end|>\n<|im_start|>assistant\n",
        context=context,
    )


def qwen2_hatespeech_prompt(context: str, **kwargs) -> dict[str, int | str]:
    return create_prompt(
        before_context="""<|im_start|>system\nYou are a hatespeech detection model. You answer Yes if a passage contains hatespeech and No if it does not. It is a fine line between text being hatespeech or offensive. For it to be hatespeech it must attack an individual or group based on a protected attribute. Here are some samples:
        Text: we cannot continue calling ourselves feminists if the rights of all womxn arent addressed yes to a sexual offences public list but will a trans lesbian bisexual and queer womxn be able to enter their information on the reporting sheet gender forum. Answer: No
        Text: <user> i am bit confused coz chinese ppl can not access twitter thn how this ching chong using it i think he pakistani 🤔 🤔 🤔. Answer: Yes
        Text: and this is why i end up with nigger trainee doctors who can not speak properly lack basic knowledge of biology it truly scary if the public only knew. Answer: Yes
        Text: <user> <user> <user> <user> <user> he married a negress because white women did not want him but black females all over america love nobama. Answer: No
        <|im_end|>\n<|im_start|>user\nText: """,
        after_context="Answer: <|im_end|>\n<|im_start|>assistant\n",
        context=context,
    )


def qwen2_sentiment_prompt(context: str, **kwargs) -> dict[str, int | str]:
    return create_prompt(
        before_context="<|im_start|>system\nYou are a sentiment analysis model. You answer Yes if a passage has positive sentiment and No if negative. <|im_end|>\n<|im_start|>user\nIs the sentiment positive?\n\n",
        after_context="<|im_end|>\n<|im_start|>assistant\n",
        context=context,
    )


def gemma2_fact_verification_prompt(
    context: str, query: str, **kwargs
) -> dict[str, int | str]:
    return create_prompt(
        before_context="<start_of_turn>user\nYou are a fact verification chatbot. You answer Yes if a claim is true and No if the claim is false. You verify the claim based on the following text: ",
        after_context=f"\n{query}<end_of_turn>\n<start_of_turn>model\n",
        context=context,
    )


def gemma2_boolq_prompt(context: str, query: str, **kwargs) -> dict[str, int | str]:
    return create_prompt(
        before_context="<start_of_turn>user\nYou are a question-answering chatbot. You answer Yes or No based on the following text: ",
        after_context=f"\n{query}?<end_of_turn>\n<start_of_turn>model\n",
        context=context,
    )


def gemma2_hatespeech_prompt(context: str, **kwargs) -> dict[str, int | str]:
    return create_prompt(
        before_context="""<start_of_turn>user\nYou are a hatespeech detection model. You answer Yes if a passage contains hatespeech and No if it does not. It is a fine line between text being hatespeech or offensive. For it to be hatespeech it must attack an individual or group based on a protected attribute. Here are some samples:
        Text: we cannot continue calling ourselves feminists if the rights of all womxn arent addressed yes to a sexual offences public list but will a trans lesbian bisexual and queer womxn be able to enter their information on the reporting sheet gender forum. Answer: No
        Text: <user> i am bit confused coz chinese ppl can not access twitter thn how this ching chong using it i think he pakistani 🤔 🤔 🤔. Answer: Yes
        Text: and this is why i end up with nigger trainee doctors who can not speak properly lack basic knowledge of biology it truly scary if the public only knew. Answer: Yes
        Text: <user> <user> <user> <user> <user> he married a negress because white women did not want him but black females all over america love nobama. Answer: No
        \nText: """,
        after_context="Answer: <end_of_turn>\n<start_of_turn>model\n",
        context=context,
    )


def gemma2_sentiment_prompt(context: str, **kwargs) -> dict[str, int | str]:
    return create_prompt(
        before_context="<start_of_turn>user\nYou are a sentiment analysis model. You answer Yes if a passage has positive sentiment and No if negative. \nIs the sentiment positive?\n",
        after_context="<end_of_turn>\n<start_of_turn>model\n",
        context=context,
    )


# ======================= DEFAULT OPTIONS =======================
DEFAULT_OPTIONS = {
    "hatexplain": [" No", " Yes"],
    "twitter": [" No", " Yes"],
    "movie": [" No", " Yes"],
    "boolq": [" No", " Yes"],
    "fever": [" No", " Yes"],
    "scifact": [" No", " Yes"],
}

# ======================= DEFAULT CONFIGURATIONS =======================
DEFAULT_CONFIGS = {
    "hatexplain": ModelDatasetConfig(
        prompt=default_hatexplain_prompt,
        options=DEFAULT_OPTIONS["hatexplain"],
    ),
    "twitter": ModelDatasetConfig(
        prompt=default_twitter_prompt,
        options=DEFAULT_OPTIONS["twitter"],
    ),
    "movie": ModelDatasetConfig(
        prompt=default_movie_prompt, options=DEFAULT_OPTIONS["movie"]
    ),
    "boolq": ModelDatasetConfig(
        prompt=default_boolq_prompt, options=DEFAULT_OPTIONS["boolq"]
    ),
    "fever": ModelDatasetConfig(
        prompt=default_fever_prompt, options=DEFAULT_OPTIONS["fever"]
    ),
    "scifact": ModelDatasetConfig(
        prompt=default_scifact_prompt, options=DEFAULT_OPTIONS["scifact"]
    ),
}

# ======================= MODEL-SPECIFIC CONFIGURATIONS =======================
GPT_NEO_125M_CONFIGS = {
    "hatexplain": ModelDatasetConfig(
        prompt=lambda context, **kwargs: create_prompt(
            before_context="Is the following text hateful? Answer Yes or No only.\n\nText: ",
            after_context="\nAnswer:",
            context=context,
        ),
        options=[" No", " Yes"],
    ),
    "twitter": ModelDatasetConfig(
        prompt=lambda context, **kwargs: create_prompt(
            before_context="Sentiment analysis task:\nText: ",
            after_context="\nIs this positive? Answer:",
            context=context,
        ),
        options=[" No", " Yes"],
    ),
}

LLAMA_IT_CONFIGS = {
    "scifact": ModelDatasetConfig(
        prompt=llama3_fact_verification_prompt, options=["No", "Yes"]
    ),
    "fever": ModelDatasetConfig(
        prompt=llama3_fact_verification_prompt, options=["No", "Yes"]
    ),
    "hatexplain": ModelDatasetConfig(
        prompt=llama3_hatespeech_prompt, options=["No", "Yes"]
    ),
    "boolq": ModelDatasetConfig(prompt=llama3_boolq_prompt, options=["No", "Yes"]),
    "twitter": ModelDatasetConfig(
        prompt=llama3_sentiment_prompt, options=["No", "Yes"]
    ),
    "movie": ModelDatasetConfig(prompt=llama3_sentiment_prompt, options=["No", "Yes"]),
}

QWEN_CONFIGS = {
    "scifact": ModelDatasetConfig(
        prompt=qwen2_fact_verification_prompt, options=["No", "Yes"]
    ),
    "fever": ModelDatasetConfig(
        prompt=qwen2_fact_verification_prompt, options=["No", "Yes"]
    ),
    "hatexplain": ModelDatasetConfig(
        prompt=qwen2_hatespeech_prompt, options=["No", "Yes"]
    ),
    "boolq": ModelDatasetConfig(prompt=qwen2_boolq_prompt, options=["No", "Yes"]),
    "twitter": ModelDatasetConfig(prompt=qwen2_sentiment_prompt, options=["No", "Yes"]),
    "movie": ModelDatasetConfig(prompt=qwen2_sentiment_prompt, options=["No", "Yes"]),
}

GEMMA_CONFIGS = {
    "scifact": ModelDatasetConfig(
        prompt=gemma2_fact_verification_prompt, options=["No", "Yes"]
    ),
    "fever": ModelDatasetConfig(
        prompt=gemma2_fact_verification_prompt, options=["No", "Yes"]
    ),
    "hatexplain": ModelDatasetConfig(
        prompt=gemma2_hatespeech_prompt, options=["No", "Yes"]
    ),
    "boolq": ModelDatasetConfig(prompt=gemma2_boolq_prompt, options=["No", "Yes"]),
    "twitter": ModelDatasetConfig(
        prompt=gemma2_sentiment_prompt, options=["No", "Yes"]
    ),
    "movie": ModelDatasetConfig(prompt=gemma2_sentiment_prompt, options=["No", "Yes"]),
}


MODEL_CONFIGS = {
    "qwen2-0.5b-it": QWEN_CONFIGS,
    "qwen2-1.5b-it": QWEN_CONFIGS,
    "qwen2-3b-it": QWEN_CONFIGS,
    "llama3-1b-it": LLAMA_IT_CONFIGS,
    "llama3-3b-it": LLAMA_IT_CONFIGS,
    "llama3-8b-it": LLAMA_IT_CONFIGS,
    "gemma2-2b-it": GEMMA_CONFIGS,
    "gemma2-9b-it": GEMMA_CONFIGS,
}


# ======================= HELPER FUNCTIONS =======================
def get_prompt_generator(
    model_name: str, dataset_name: str, default_prompt: PromptGenerator
) -> PromptGenerator:
    if model_name in MODEL_CONFIGS and dataset_name in MODEL_CONFIGS[model_name]:
        print("model_name", model_name)
        print("dataset_name", dataset_name)
        model_config = MODEL_CONFIGS[model_name][dataset_name]
        if model_config.prompt is not None:
            return model_config.prompt
    if dataset_name in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[dataset_name].prompt
    return default_prompt


def get_options(
    model_name: str, dataset_name: str, default_options: List[str]
) -> List[str]:
    if model_name in MODEL_CONFIGS and dataset_name in MODEL_CONFIGS[model_name]:
        model_config = MODEL_CONFIGS[model_name][dataset_name]
        if model_config.options is not None:
            return model_config.options
    if dataset_name in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[dataset_name].options
    return default_options
