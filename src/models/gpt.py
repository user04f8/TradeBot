from src.data.llm_serializer import serialize_arr, SerializerSettings, deserialize_str, vec_repr2num
from constants import GPT_MAX_CONTEXT_LEN

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
from jax import grad, vmap
import pandas as pd
import tiktoken
from tqdm import tqdm
from typing import Callable, List

from openai import OpenAI

from secret_constants import OPENAI_API_KEY

client = OpenAI(
    api_key=OPENAI_API_KEY
)

# This leverages the tokenization approach from "Large Language Models Are Zero-Shot Time Series Forecasters" https://arxiv.org/abs/2310.07820

def tokenize_fn(str, model) -> List[int]:
    return tiktoken.encoding_for_model(model).encode(str)

def get_allowed_ids(strs: List[str], model: str) -> List[int]:
    encoding = tiktoken.encoding_for_model(model)
    ids = []
    for s in strs:
        id = encoding.encode(s)
        ids.extend(id)
    return ids

def gpt_completion_fn(model: str, input_str: str, steps: int, settings: SerializerSettings, num_samples: int, temp: float, stock: str = None, summary: str = None) -> List[str]:
    """
    Generate text completions from GPT using OpenAI's API.
    """
    avg_tokens_per_step = len(tokenize_fn(input_str, model)) / len(input_str.split(settings.time_sep))
    # define logit bias to prevent GPT-3 from producing unwanted tokens
    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)] 
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign]
    allowed_tokens = [t for t in allowed_tokens if len(t) > 0] # remove empty tokens like an implicit plus sign
    logit_bias = {id: 30 for id in get_allowed_ids(allowed_tokens, model)}

    if model in ['gpt-3.5-turbo','gpt-4']:
        sys_message = "You are a helpful assistant that performs time series predictions of stock prices, informed by news summaries. The user will provide a news summary and a sequence and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas."
        if stock is not None:
            preprompt = f"Here's a summary of recent news about {stock}: {summary} \n\n"
        else:
            preprompt = ""
        preprompt += "Please continue the following sequence without producing any additional text. Do not say anything like 'the next terms in the sequence are', just return the numbers. Sequence:\n"
        response = client.chat.completions.create(
            model=model,
            messages=[
                    {"role": "system", "content": sys_message},
                    {"role": "user", "content": preprompt+input_str+settings.time_sep}
                ],
            max_tokens=int(avg_tokens_per_step*steps),
            temperature=temp,
            logit_bias=logit_bias,
            n=num_samples,
        )
        return [choice.message.content for choice in response.choices]
    else:
        response = client.chat.completions.create(
            model=model,
            prompt=input_str, 
            max_tokens=int(avg_tokens_per_step*steps), 
            temperature=temp,
            logit_bias=logit_bias,
            n=num_samples
        )
        return [choice.text for choice in response.choices]
    
def gpt_nll_fn(model: str, input_arr, target_arr, settings: SerializerSettings, transform: Callable, count_seps=True, temp=1):
    """
    Calculate the Negative Log-Likelihood (NLL) per dimension of the target array according to the LLM.
    """
    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    assert input_str.endswith(settings.time_sep), f'Input string must end with {settings.time_sep}, got {input_str}'
    full_series = input_str + target_str
    response = client.chat.completions.create(model=model, prompt=full_series, logprobs=5, max_tokens=0, echo=True, temperature=temp)
    
    #print(response['choices'][0])

    logprobs = np.array(response['choices'][0].logprobs.token_logprobs, dtype=np.float32)
    tokens = np.array(response['choices'][0].logprobs.tokens)
    top5logprobs = response['choices'][0].logprobs.top_logprobs
    seps = tokens==settings.time_sep
    target_start = np.argmax(np.cumsum(seps)==len(input_arr)) + 1
    logprobs = logprobs[target_start:]
    tokens = tokens[target_start:]
    top5logprobs = top5logprobs[target_start:]
    seps = tokens==settings.time_sep
    assert len(logprobs[seps]) == len(target_arr), f'There should be one separator per target. Got {len(logprobs[seps])} separators and {len(target_arr)} targets.'

    # adjust logprobs by removing extraneous and renormalizing
    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)] 
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign, settings.bit_sep+settings.decimal_point]
    allowed_tokens = {t for t in allowed_tokens if len(t) > 0}
    p_extra = np.array([sum(np.exp(ll) for k,ll in top5logprobs[i].items() if not (k in allowed_tokens)) for i in range(len(top5logprobs))])
    if settings.bit_sep == '':
        p_extra = 0
    adjusted_logprobs = logprobs - np.log(1-p_extra)
    digits_bits = -adjusted_logprobs[~seps].sum()
    seps_bits = -adjusted_logprobs[seps].sum()
    BPD = digits_bits/len(target_arr)
    if count_seps:
        BPD += seps_bits/len(target_arr)

    # log p(x) = log p(token) - log bin_width = log p(token) + prec * log base
    transformed_nll = BPD - settings.prec*np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    return transformed_nll-avg_logdet_dydx

STEP_MULTIPLIER = 1.2

@dataclass
class Scaler:
    """
    Represents a data scaler with transformation and inverse transformation functions.

    Attributes:
        transform (callable): Function to apply transformation.
        inv_transform (callable): Function to apply inverse transformation.
    """
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x    

def get_scaler(history, alpha=0.95, beta=0.3, basic=True):
    """
    Generate a Scaler object based on given history data.

    Args:
        history (array-like): Data to derive scaling from.
        alpha (float, optional): Quantile for scaling. Defaults to .95.
        # Truncate inputs
        tokens = [tokeniz]
        beta (float, optional): Shift parameter. Defaults to .3.
        basic (bool, optional): If True, no shift is applied, and scaling by values below 0.01 is avoided. Defaults to False.

    Returns:
        Scaler: Configured scaler object.
    """
    history = history[~np.isnan(history)]
    if basic:
        q = np.maximum(np.quantile(np.abs(history), alpha),.01)
        def transform(x):
            return x / q
        def inv_transform(x):
            return x * q
    else:
        min_ = np.min(history) - beta*(np.max(history)-np.min(history))
        q = np.quantile(history-min_, alpha)
        if q == 0:
            q = 1
        def transform(x):
            return (x - min_) / q
        def inv_transform(x):
            return x * q + min_
    return Scaler(transform=transform, inv_transform=inv_transform)

def truncate_input(input_arr, input_str, settings, model, steps):
    """
    Truncate inputs to the maximum context length for a given model.
    
    Args:
        input (array-like): input time series.
        input_str (str): serialized input time series.
        settings (SerializerSettings): Serialization settings.
        model (str): Name of the LLM model to use.
        steps (int): Number of steps to predict.
    Returns:
        tuple: Tuple containing:
            - input (array-like): Truncated input time series.
            - input_str (str): Truncated serialized input time series.
    """
    context_length = GPT_MAX_CONTEXT_LEN
    input_str_chuncks = input_str.split(settings.time_sep)
    for i in range(len(input_str_chuncks) - 1):
        truncated_input_str = settings.time_sep.join(input_str_chuncks[i:])
        # add separator if not already present
        if not truncated_input_str.endswith(settings.time_sep):
            truncated_input_str += settings.time_sep
        input_tokens = tokenize_fn(truncated_input_str, model=model)
        num_input_tokens = len(input_tokens)
        avg_token_length = num_input_tokens / (len(input_str_chuncks) - i)
        num_output_tokens = avg_token_length * steps * STEP_MULTIPLIER
        if num_input_tokens + num_output_tokens <= context_length:
            truncated_input_arr = input_arr[i:]
            break
    if i > 0:
        print(f'Warning: Truncated input from {len(input_arr)} to {len(truncated_input_arr)}')
    return truncated_input_arr, truncated_input_str

def handle_prediction(pred, expected_length, strict=False):
    """
    Process the output from LLM after deserialization, which may be too long or too short, or None if deserialization failed on the first prediction step.

    Args:
        pred (array-like or None): The predicted values. None indicates deserialization failed.
        expected_length (int): Expected length of the prediction.
        strict (bool, optional): If True, returns None for invalid predictions. Defaults to False.

    Returns:
        array-like: Processed prediction.
    """
    if pred is None:
        return None
    else:
        if len(pred) < expected_length:
            if strict:
                print(f'Warning: Prediction too short {len(pred)} < {expected_length}, returning None')
                return None
            else:
                print(f'Warning: Prediction too short {len(pred)} < {expected_length}, padded with last value')
                return np.concatenate([pred, np.full(expected_length - len(pred), pred[-1])])
        else:
            return pred[:expected_length]
   

def run_gpt(train, test, model='gpt-3.5-turbo', settings: SerializerSettings = SerializerSettings(), num_samples=10, temp=0.7, alpha=0.95, beta=0.3, basic=True, parallel=True, stock=None, summary=None):
    if not isinstance(train, list):
        # Assume single train/test case
        train = [train]
        test = [test]
    
    for i, train_i in enumerate(train):
        if not isinstance(train_i, pd.Series):
            train_i = pd.Series(train_i, index=pd.RangeIndex(len(train_i)))
            test[i] = pd.Series(test[i], index=pd.RangeIndex(len(train_i), len(test[i])+len(train_i)))

    test_len = len(test[0])
    assert all(len(t)==test_len for t in test), f'All test series must have same length, got {[len(t) for t in test]}'

    # Create a unique scaler for each series
    scalers = [get_scaler(train[i].values, alpha=alpha, beta=beta, basic=basic) for i in range(len(train))]

    # transform input_arrs
    input_arrs = [train[i].values for i in range(len(train))]
    transformed_input_arrs = np.array([scaler.transform(input_array) for input_array, scaler in zip(input_arrs, scalers)])
    # serialize input_arrs
    input_strs = [serialize_arr(scaled_input_arr, settings) for scaled_input_arr in transformed_input_arrs]
    # Truncate input_arrs to fit the maximum context length
    input_arrs, input_strs = zip(*[truncate_input(input_array, input_str, settings, model, test_len) for input_array, input_str in zip(input_arrs, input_strs)])
    
    steps = test_len
    samples = None
    medians = None
    completions_list = None
    if num_samples > 0:
        completions_list = []
        complete = lambda x: gpt_completion_fn(model=model, input_str=x, steps=steps*STEP_MULTIPLIER, settings=settings, num_samples=num_samples, temp=temp, stock=stock, summary=summary)
        if parallel and len(input_strs) > 1:
            print('Running completions in parallel for each input')
            with ThreadPoolExecutor(min(10, len(input_strs))) as p:
                completions_list = list(tqdm(p.map(complete, input_strs), total=len(input_strs)))
        else:
            completions_list = [complete(input_str) for input_str in tqdm(input_strs)]

        def completion_to_pred(completion, inv_transform): 
            pred = handle_prediction(deserialize_str(completion, settings, ignore_last=False, steps=steps), expected_length=steps, strict=False)
            if pred is not None:
                return inv_transform(pred)
            else:
                return None
        
        preds = [[completion_to_pred(completion, scaler.inv_transform) for completion in completions] for completions, scaler in zip(completions_list, scalers)]
        # print(preds, completions_list, input_strs)

        samples = [pd.DataFrame(preds[i], columns=test[i].index) for i in range(len(preds))]
        medians = [sample.median(axis=0) for sample in samples]
        samples = samples if len(samples) > 1 else samples[0]
        medians = medians if len(medians) > 1 else medians[0]
    out_dict = {
        'samples': samples,
        'median':  medians,
        'info': {
            'Method': model,
        },
        'completions_list': completions_list,
        'input_strs': input_strs,
    }

    return out_dict
    
if __name__ == '__main__':
    # basic sanity test
    out = run_gpt(np.array([1,2,3,4,5], dtype=np.float64), np.array([3, 4, 5, 6, 7], dtype=np.float64))
    print(out)
    