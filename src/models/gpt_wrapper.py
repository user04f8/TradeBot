from src.models.gpt import *
from constants import GPT_N_PARALLEL_TIMELINES

class GPTModule:
    def __init__(self, model='gpt-3.5-turbo', settings: SerializerSettings = SerializerSettings()):
        self.model = model
        self.settings = settings
        
        # if init_test_train is not None:
        #     test, train = init_test_train
        #     self.pretrain(test, train)        # DEPRECATED, no performance gain
    
    def __call__(self, inpts, len_out=1, num_samples=GPT_N_PARALLEL_TIMELINES, temp=0.7, alpha=0.95, beta=0.3, basic=True, parallel=True, stock=None, summary=None):
        if not isinstance(inpts, list):
            # Assume single case
            inpts = [inpts]
        
        dummy_test = []
        for i, inpt in enumerate(inpts):
            if not isinstance(inpt, pd.Series):
                inpts[i] = pd.Series(inpt, index=pd.RangeIndex(len(inpt)))
                dummy_test.append(pd.Series(0, index=pd.RangeIndex(len(inpt), len(inpt)+len_out)))  # for output sample indices

        # Create a unique scaler for each series
        scalers = [get_scaler(inpt.values, alpha=alpha, beta=beta, basic=basic) for inpt in inpts]

        # transform input_arrs
        input_arrs = [inpt.values for inpt in inpts]
        transformed_input_arrs = np.array([scaler.transform(input_array) for input_array, scaler in zip(input_arrs, scalers)])
        # serialize input_arrs
        input_strs = [serialize_arr(scaled_input_arr, self.settings) for scaled_input_arr in transformed_input_arrs]
        # Truncate input_arrs to fit the maximum context length
        input_arrs, input_strs = zip(*[truncate_input(input_array, input_str, self.settings, self.model, len_out) for input_array, input_str in zip(input_arrs, input_strs)])
        
        steps = len_out
        samples = None
        medians = None
        completions_list = None
        if num_samples > 0:
            completions_list = []
            complete = lambda x: gpt_completion_fn(model=self.model, input_str=x, steps=steps*STEP_MULTIPLIER, settings=self.settings, num_samples=num_samples, temp=temp, stock=stock, summary=summary)
            if parallel and len(input_strs) > 1:
                print('Running completions in parallel for each input')
                with ThreadPoolExecutor(min(10, len(input_strs))) as p:
                    completions_list = list(tqdm(p.map(complete, input_strs), total=len(input_strs)))
            else:
                completions_list = [complete(input_str) for input_str in tqdm(input_strs)]

            def completion_to_pred(completion, inv_transform): 
                pred = handle_prediction(deserialize_str(completion, self.settings, ignore_last=False, steps=steps), expected_length=steps, strict=False)
                if pred is not None:
                    return inv_transform(pred)
                else:
                    return None
            
            preds = [[completion_to_pred(completion, scaler.inv_transform) for completion in completions] for completions, scaler in zip(completions_list, scalers)]
            # print(preds, completions_list, input_strs)
            
            samples = [pd.DataFrame(preds[i], columns=dummy_test[i].index) for i in range(len(preds))]
            medians = [sample.median(axis=0) for sample in samples]
            samples = samples if len(samples) > 1 else samples[0]
            medians = medians if len(medians) > 1 else medians[0]
        

        return samples, medians