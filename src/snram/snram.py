

from openai import OpenAI

from secret_constants import OPENAI_API_KEY

client = OpenAI(
    api_key=OPENAI_API_KEY
)



client.chat.completions.create(model='gpt-3.5-turbo', prompt=full_series, max_tokens=4, echo=True, temperature=0.3)


    sys_message = "You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas."
    preprompt = "Please continue the following sequence without producing any additional text. Do not say anything like 'the next terms in the sequence are', just return the numbers. Sequence:\n"
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
                {"role": "system", "content": sys_message},
                {"role": "user", "content": preprompt+input_str+settings.time_sep}
            ],
        max_tokens=10,
        temperature=temp,
        n=1
    )






# Hey you got towards the end of the codebase! that's great :)
# Here's a fun fact:
# Originally this file was called StoNER = Stock/News Evaluation of Relevancy,
#   but I think this refactor to SNRAM is for the best

