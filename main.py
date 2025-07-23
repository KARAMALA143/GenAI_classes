#https://platform.openai.com/settings/organization/api-keys

import openai

import os

api_key = os.getenv()

def chat_with_gpi(prompt): # prompt we privide to gpt to provide repsonse
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    while True:
        user_input = input("you: ")
        if user_input.lower() in ["quit", "exit","bye"]:
            break

        response = chat_with_gpi(user_input)
        print("charbot response: ", response)


#with default latest version of openai gave below error:
'''
You tried to access openai.ChatCompletion, but this is no longer supported in openai>=1.0.0 - see the README at https://github.com/openai/openai-python for the API.

You can run `openai migrate` to automatically upgrade your codebase to use the 1.0.0 interface.

Alternatively, you can pin your installation to the old version, e.g. `pip install openai==0.28`

A detailed migration guide is available here: https://github.com/openai/openai-python/discussions/742
'''