import openai
import pandas as pd

client = openai.Client()

ROLE_MESSAGE = """
You are a helpful assistant that receives product a prompt that looks like the following:
Item: (ITEM DESCRIPTION)
Restricted:
1. (RESTRICTED ITEM DESCRIPTION 1)
2. (RESTRICTED ITEM DESCRIPTION 2)
and so on.

The RESTRICTED ITEM DESCRIPTIONS that appear in the bulleted list are items that are restricted.  The ITEM DESCRIPTION at the beginning is the item I am trying to import.

You will read all descriptions, and then determine if any of them apply to my item.  You will format your response like so:

Restrictions:(all the line numbers of RESTRICTED ITEM DESCRIPTIONs from the numbered list that you believe apply to my item, remember, more than one can apply.  Separate them with a comma.  Do not be too strict.  If multiple restrictions apply, include them all.  If you have something very general, say 'Dangerous Goods' or 'Items contrary to Bulgarian fashion', make your best guess))

Example input and response:
input:
Item: Steel spectacle frames
Restricted:
1. Spectacle frames.
2. Face protectors for ice hockey and box lacrosse players.
3. Ice hockey helmets.
4. Jewellery.

your response:
Restrictions:1
"""


def format_input(item: str, restrictions: list[str]) -> str:
    formatted = f"""
    Item: {item}
    Restricted:
    """
    for i, restriction in enumerate(restrictions):
        formatted += f"{i+1}. {restriction}\n"
    return formatted


def main(path_to_csv):
    df = pd.read_csv(path_to_csv)


response = client.chat.completions.with_raw_response.create(
    messages=[
        {
            "role": "system",
            "content": ROLE_MESSAGE,
        },
        {
            "role": "user",
            "content": """Item: Steel spectacle frames
                        Restricted: 
                            1. Spectacle frames.
                            2. Face protectors for ice hockey and box lacrosse players.
                            3. Ice hockey helmets.
                            4. Jewellery.""",
        },
    ],
    model="gpt-3.5-turbo",
)

print(response)
