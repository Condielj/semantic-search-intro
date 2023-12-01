import openai

client = openai.Client()

response = client.chat.completions.with_raw_response.create(
    messages=[
        {
            "role": "system",
            "content": "You receive product descriptions and a list of short descriptions of items that have trade restrictions placed on them.  You will need to judge whether or not the restriction should apply to the item given to you based on its description.",
        },
        {
            "role": "user",
            "content": """Item description: Steel spectacle frames
                        Restricted Item descriptions: 
                            1. Spectacle frames.
                            2. Face protectors for ice hockey and box lacrosse players.
                            3. Ice hockey helmets.
                            4. Jewellery.""",
        },
    ],
    model="gpt-3.5-turbo",
)

print(response)
