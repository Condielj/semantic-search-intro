import os
import json
import time
import weaviate
import pandas as pd


def get_client() -> weaviate.Client:
    # Initialize client
    client = weaviate.Client(
        "http://localhost:8080",
        additional_headers={
            "X-OpenAI-API-Key": os.getenv("OPENAI_API_KEY"),
            "X-Cohere-API-Key": os.getenv("COHERE_API_KEY"),
        },
    )
    return client


def embed(
    path_to_data: str = "data.csv",
    client: weaviate.Client = None,
    vectorizer: str = "openai",
) -> None:
    if client is None:
        client = get_client()

    # Define class
    class_obj = {
        "class": f"Restriction_{vectorizer}",
        "vectorizer": f"text2vec-{vectorizer}",
        "moduleConfig": {f"text2vec-{vectorizer}": {}},
    }

    try:
        client.schema.create_class(class_obj)
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        if e.status_code == 422:
            print("Class already exists")
        else:
            raise e

    # Read in data
    interesting_columns = [
        "hs_code",
        "full_text",
        "item",
        "restriction",
    ]
    df = pd.read_csv(path_to_data)[interesting_columns]
    data = df.to_dict(orient="records")

    # Add data to Weaviate
    client.batch.configure(batch_size=100)
    with client.batch as batch:
        for i, dp in enumerate(data):
            print(f"Importing {class_obj['class']}: {i+1}/{len(data)}")
            properties = {
                "hs_code": dp["hs_code"],
                "full_text": dp["full_text"],
                "item": dp["item"],
                "restriction": dp["restriction"],
            }
            batch.add_data_object(
                data_object=properties,
                class_name=class_obj["class"],
            )

    return


def query(
    text: str, hs_code: str, client: weaviate.Client = None, vectorizer: str = "openai"
) -> None:
    if client is None:
        client = get_client()

    response = (
        client.query.get(
            f"Restriction_{vectorizer}", ["item", "hs_code", "restriction"]
        )
        .with_near_text({"concepts": [text]})
        .with_where(
            {
                "operator": "Or",
                "operands": [
                    {
                        "path": "hs_code",
                        "operator": "ContainsAll",
                        "valueText": [hs_code],
                    },
                    {"path": "hs_code", "operator": "Equal", "valueText": "'00"},
                ],
            }
        )
        .with_additional(["distance"])
        .do()
    )

    print(json.dumps(response, indent=2))
    return response


if __name__ == "__main__":
    vectorizer = ["openai", "cohere"][0]
    skip_embed = True
    client = get_client()
    if not skip_embed:
        embed(
            path_to_data="data/canada_import_restrictions.csv",
            client=client,
            vectorizer=vectorizer,
        )
    query("bromofluorocarbon", "290319", client=client, vectorizer=vectorizer)
