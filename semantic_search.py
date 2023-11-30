import os
import time
import weaviate
import pandas as pd
from format_input import format_input_df


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
    path_to_restriction_data: str = "data.csv",
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
    df = pd.read_csv(path_to_restriction_data)[interesting_columns]
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


def closest_neighbor_query(
    text: str, hs_code: str, client: weaviate.Client = None, vectorizer: str = "openai"
) -> dict:
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
                        "operator": "Like",
                        "valueText": f"{hs_code}*",
                    },
                    {"path": "hs_code", "operator": "Equal", "valueText": "0"},
                ],
            }
        )
        .with_additional(["distance"])
        .with_limit(1)
        .do()
    )

    return response


if __name__ == "__main__":
    vectorizers = ["openai", "cohere"]
    skip_embed = True
    client = get_client()

    if not skip_embed:
        for vectorizer in vectorizers:
            embed(
                path_to_restriction_data="data/canada_restrictions.csv",
                client=client,
                vectorizer=vectorizer,
            )

    queries = pd.read_csv("data/walmart_input.csv", encoding="latin1")
    # Strip the periods out of the hs_codes if any
    queries["hs_code"] = queries["hs_code"].str.replace(".", "")

    for vectorizer in vectorizers:
        response_item = []
        response_restriction = []
        response_distance = []
        response_error = []

        t1 = time.time()
        for index, row in queries.iterrows():
            response = closest_neighbor_query(
                row["description"], row["hs_code"], client=client, vectorizer=vectorizer
            )
            if response is None:
                response_item.append("")
                response_restriction.append("")
                response_distance.append("")
                response_error.append("")
            elif "errors" in response:
                response_item.append("")
                response_restriction.append("")
                response_distance.append("")
                response_error.append(response["errors"][0]["message"])
            else:
                response = response["data"]["Get"][f"Restriction_{vectorizer}"][0]
                response_item.append(response["item"])
                response_restriction.append(response["restriction"])
                response_distance.append(response["_additional"]["distance"])
                response_error.append("")
        t2 = time.time()
        print(f"{vectorizer} took {t2-t1} seconds")

        queries[f"{vectorizer}_response_item"] = response_item
        queries[f"{vectorizer}_response_restriction"] = response_restriction
        queries[f"{vectorizer}_response_distance"] = response_distance
        queries[f"{vectorizer}_response_error"] = response_error

    queries.to_csv("output.csv", index=False)
