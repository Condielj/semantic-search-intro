import os
import weaviate
import pandas as pd
import weaviate.classes as wvc


def get_client() -> weaviate.Client:
    client = weaviate.connect_to_local(
        port=8080,
        grpc_port=50051,
        headers={
            "X-OpenAI-API-Key": os.getenv("OPENAI_API_KEY"),
        },
    )

    return client


def embed(
    path_to_restriction_data: str = "data.csv", client: weaviate.Client = None
) -> None:
    if client is None:
        client = get_client()

    # Define class
    client.collections.delete("Restriction")
    restrictions = client.collections.create(
        name="Restriction",
        vectorizer_config=wvc.Configure.Vectorizer.text2vec_openai(),
        properties=[
            wvc.Property(
                name="hs_code",
                data_type=wvc.DataType.TEXT,
                skip_vectorization=True,
            ),
            wvc.Property(name="item", data_type=wvc.DataType.TEXT),
            wvc.Property(
                name="restriction_text",
                data_type=wvc.DataType.TEXT,
                skip_vectorization=True,
            ),
        ],
    )

    # Read in data
    interesting_columns = [
        "hs_code",
        "item",
        "restriction",
    ]
    df = pd.read_csv(path_to_restriction_data)[interesting_columns]
    data = df.to_dict(orient="records")
    restriction_objs = list()

    for d in data:
        restriction_objs.append(
            {
                "hs_code": d["hs_code"],
                "item": d["item"],
                "restriction_text": d["restriction"],
            }
        )

    restrictions = client.collections.get("Restriction")
    restrictions.data.insert_many(restriction_objs)

    return


def get_neighbors(
    item_description: str, item_hs_code: str, client: weaviate.Client = None
) -> dict:
    if client is None:
        client = get_client()

    restrictions = client.collections.get("Restriction")

    response = restrictions.query.near_text(
        query=item_description,
        filters=wvc.Filter(path="hs_code").like(f"{item_hs_code}*")
        | wvc.Filter(path="hs_code").equal("0"),
    )

    print(response.objects[0].properties)
    return response


if __name__ == "__main__":
    # embed(path_to_restriction_data="data/canada_restrictions.csv")
    get_neighbors("chicken", "0207")
