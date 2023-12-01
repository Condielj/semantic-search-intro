import os
import asyncio
import weaviate
import pandas as pd
import weaviate.classes as wvc

WILDCARD = "0"  # Used to represent a restriction that applies to all hs codes


def get_client() -> weaviate.Client:
    """
    Connect to a local Weaviate instance deployed using Docker compose with standard port configurations.

    Parameters:
        None
    Returns
        `weaviate.WeaviateClient`
            The client connected to the local instance
    """
    client = weaviate.connect_to_local(
        port=8080,
        grpc_port=50051,
        headers={
            "X-OpenAI-API-Key": os.getenv("OPENAI_API_KEY"),
        },
    )

    return client


def embed(
    client: weaviate.Client,
    path_to_restriction_data: str = "data.csv",
) -> None:
    """
    Vectorize and embed restriction data into weaviate

    Parameters:
        client: weaviate.Client
            The client connected to the local instance
        path_to_restriction_data: str
            The path to the restriction data csv file.
    Returns
        None
    """
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


def get_filters(code: str) -> wvc.Filter:
    """
    Creates filters in order to get only restrictions with applicable hs codes.
    Example: if the code is 0207, then 0, 02, 020, 0207, and anything that starts with 0207 apply,
            Any restrictions that apply to all hs_codes, represented by WILDCARD, also apply.

    Parameters:
        code: str
            The hs code to create filters for
    Returns
        wvc.Filter
            The filters to apply to the query
    """
    filters = wvc.Filter(path="hs_code").like(
        f"{code}*"
    )  # | wvc.Filter(path="hs_code").equal(WILDCARD) # TODO ENABLE WILDCARD
    built_code = ""
    for c in code:
        built_code += c
        filters = filters | wvc.Filter(path="hs_code").equal(built_code)
    return filters


def get_neighbors(
    item_description: str,
    item_hs_code: str,
    client: weaviate.Client,
    debug: bool = False,
) -> list:
    """
    Returns all of the 'neighbors' of a given item description, filtering on the hs code.

    Parameters:
        item_description: str
            The item description to search for
        item_hs_code: str
            The hs code to filter on
        client: weaviate.Client
            The client connected to the local instance
        debug: bool
            Whether or not to print debug statements
    Returns
        reponse.objects: list
            A list of weaviate objects that are neighbors to the given item description
    """
    restrictions = client.collections.get("Restriction")

    response = restrictions.query.near_text(
        query=item_description,
        filters=get_filters(item_hs_code),
        return_metadata=wvc.MetadataQuery(distance=True),
    )

    if len(response.objects) == 0:
        return []

    if debug:
        print("---")
        print(f"FILTERING ON '{item_hs_code}'")
        for object in response.objects:
            print(f"""IN:({object.properties["hs_code"]})""")
        print("---")

        print("CLOSEST NEIGHBOR:")
        print(response.objects[0].properties)
        print("---")

    return response.objects


def process_row(row, client: weaviate.Client) -> list:
    """
    Gets the neighbors for a row and returns their data formatted as a list of dictionaries.

    Parameters:
        row: pd.Series
            The row to process
        client: weaviate.Client
            The client connected to the local instance
    Returns
        new_rows: list
            A list of dictionaries containing the data for the neighbors of the row
    """
    response_objects = get_neighbors(row["description"], row["hs_code"], client=client)
    new_rows = []
    for object in response_objects:
        o = object.properties
        new_rows.append(
            {
                "hs_code": row["hs_code"],
                "description": row["description"],
                "restricted_codes": o["hs_code"],
                "restricted_item": o["item"],
                "restriction": o["restriction_text"],
                "distance": object.metadata.distance,
            }
        )
    return new_rows


def restrict_from_csv(
    client: weaviate.Client,
    filepath: str,
    encoding: str = "utf8",
) -> pd.DataFrame:
    """
    Gets all neighbors for each item in a csv and formats them into a dataframe.

    Parameters:
        client: weaviate.Client
            The client connected to the local instance
        filepath: str
            The path to the csv file
        encoding: str
            The encoding of the csv file
    Returns
        new_rows: list
            A list of dictionaries containing the data for the neighbors of the row
    """
    queries = pd.read_csv(filepath, encoding=encoding)
    queries["hs_code"] = queries["hs_code"].astype(str)
    queries["hs_code"] = queries["hs_code"].str.replace(".", "")

    columns = [
        "hs_code",
        "description",
        "restricted_codes",
        "restricted_item",
        "restriction",
        "distance",
    ]

    new_rows = []
    for index, row in queries.iterrows():
        if index % 10 == 0:
            print(f"{index}/{len(queries)} rows processed.")
        new_rows.extend(process_row(row, client=client))

    print("hello")

    rdf = pd.DataFrame(new_rows, columns=columns)

    return rdf


if __name__ == "__main__":
    client = get_client()
    # embed(client=client, path_to_restriction_data="data/canada_restrictions.csv")
    # get_neighbors("chicken", "0207", debug=True)

    restrict_from_csv(
        filepath="data/walmart_input.csv", encoding="latin1", client=client
    )
