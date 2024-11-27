fn test_query_ann() {
    let json_data1 = r#"
    {
        "vector_db_name": "example_db",
        "vector": [0.1, 0.2, 0.3],
        "filter": {
            "genre": { "$in": ["comedy", "documentary", "drama"] }
        },
        "nn_count": 10
    }
    "#;

    let json_data2 = r#"
    {
        "vector_db_name": "example_db",
        "vector": [0.4, 0.5, 0.6],
        "filter": {
            "$or": [
                {
                    "$and": [
                        { "genre": { "$eq": "drama" } },
                        { "year": { "$gte": 2020 } }
                    ]
                },
                {
                    "$and": [
                        { "director": { "$eq": "Christopher Nolan" } },
                        { "rating": { "$gte": 8.0 } }
                    ]
                }
            ]
        },
        "nn_count": 5
    }
    "#;

    let json_data3 = r#"
    {
        "vector_db_name": "example_db",
        "vector": [0.7, 0.8, 0.9],
        "filter": {
            "$and": [
                {
                    "year": { "$gte": 2000 }
                },
                {
                    "year": { "$lte": 2023 }
                },
                {
                    "$or": [
                        { "genre": { "$eq": "sci-fi" } },
                        { "genre": { "$eq": "fantasy" } }
                    ]
                }
            ]
        },
        "nn_count": 15
    }
    "#;

    let json_data4 = r#"
    {
        "vector_db_name": "example_db",
        "vector": [1.0, 1.1, 1.2],
        "filter": {
            "$and": [
                {
                    "runtime": { "$gte": 90 }
                },
                {
                    "$or": [
                        { "genre": { "$in": ["comedy", "drama"] } },
                        { "rating": { "$gte": 8.5 } }
                    ]
                },
                {
                    "release_date": { "$eq": "2023-06-01" }
                }
            ]
        },
        "nn_count": 8
    }
    "#;

    let json_data5 = r#"
    {
        "vector_db_name": "example_db",
        "vector": [1.3, 1.4, 1.5],
        "filter": {
            "$or": [
                {
                    "$and": [
                        { "genre": { "$eq": "thriller" } },
                        { "rating": { "$gte": 7 } }
                    ]
                },
                {
                    "$or": [
                        { "year": { "$lte": 2000 } },
                        {
                            "$and": [
                                { "runtime": { "$gte": 120 } },
                                { "rating": { "$gte": 8 } }
                            ]
                        }
                    ]
                }
            ]
        },
        "nn_count": 12
    }
    "#;

    let ann1: VectorANN = serde_json::from_str(json_data1).unwrap();
    let ann2: VectorANN = serde_json::from_str(json_data2).unwrap();
    let ann3: VectorANN = serde_json::from_str(json_data3).unwrap();
    let ann4: VectorANN = serde_json::from_str(json_data4).unwrap();
    let ann5: VectorANN = serde_json::from_str(json_data5).unwrap();

    println!("{:?}", ann1);
    println!("{:?}", ann2);
    println!("{:?}", ann3);
    println!("{:?}", ann4);
    println!("{:?}", ann5);
}
