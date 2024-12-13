import azure.functions as func
import logging
import json
import os
import pyodbc
from openai import OpenAI
import requests
import pandas as pd

def main(req: func.HttpRequest) -> func.HttpResponse:

    # # Load connection string from environment variable
    connection_string = os.getenv("DATABASE_CONNECTION_STRING")
    ai_key = os.getenv("OPENAI")
    
    if not connection_string or not ai_key:

        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": "Server configuration is missing. Please check environment variables."
            }),
            status_code=500,
            mimetype="application/json"
        )

    try:
        # Test the database connection
        conn, test_connection_response = test_db_connection(connection_string)
        
        if conn is None:
            response = {
                "status": "error",
                "message": "Sorry! My database is waking up from a nap. Can you try again in a minute or so?",
                "data": None
            }
            return func.HttpResponse(
                json.dumps(response),
                status_code=200,
                mimetype="application/json"
            )

        client = OpenAI(api_key=ai_key)

        # Process the request body
        req_body = req.get_json()
        user_query = req_body.get('query')

        if test_connection_response == "Database connection successful.":
            if user_query:
                # Process query and return a DataFrame
                result = process_query(user_query, conn, client)

                if result["status"] == "success":
                    # Serialize the DataFrame to JSON
                                                           
                    data_json = result["data"]

                    response = {
                        "status": result["status"],
                        "message": result["message"],
                        "title": result.get("title", "Results"),  # Use the title from the result or a default value
                        "data": json.loads(data_json),  # Ensure JSON compatibility
                    }
                    return func.HttpResponse(
                        json.dumps(response),
                        status_code=200,
                        mimetype="application/json"
                    )
                
                if result["status"] == "error":
                    data_json = result["data"]
                    response = {
                        "status": result["status"],
                        "message": result["message"],
                        "data": None
                    }
                    return func.HttpResponse(
                        json.dumps(response),
                        status_code=200,
                        mimetype="application/json"
                    )

                else:
                    return func.HttpResponse(
                        json.dumps(result),
                        status_code=400,
                        mimetype="application/json"
                    )
            else:
                return func.HttpResponse(
                    json.dumps({"result": "No query received."}),
                    status_code=400,
                    mimetype="application/json"
                )
        else:
            return func.HttpResponse(
                json.dumps({"result": "Database connection failure."}),
                status_code=400,
                mimetype="application/json"
            )
    except Exception as e:
        response = {
                "status": "error",
                "message": e,
                "data": None
            }
        return func.HttpResponse(
            json.dumps(response),
            status_code=200,
            mimetype="application/json"
        )


def test_db_connection(connection_string):
    """
    Tests the databaase connection to see if it is online, also will resume if paused.
    
    Args:
        connection_string: Used for establishing the connection to the SQL server
    
    Returns:
        conn: Connection to database and a success message
    """
    try:
        conn = pyodbc.connect(connection_string, timeout=5)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        
        return conn, "Database connection successful."
    
    except Exception as e:
        return None, f"Database connection failed: {str(e)}"

def validate_query_with_llm(query, client):
    """
    Validates a user query to ensure it meets the app's requirements for ACS data analysis.
    
    Args:
        query (str): The user's query.
        client: The OpenAI client instance.
    
    Returns:
        dict: A JSON response from the LLM with validation status and message.
    """
    # Define the prompt
    prompt = f"""
    You are an assistant designed to validate user queries for an American Community Survey (ACS) data analysis app.
    Your task is to:
     1. Determine if the query can be answered using ACS data. Here is a description of the possible ACS categories. Education: Refers to information about the educational attainment of individuals, such as the highest level of education completed (e.g., high school diploma, bachelor’s degree, or advanced degree). Includes data on school enrollment, literacy rates, and fields of study.
 Disability: Relates to data on individuals with disabilities, including types of disabilities (e.g., visual, hearing, cognitive, mobility), their prevalence, and the impact on daily activities or employment.
 Population: Covers the total count of people living in a geographic area, including demographics such as age, gender, and population density. This is a fundamental measure used for understanding the size and structure of communities.
 Transportation: Includes data on how people commute to work (e.g., car, public transit, walking), average travel times, vehicle ownership, and access to transportation services. Useful for urban planning and infrastructure development.
 Household and Family: Focuses on the composition and characteristics of households, including family size, types of households (e.g., single-parent, multi-generational), and housing arrangements (e.g., rented vs. owned homes).
 Geographical Mobility: Refers to data on the movement of people between different geographic locations, including migration patterns, residence one year ago, and reasons for moving (e.g., job relocation, housing needs).
 Race and Ancestry: Includes information on racial and ethnic groups, self-reported ancestry, and heritage. Provides insights into the diversity of communities and cultural backgrounds.
 Employment: Relates to data on work-related information such as employment status, industry, occupation, work hours, and labor force participation. Often paired with income and poverty data to assess economic well-being.
 Marriage and Birth: Covers data on marital status (e.g., single, married, divorced), number of births within a specific time frame, and family formation trends. Helps analyze population growth and family dynamics.
 Language: Includes information on languages spoken at home, English proficiency, and linguistic diversity. Often used to assess language needs for public services or educational programs.
 Poverty: Refers to data on individuals and families living below the poverty line, including poverty rates, income thresholds, and economic hardship indicators. Crucial for understanding and addressing inequality.
 Income: Focuses on data about earnings from wages, salaries, investments, and other sources. Includes measures like median household income and income distribution across different demographics.
 Age: Refers to data on the age distribution of the population, including age groups (e.g., children, working-age adults, seniors). Provides insights into generational dynamics and demographic trends.
    2. Check if the query contains appropriate and valid content for ACS data analysis.
    3. Ensure the query includes at least one valid location. A valid location must specify a state (e.g., state name or abbreviation). 

    Instructions:
        - Return the output as a JSON object only.
        - Do not include any formatting markers such as ```json, or comments.
        - The response must strictly adhere to the following JSON formats:

    If the query passes all checks, respond with a JSON object like this:
    {{
        "status": "pass",
        "message": "The query is valid."
    }}

    If the query is ambiguous but can be refined, respond with:
    {{
        "status": "refine",
        "message": "Reason the query is ambiguous and instructions to refine it."
    }}

    If the query fails all checks, respond with:
    {{
        "status": "fail",
        "message": "Reason the query is invalid and why it cannot be answered by ACS data."
    }}

    Example Queries:
    - "What is the population of Michigan?" -> Pass.
    - "What is the unemployment rate in Detroit, MI?" -> Pass.
    - "How many people work in tech?" -> Fail (No location provided).
    - "Tell me about trees in Michigan." -> Fail (Not an ACS topic).

    User Query: "{query}"
    """

    try:
        # Call the LLM to validate the query
        validation_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an intelligent query validator for ACS data."},
                {"role": "user", "content": prompt}
            ],
        )
        validation_result = json.loads(validation_response.choices[0].message.content)

        # If validation passes, return the result
        if validation_result["status"] == "pass":
            return validation_result

        # If validation fails but can be refined, attempt refinement
        if validation_result["status"] == "refine":
            refinement_prompt = f"""
            The following query failed validation for ACS data: "{query}"
            Reason: "{validation_result['message']}"
            
            Your task is to refine the query to make it valid for ACS data analysis.
            Instructions:
            - Return the output as a JSON object only.
            - Do not include any formatting markers such as ```json, or comments.
            - The response must strictly adhere to this format:
            {{
                "status": "refined",
                "query": "Refined query here"
            }}
            """
            try:
                refinement_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an assistant designed to refine invalid queries for ACS data."},
                        {"role": "user", "content": refinement_prompt}
                    ],
                )
                refined_query_result = json.loads(refinement_response.choices[0].message.content)
                return {
                    "status": "refined",
                    "original_query": query,
                    "refined_query": refined_query_result.get("query", "No refined query available."),
                    "message": "The query was refined to make it valid for ACS data analysis."
                }
            except Exception as refinement_error:
                return {
                    "status": "fail",
                    "message": "Validation failed, and refinement could not be performed."
                }

        # If validation fails completely, return the failure message
        if validation_result["status"] == "fail":
            return {
                "status": "fail",
                "message": validation_result["message"]
            }

    except json.JSONDecodeError as json_error:
        return {
            "status": "fail",
            "message": "There was an error parsing the validation response. Please try again later."
        }

    except Exception as e:
        # Handle unexpected exceptions
        return {
            "status": "fail",
            "message": f"An unexpected error occurred: {e}"
        }


def classify_query_with_llm(query, client):

    try:
        # OpenAI prompt for query classification
        prompt = f"""
        You are an assistant trained to classify user queries related to the American Community Survey (ACS).
        Analyze the query below and extract:
        1. Intent (e.g., single metric, comparison, or multi-dimensional analysis).
        2. Keywords (relevant terms or concepts from the query). Keywords should not include any location information.
        3. High-level categories (Education, Income, Race, etc.) most relevant to the query. Here is a description of the possible categories. Education: Refers to information about the educational attainment of individuals, such as the highest level of education completed (e.g., high school diploma, bachelor’s degree, or advanced degree). Includes data on school enrollment, literacy rates, and fields of study.
        Disability: Relates to data on individuals with disabilities, including types of disabilities (e.g., visual, hearing, cognitive, mobility), their prevalence, and the impact on daily activities or employment.
        Population: Covers the total count of people living in a geographic area, including demographics such as age, gender, and population density. This is a fundamental measure used for understanding the size and structure of communities.
        Transportation: Includes data on how people commute to work (e.g., car, public transit, walking), average travel times, vehicle ownership, and access to transportation services. Useful for urban planning and infrastructure development.
        Household and Family: Focuses on the composition and characteristics of households, including family size, types of households (e.g., single-parent, multi-generational), and housing arrangements (e.g., rented vs. owned homes).
        Geographical Mobility: Refers to data on the movement of people between different geographic locations, including migration patterns, residence one year ago, and reasons for moving (e.g., job relocation, housing needs).
        Race and Ancestry: Includes information on racial and ethnic groups, self-reported ancestry, and heritage. Provides insights into the diversity of communities and cultural backgrounds.
        Employment: Relates to data on work-related information such as employment status, industry, occupation, work hours, and labor force participation. Often paired with income and poverty data to assess economic well-being.
        Marriage and Birth: Covers data on marital status (e.g., single, married, divorced), number of births within a specific time frame, and family formation trends. Helps analyze population growth and family dynamics.
        Language: Includes information on languages spoken at home, English proficiency, and linguistic diversity. Often used to assess language needs for public services or educational programs.
        Poverty: Refers to data on individuals and families living below the poverty line, including poverty rates, income thresholds, and economic hardship indicators. Crucial for understanding and addressing inequality.
        Income: Focuses on data about earnings from wages, salaries, investments, and other sources. Includes measures like median household income and income distribution across different demographics.
        Age: Refers to data on the age distribution of the population, including age groups (e.g., children, working-age adults, seniors). Provides insights into generational dynamics and demographic trends.
        4. Locations explicitly or implicitly mentioned in the query, categorized as "city", "county", or "state". 

        Query: "{query}"

        Special instructions for locations:
        - Extract only one entry for each location explicitly mentioned in the query. Avoid duplicating geographic units (e.g., "Detroit, Michigan" should not result in separate entries for "Detroit" and "Michigan").
        - If multiple locations are mentioned (e.g., "Los Angeles, California and New York City, New York"), ensure each distinct location is listed only once.
        - Each location should include its name and type ("city", "county", or "state").
        - County locations should include the word county or County to indicate that a county location is being requested.

        Instructions:
        - Return the output as a JSON object only.
        - Do not include any formatting markers such as ```json, or comments.
        - The response must strictly adhere to the following JSON format:

        {{
            "intent": "<user's intent>",
            "keywords": ["<keyword1>", "<keyword2>", ...],
            "categories": ["<category1>", "<category2>", ...],
            "locations": [
                {{"name": "<location name>", "type": "<city/county/state>", "state": "state"}},
                {{"name": "<location name>", "type": "<city/county/state>", "state": "state"}}
            ]
        }}
        """
        # Call OpenAI's ChatCompletion API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Parse response
        llm_output = response.choices[0].message.content
        
        return json.loads(llm_output)
        

    except Exception as e:
        return {"error": f"Failed to process query with LLM: {str(e)}"}

def get_location_codes(locations, conn):
    """
    Given a JSON list of locations, returns a list of tuples with the location type,
    PLACEFP (or COUNTYFP), and STATEFP codes for each location.

    Handles cases where only the state is mentioned (e.g., "MI" or "Michigan").
    
    Args:
        locations (list): A list of JSON dictionaries containing "name", "type", and optionally "state".
        conn (pyodbc.Connection): Connection object for the SQL database.

    Returns:
        list: A list of tuples in the format (location_type, PLACEFP/COUNTYFP, STATEFP).
    """
    results = []

    for location in locations:
        location_name = location["name"].lower().strip()
        location_type = location["type"].lower()
        state_name = location.get("state", "").lower().strip()

        try:
            # Prepare SQL query based on location type
            if location_type == "city":
                query = """
                SELECT TOP 1 PLACEFP, STATEFP
                FROM geo
                WHERE LOWER(PLACENAME) = LOWER(?) AND 
                      (LOWER(STATENAME) = LOWER(?) OR LOWER(STATE) = LOWER(?))
                """
                params = (location_name, state_name, state_name)
            elif location_type == "county":                             
                query = """
                SELECT TOP 1 COUNTYFP, STATEFP
                FROM geo
                WHERE LOWER(COUNTYNAME) = LOWER(?) AND 
                      (LOWER(STATENAME) = LOWER(?) OR LOWER(STATE) = LOWER(?))
                """
                params = (location_name, state_name, state_name)
            elif location_type == "state":
                # State query matches either state name or abbreviation
                query = """
                SELECT DISTINCT STATEFP
                FROM geo
                WHERE LOWER(STATENAME) = LOWER(?) OR LOWER(STATE) = LOWER(?)
                """
                params = (location_name, location_name)
            else:
                raise ValueError(f"Unsupported location type: {location_type}")

            # Execute query
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()

            # Process result based on location type
            if row:
                if location_type == "city":
                    results.append((location_type, row.PLACEFP, row.STATEFP))
                elif location_type == "county":
                    results.append((location_type, row.COUNTYFP, row.STATEFP))
                elif location_type == "state":
                    results.append((location_type, None, row.STATEFP))
            else:
                # Handle case where no match is found
                results.append((location_type, None, None))
        except Exception as e:
            #print(f"Error processing location '{location_name}': {e}")
            results.append((location_type, None, None))

    return results

def get_categories(user_query, client):
    """
    Calls an LLM to determine applicable categories for answering the user's query.

    Args:
        user_query (str): The user's query.

    Returns:
        dict: Contains the selected categories and reasoning.
    """
    prompt = f"""
    You are an assistant trained to analyze user queries related to the American Community Survey (ACS) and select the most applicable high-level categories.

    Your goal is to evaluate the user’s query and identify the categories of data that are most relevant to answering the query. The available categories and their descriptions are:

    1. **Education**: Refers to educational attainment, school enrollment, and literacy rates.
    2. **Disability**: Includes data on individuals with disabilities and their impact on daily life or employment.
    3. **Population**: Covers total population counts, age, gender, and density.
    4. **Transportation**: Includes commuting patterns, travel times, and vehicle ownership.
    5. **Household and Family**: Focuses on family size, household types, and housing arrangements.
    6. **Geographical Mobility**: Refers to movement between geographic locations and migration patterns.
    7. **Race and Ancestry**: Includes racial and ethnic groups, ancestry, and cultural heritage.
    8. **Employment**: Relates to work-related information such as employment status, industries, and occupations.
    9. **Marriage and Birth**: Covers marital status, births, and family formation trends.
    10. **Language**: Includes data on languages spoken at home and English proficiency.
    11. **Poverty**: Refers to data on individuals living below the poverty line and economic hardship indicators.
    12. **Income**: Focuses on earnings, household income, and income inequality.
    13. **Age**: Refers to the age distribution of the population. Includes race and gender information.

    ### Task:
    Analyze the following user query and return the most applicable categories (one or more) that are likely to provide the data needed to answer the query.

    ### User Query:
    "{user_query}"

    Instructions:
        - Return the output as a JSON object only.
        - Do not include any formatting markers such as ```json, or comments.
        - The response must strictly adhere to the following JSON format:

    {{
        "categories": ["<category1>", "<category2>", ...],
        "reasoning": "<brief explanation of why these categories were selected>"
    }}
    """
    
    # Call the OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant specialized in ACS data analysis."},
                {"role": "user", "content": prompt}
            ]
        )
        llm_output = response.choices[0].message.content
        return json.loads(llm_output)
    
    except Exception as e:
        return {"error": str(e)}


def get_combined_keywords(user_keywords, conn):
    """
    Combines user keywords with their synonyms from the SQL table and removes duplicates.

    Args:
        user_keywords (list): List of user-provided keywords.
        conn: Active SQL connection object.

    Returns:
        list: Combined list of keywords including synonyms, with duplicates removed.
    """
    combined_keywords = set(user_keywords)  # Start with user keywords, using a set to handle duplicates

    try:
        # Query to fetch all rows from the `dict` table
        query = "SELECT keyword, synonyms FROM dict"
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        # Convert the SQL result to a dictionary for easier processing
        synonym_dict = {row[0]: row[1] for row in rows}

        # Iterate over the user keywords
        for user_keyword in user_keywords:
            # Check for a direct match in the `keyword` column
            if user_keyword in synonym_dict:
                synonyms = synonym_dict[user_keyword]
                if synonyms:
                    combined_keywords.update(synonyms.split(", "))

            # Check if the user keyword exists in the `synonyms` column
            for keyword, synonyms in synonym_dict.items():
                if synonyms and user_keyword in synonyms.split(", "):
                    combined_keywords.add(keyword)
                    combined_keywords.update(synonyms.split(", "))

    except Exception as e:
        return (f"Error querying synonyms: {e}")

    # Convert to list, split phrases, and make all lowercase

    combined_keywords = list(combined_keywords)
    separated_keywords = []

    for keyword in combined_keywords:
        separated_keywords.extend(keyword.lower().split())

    return separated_keywords


def validate_keywords_with_llm(user_query, combined_keywords, client):
    """
    Validates the combined keywords for database query relevance using an LLM.

    Args:
        user_query (str): The user's input query.
        combined_keywords (list): The list of combined keywords from user query and synonyms.

    Returns:
        dict: A dictionary containing the LLM's response with validation or suggestions.
    """
    # Convert the list of combined keywords to a comma-separated string for the prompt
    keyword_string = ", ".join(combined_keywords)
    
    # Define the LLM prompt
    prompt = f"""
    You are an intelligent assistant trained to validate keywords for querying an SQL database for API parameters.
    The database contains keywords derived from API parameter descriptions, which define the type of data returned for each API call.

    Your task:
    1. Analyze the user's query and the provided combined list of keywords.
    2. Validate whether the keywords match the context of the user's query and are appropriate for querying the database.
    3. Suggest any corrections, additions, or removals to improve the relevance of the keyword list.
    4. Provide a summary indicating whether the list is valid or requires adjustments.

    Here is the input:
    User Query: "{user_query}"
    Combined Keywords: [{keyword_string}]

    Instructions:
        - Return the output as a JSON object only.
        - Do not include any formatting markers such as ```json, or comments.
        - The response must strictly adhere to the following JSON format:
    {{
        "is_valid": <true or false>,
        "suggestions": ["<suggested_keyword1>", "<suggested_keyword2>", ...],
        "summary": "<summary of validation>"
    }}
    """

    # Call the OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
        )
        
        # Extract the LLM's response
        llm_output = response.choices[0].message["content"]
        return json.loads(llm_output)

    except Exception as e:
        return {"error": str(e)}


def query_api_table(categories, keywords, conn):
    """
    Queries the SQL database for matching rows based on categories and keywords.

    Args:
        categories (list): List of high-level categories (e.g., "Population", "Income").
        keywords (list): List of keywords from the user's query.

    Returns:
        pandas.DataFrame: DataFrame containing matching rows from the database.
    """
    # Normalize inputs for SQL
    category_placeholders = ', '.join(['?'] * len(categories))
    keyword_conditions = ' OR '.join([
        f"LOWER(keywords) LIKE ?" for _ in keywords
    ])

    # Prepare SQL query
    query = f"""
    SELECT name, label, concept, category
    FROM api
    WHERE category IN ({category_placeholders})
      AND ({keyword_conditions})
    """

    # Prepare parameters
    keyword_params = [f"%{keyword.lower()}%" for keyword in keywords]  # No spaces around keywords
    params = categories + keyword_params

    # Execute query
    cursor = conn.cursor()
    cursor.execute(query, params)

    # Fetch results into a DataFrame
    results = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    results_list = [list(row) for row in results]

    return pd.DataFrame(results_list, columns=columns)

def filter_api_codes_with_llm(client, user_query, api_codes_df):
    """
    Filters API codes based on relevance to the user's query using an LLM.

    Args:
        user_query (str): The user's query.
        api_codes_df (pd.DataFrame): DataFrame containing API codes and their metadata.

    Returns:
        pd.DataFrame: Filtered DataFrame containing the most relevant API codes.
    """
    # Convert the DataFrame to a list of dictionaries
    api_codes_descriptions = api_codes_df.to_dict(orient="records")

    # Prepare the LLM prompt
    prompt = f"""
    You are a data assistant tasked with analyzing API codes and their descriptions to determine if any are relevant to answering a user's query. You should evaluate the intention of the user's query 
    and determine if the API codes are relevant to their intention. This prompt is being used recursively over a large number of potential API codes and so it is not crucial that you are strict in your determination.

    User Query: "{user_query}"

    Below is a list of API codes and their associated information. Filter the API codes to include only those that are supportive to answer the user's query. Exclude any codes that are not necessary or redundant.

    API Codes Information:
    {', '.join([f"Code: {item['name']}, Label: {item['label']}, Concept: {item['concept']}, Category: {item['category']}" for item in api_codes_descriptions])}

    Provide the output as a JSON array of the most relevant API codes. Do not include any comments, notes, or explanations in the output. Ensure the JSON is well-formed and strictly adheres to this format:
    [
        {{"name": "API_CODE_1"}},
        {{"name": "API_CODE_2"}}
    ]
    """

    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant trained to analyze and filter API codes based on relevance."},
                {"role": "user", "content": prompt},
            ],
        )

        # Extract the LLM's content
        llm_output = response.choices[0].message.content
        # print("LLM Raw Output:", llm_output)  # Debugging: Inspect the raw output

        # Remove triple backticks and process JSON
        llm_output_cleaned = llm_output.strip("```json").strip("```").strip()
        # print("LLM Cleaned Output:", llm_output_cleaned)  # Debugging: Inspect cleaned output

        # Parse the JSON response
        filtered_codes = json.loads(llm_output_cleaned)

        # Filter the DataFrame to include only the relevant API codes
        filtered_df = api_codes_df[api_codes_df["name"].isin([item["name"] for item in filtered_codes])]
        return filtered_df

    except json.JSONDecodeError as e:
        #print(f"Error while parsing LLM output: {e}")
        #print("Raw LLM Output for Debugging:", llm_output)  # Log raw output for debugging
        return pd.DataFrame()  # Return an empty DataFrame in case of JSON parsing errors

    except Exception as e:
        #print(f"Unexpected error while filtering API codes: {e}")
        return pd.DataFrame()  # Return an empty DataFrame for other errors

def chunk_dataframe(df, chunk_size):
    """
    Splits a Pandas DataFrame into chunks of a specified size.

    Args:
        df (pd.DataFrame): The DataFrame to be chunked.
        chunk_size (int): The maximum number of rows per chunk.

    Returns:
        list of pd.DataFrame: List of DataFrame chunks.
    """
    return [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]


def recursive_filter_api_codes(user_query, api_codes_df, client, chunk_size=300, max_final_size=30):
    """
    Improved recursive filtering of API codes using an LLM.
    """
    
    # Base case: if the DataFrame size is below or equal to the target, return it
    if len(api_codes_df) <= max_final_size:
        return api_codes_df

    # Chunk the DataFrame for initial filtering
    chunks = chunk_dataframe(api_codes_df, chunk_size)
    filtered_dfs = []

    for i, chunk in enumerate(chunks):
        filtered_chunk = filter_api_codes_with_llm(client, user_query, chunk)
        
        if not filtered_chunk.empty:
            filtered_dfs.append(filtered_chunk)

    # Combine all filtered chunks
    if filtered_dfs:
        filtered_combined_df = pd.concat(filtered_dfs, ignore_index=True)

    else:
        logging.warning("No matches found in any chunks.")
        return pd.DataFrame()  # Return an empty DataFrame if no matches

    # Recurse to refine further if the combined result is still too large
    return recursive_filter_api_codes(user_query, filtered_combined_df, client, chunk_size, max_final_size)


def rename_api_labels_with_llm(user_query, api_codes_df, client):
    """
    Renames API labels to concise, user-friendly names using an LLM.
    
    Args:
        user_query (str): The user's query to provide context for renaming.
        api_codes_df (pd.DataFrame): DataFrame containing 'name', 'label', 'concept', and 'category' columns.

    Returns:
        pd.DataFrame: DataFrame with an additional 'user_friendly_label' column containing concise labels.
    """
    # Convert the DataFrame to a list of dictionaries for the LLM
    api_codes_descriptions = api_codes_df.to_dict(orient="records")

    # Prepare LLM prompt
    prompt = f"""
    You are a data assistant tasked with creating concise, user-friendly names for API labels based on the user's query.

    User Query: "{user_query}"

    Below is a list of API codes and their details. Your job is to:
    - Rename the labels to be concise, clear, and suitable for display in a table or chart.
    - Ensure the names are intuitive and related to the data described in the label.
    - Consider the context of the user's query to make the labels more relevant.
    - Be careful when renaming the Total Population label. This can confuse the total population with demographic population and lead to inaccurate results.
    - Avoid using the word 'Estimate'

    API Codes Details:
    {', '.join([f"Code: {item['name']}, Label: {item['label']}, Concept: {item['concept']}, Category: {item['category']}" for item in api_codes_descriptions])}

    Provide the output as a JSON array with each entry containing:
    - 'code': The API code.
    - 'user_friendly_label': The concise, user-friendly name for the label.

    Instructions:
        - Return the output as a JSON object only.
        - Do not include any formatting markers such as ```json, or comments.
        - The response must strictly adhere to the following JSON format:
    [
        {{"code": "API_CODE_1", "user_friendly_label": "Concise Label 1"}},
        {{"code": "API_CODE_2", "user_friendly_label": "Concise Label 2"}}
    ]
    """

    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that renames API labels to concise and user-friendly names."},
                {"role": "user", "content": prompt},
            ],
        )
        
        # Extract and clean the LLM response content
        raw_content = response.choices[0].message.content

        # Strip Markdown-style code block if present
        if raw_content.startswith("```json"):
            raw_content = raw_content.strip("```json").strip("```").strip()
        
        # Parse the cleaned LLM response
        renamed_labels = json.loads(raw_content)

        # Convert the renamed labels to a DataFrame
        renamed_labels_df = pd.DataFrame(renamed_labels)

        # Merge with the original DataFrame
        result_df = api_codes_df.merge(
            renamed_labels_df, left_on='name', right_on='code', how='left'
        ).drop(columns=['code'])

        return result_df

    except json.JSONDecodeError as decode_error:
        #print(f"JSON decode error: {decode_error}")
        #print(f"LLM Response Content: {response.choices[0].message.content}")
        return api_codes_df.assign(user_friendly_label="Error generating label")
    except Exception as e:
        #print(f"Error while renaming API labels: {e}")
        return api_codes_df.assign(user_friendly_label="Error generating label")


def build_API_calls(location_codes, api_codes):
    """
    Builds the API calls for each location.
    
    Args:
        location_codes: list of tuples (type, PLACEFP/COUNTYFP, STATEFP)
        api_codes: list of API codes to query

    Returns:
        list of API call strings
    """
    base_url = 'https://api.census.gov/data/2022/acs/acs5?get='
    api_list = ','.join(api_codes)  # Build comma-separated list of API codes
    
    api_call_list = []
    
    for location in location_codes:
        location_type, code, state_fp = location
        
        if location_type == 'state':
            # State-level query
            location_param = f'for=state:{state_fp}'
        elif location_type == 'county':
            # County-level query
            location_param = f'for=county:{code}&in=state:{state_fp}'
        elif location_type == 'city':
            # City (place)-level query
            location_param = f'for=place:{code}&in=state:{state_fp}'
        else:
            raise ValueError(f"Unsupported location type: {location_type}")
        
        # Build full API call
        api_call = f"{base_url}{api_list}&{location_param}"
        api_call_list.append(api_call)
    
    return api_call_list


def fetch_acs_data(api_url):
    """
    Calls the ACS API, retrieves data, and returns it as a Pandas DataFrame.

    Args:
        api_url (str): The API URL to call.

    Returns:
        pd.DataFrame: Processed data as a DataFrame, or an empty DataFrame if the call fails.
    """
    try:
        # Make the API call
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        
        # Parse JSON response
        raw_data = response.json()
        
        # Validate the response structure
        if not raw_data or len(raw_data) < 2:
            #print("Invalid or empty data received.")
            return pd.DataFrame()

        # First row contains column names, remaining rows contain the data
        columns = raw_data[0]
        rows = raw_data[1:]

        # Create and return the DataFrame
        df = pd.DataFrame(rows, columns=columns)
        
        # Convert numeric columns to appropriate types (adjust as needed)
        numeric_columns = df.columns[1:]  # Assuming the first column is non-numeric
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    except requests.exceptions.RequestException as e:
        #print(f"Error fetching data: {e}")
        return pd.DataFrame()
    except ValueError as e:
        #print(f"Error processing data: {e}")
        return pd.DataFrame()

def combine_location_data(locations, location_codes):
    """
    Combines locations and location codes into a single list of dictionaries,
    and adds a location_name key based on location type.

    Args:
        locations (list): List of dictionaries containing location details (e.g., name, type, state).
        location_codes (list): List of tuples containing location type, location code, and state code.

    Returns:
        list: Combined list of dictionaries with added state code, location code, and location name.
    """
    combined_locations = []

    # Iterate through both lists in parallel
    for loc, code in zip(locations, location_codes):
        loc_type, location_code, state_code = code
        
        # Create a copy of the original location to avoid modifying the input
        combined_location = loc.copy()
        
        # Add the location code and state code to the location dictionary
        combined_location['state code'] = state_code
        combined_location['location code'] = location_code if location_code else None
        
        # Determine the location_name based on location type
        if loc_type == 'state':
            combined_location['location_name'] = loc['state'].title()
        else:
            combined_location['location_name'] = f"{loc['name'].title()}, {loc['state'].title()}"

        combined_locations.append(combined_location)

    return combined_locations


def update_dataframe_columns(api_df, friendly_df, locations):
    """
    Updates the API dataframe with user-friendly column names and a unified location column,
    and restructures it to have one row per metric and one column per location.

    Args:
        api_df (pd.DataFrame): DataFrame returned from the API call.
        friendly_df (pd.DataFrame): DataFrame containing 'name' and 'user_friendly_label' columns.
        locations (list): List of dictionaries with location details.

    Returns:
        pd.DataFrame: Restructured DataFrame with "Metric" and "Value" columns and one column per location.
    """
    # Step 1: Map API codes to friendly labels
    friendly_label_map = dict(zip(friendly_df['name'], friendly_df['user_friendly_label']))
    api_df.rename(columns=friendly_label_map, inplace=True)

    # Step 2: Add 'location_name' column to api_df
    location_names = [loc['location_name'] for loc in locations]
    if len(location_names) == len(api_df):
        api_df['location_name'] = location_names
    else:
        raise ValueError("Ruh roh. Ran into some data issues. Can you try again?")

    # Step 3: Drop unwanted columns
    api_df = api_df.drop(columns=['state', 'place', 'county'], errors='ignore')

    # Step 4: Melt the DataFrame to create 'Metric' and 'Value' columns
    melted_df = api_df.melt(
        id_vars=['location_name'],  # Columns to keep fixed
        var_name='Metric',         # New column for the metric names
        value_name='Value'         # New column for the metric values
    )

    # Step 5: Pivot the DataFrame to have one column per location
    pivot_df = melted_df.pivot(
        index='Metric',            # Metrics become rows
        columns='location_name',   # Locations become columns
        values='Value'             # Values populate the table
    ).reset_index()

    # Step 6: Clean up and return the DataFrame
    pivot_df.columns.name = None  # Remove the column name for a cleaner table
    return pivot_df


def generate_result_title(user_query, client):
    prompt = f"""
    You are a helpful assistant trained to analyze user queries and generate concise, descriptive titles for data results.

    Your task is to:
    1. Understand the intent of the user's query.
    2. Generate a clear and concise title that reflects the main content of the results.
    3. Ensure the title is engaging, user-friendly, and appropriate for display in a results section of a web application.
    4. Keep the title simple.

    Here is the user's query:
    "{user_query}"

    Instructions:
    - The title should summarize the main subject of the query in 10 words or less.
    - Focus on the key topics, metrics, and locations mentioned in the query.
    - Avoid generic phrases like "Results for Query" and instead be specific.

    Generate the title for this query and return it in plain text only, without any additional formatting or explanation.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Results"

def prep_results(df):
    """
    Formats numbers in a DataFrame with commas, removes rows with all empty or zero values (excluding the first column),
    and sorts rows by the metric name (first column).
    Assumes the first row and first column contain non-numeric data 
    (e.g., headers and row labels) and should not be modified.

    Args:
        df (pd.DataFrame): The DataFrame to format.

    Returns:
        pd.DataFrame: A new DataFrame with numbers formatted with commas, sorted by the first column.
    """
    try:

        # Create a copy of the DataFrame to avoid modifying the original
        formatted_df = df.copy()

        # Remove rows where all values (excluding the first column) are empty or zero
        non_metric_columns = formatted_df.columns[1:]
        formatted_df = formatted_df[~formatted_df[non_metric_columns].apply(
            lambda row: all((value == 0 or pd.isna(value)) for value in row), axis=1
        )]

        # Iterate over rows and columns, skipping the first column
        for col in formatted_df.columns[1:]:
            try:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{int(x):,}" if isinstance(x, (int, float)) else 
                              f"{int(float(x)):,}" if isinstance(x, str) and x.replace('.', '', 1).replace('-', '', 1).isdigit() else x
                )
            except Exception as col_error:
                logging.error("Error formatting column '%s': %s", col, col_error)


        # Sort the DataFrame by the first column (assumed to be the metric name)
        formatted_df = formatted_df.sort_values(by=formatted_df.columns[0])
        return formatted_df

    except Exception as e:
        raise  # Re-raise the exception after logging

def remove_redundant_rows(df):
    """
    Removes redundant rows from the DataFrame based on repeated values in the second column,
    while keeping one representative metric.

    Args:
        df (pd.DataFrame): The input DataFrame with potential redundant rows.

    Returns:
        pd.DataFrame: A DataFrame with redundant rows removed.
    """
    # Ensure the DataFrame has at least two columns
    if len(df.columns) < 2:
        raise ValueError("DataFrame must have at least two columns.")

    # Sort the DataFrame by the first column (Metric) to retain a logical order
    df = df.sort_values(by=df.columns[0])

    # Drop duplicate values in the second column, keeping the first occurrence
    df_cleaned = df.drop_duplicates(subset=[df.columns[1]])

    return df_cleaned



# Application Workflow
def process_query(user_query, conn, client):
    """
    Processes a user query and returns a standardized response message.
    Steps:
    1. Validate the query.
    2. If valid, extract data and process the query.
    3. Return either the results or an error message.

    Args:
        query (str): The user's query.

    Returns:
        dict: A standardized response with status, message, and optional data.
    """
 
    try:
        # ---- Validate the query using the LLM
        validation_result = validate_query_with_llm(user_query, client)
        
        # Handle different statuses
        if validation_result["status"] == "pass":
            # Proceed with the original valid query
            refined_query = user_query

        elif validation_result["status"] == "refined":
            # Use the refined query if available
            user_query = validation_result["refined_query"]
        
        else:
            # If validation fails and no refinement is possible, return the failure message
            return {
                "status": "error",
                "message": validation_result["message"],
                "data": None
            }

        # ----  Extract locations and other data
        classify_result = classify_query_with_llm(user_query, client)
        locations = classify_result['locations']
        keywords_user = classify_result['keywords']
        
        # Check for valid locations 
        if not locations:
            return {
                "status": "error",
                "message": "No valid locations found in the query.",
                "data": None
            }

        # Check for County word in county type name and proper State name
        for location in locations:
            location_type = location["type"].lower()
            location_name = location["name"].lower().strip()

            if location_type == 'county' and not location_name.lower().endswith('county'):
                location['name'] = f"{location['name']} County"

            if location_type == 'state' and location['state'] == 'state':
                location['state'] = location['name']

        # Get ACS location codes from locations
        location_codes = get_location_codes(locations, conn)
        
        # ----  Ensure location codes were found

        # Iterate over the location codes to validate based on location type
        for code in location_codes:
            location_type, place_or_county_fp, state_fp = code

            # For 'city' or 'county', PLACEFP or COUNTYFP must not be None
            if location_type in {"city", "county"} and place_or_county_fp is None:
                return {
                    "status": "error",
                    "message": f"Unable to properly identify all the locations in your query. Please check spelling and try again.",
                    "data": None
                }

            # For 'state', only STATEFP is required
            if location_type == "state" and state_fp is None:
                return {
                    "status": "error",
                    "message": "Issue locating the state in you query. Only the 50 US states are included in this tool. Please use a correct state name or proper abbreviation",
                    "data": None
                }
            
        # ---- Determine API codes
      
        # First, determine the best categories. Combine the LLM determined categories and user query determined categories
        
        categories_db_result = get_categories(user_query, client)
        categories_combined = categories_db_result['categories']
      
        keywords_combined = get_combined_keywords(keywords_user, conn)

        api_candidates = query_api_table(categories_combined, keywords_combined, conn)

        api_final_list = recursive_filter_api_codes(user_query, api_candidates, client)

        api_friendly = rename_api_labels_with_llm(user_query, api_final_list, client)
        api_codes = api_friendly['name'].tolist()

        # ---- Build API calls
        api_urls = build_API_calls(location_codes, api_codes)

        # ---- Make API Calls and combine data

        # ---- Make API Calls and combine data
        dfs = [fetch_acs_data(url) for url in api_urls]

        # Concatenate all DataFrames into one
        concat_df = pd.concat(dfs, ignore_index=True)

        # ---- Update Dataframe with friendly names and location names
        locations_final = combine_location_data(locations, location_codes)

        pivoted_df = update_dataframe_columns(concat_df, api_friendly, locations_final)
        
        # ---- Generate result title and return structured data
        result_title = generate_result_title(user_query, client)

        # Add commas to the numbers
        final_df = prep_results(pivoted_df)

        final_df = remove_redundant_rows(final_df)

        return {
            "status": "success",
            "message": "Query processed successfully.",
            "title": result_title,
            "data": final_df.to_json(orient="records")
}

    except Exception as e:
        # Handle unexpected errors
        return {
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}",
            "data": None
        }

